import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from argparse import ArgumentParser
import os
import numpy as np

import torch.nn.functional as F

from diffusers import DDPMPipeline, UNet2DModel
from diffusers.utils.pil_utils import make_image_grid
from torch.utils.data import DataLoader
from PIL import Image

from accelerate import notebook_launcher
from gen_ces_utils import generate_ces
import configs


def get_ims_npy(x):
    x = (x + 1.) / 2. # un-center
    return (x.clamp(0, 1).permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()


def sample_with_guidance(config, model, scheduler, conds, guidance=5.):
    samples_shape = (len(conds), config.image_chan, config.image_size, config.image_size)
    noise = torch.randn(samples_shape, device="cuda") * scheduler.init_noise_sigma

    scheduler.set_timesteps(
            config.num_inference_steps
        )

    z = noise

    conds_guide = torch.cat([conds, torch.full_like(conds, config.class_condition_dim)])

    for t in scheduler.timesteps:
        with torch.no_grad():
            z_guide = scheduler.scale_model_input(torch.cat([z, z]), timestep=t)
            noisy_residual = model(z_guide, t, class_labels=conds_guide).sample
        pred_cond, pred_uncond = noisy_residual.chunk(2)
        
        pred = pred_cond + guidance * (pred_cond - pred_uncond)
        z = scheduler.step(pred, t, z).prev_sample

    return z

def evaluate(config, epoch, model, scheduler):

    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    
    x = sample_with_guidance(config, model, scheduler, torch.arange(16, device="cuda") % 10, guidance=5.)
    x_npy = get_ims_npy(x)
    
    images = []
    for x_elem in x_npy:
        images.append(Image.fromarray(x_elem))

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")

    os.makedirs(test_dir, exist_ok=True)

    image_grid.save(f"{test_dir}/{epoch:04d}.png")

# from https://huggingface.co/docs/diffusers/en/tutorials/basic_training
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:

        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):

        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process or True)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):

            clean_images, labels = batch
            clean_images = clean_images.cuda()
            labels = labels.cuda()

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            ## Get Conditioning information for the UNet. randomly drop the conditioning information; set to n_dims
            lab_drop = torch.empty(labels.shape, device=labels.device).uniform_()
            labels = labels.where(lab_drop > cfg.cond_drop_rate, torch.full_like(labels, cfg.class_condition_dim))
            

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, class_labels=labels, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model

        if accelerator.is_main_process:

            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, model, noise_scheduler)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--config", type=str, choices=["cifar10", "svhn", "stl10"])
    parser.add_argument("--train", action="store_true", help="perform training with config settings")

    parser.add_argument("--ckpt_start", type=str, default=None, help="checkpoint name to initialize from")

    parser.add_argument("--num_processes", type=int, default=1, help="number of processes for training. should be device count")

    args = parser.parse_args()

    # select the config
    cfg = configs.utils.get_config(args.config)

    # create the directory if it doesn't exist
    exp_path = cfg.output_dir
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)

    # load in the dataset based on args
    train_dataset, test_dataset = configs.utils.get_dataset(cfg)

    sample_image = train_dataset[0][0].unsqueeze(0)
    print("Input shape:", sample_image.shape)

    # create dataloaders based on cfg
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False)

    # create model
    model = configs.utils.get_model(cfg)

    # TODO - optionally load a checkpoint

    # create the scheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    if args.train:
        # create the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

        from diffusers.optimization import get_cosine_schedule_with_warmup

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.num_epochs),
        )

        train_args = (cfg, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

        notebook_launcher(train_loop, train_args, num_processes=args.num_processes)

    # evaluate the diffusion model by generating some CEs
    
    # load the checkpointed diffusion model
    if args.ckpt_start is not None:
        print("Loading Checkpoint!")
        model = UNet2DModel.from_pretrained(exp_path, subfolder="unet")
        scheduler = DDPMScheduler.from_pretrained(exp_path, subfolder="scheduler")
        model.eval()
        model = model.cuda()

    # generate some samples as a sanity check
    
    x = sample_with_guidance(cfg, model, scheduler, torch.arange(16, device="cuda") % 10, guidance=5.).squeeze()
    x_npy = get_ims_npy(x)
    images = []
    for x_elem in x_npy:
        images.append(Image.fromarray(x_elem))
    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)
    # Save the images
    test_dir = os.path.join(cfg.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/eval_samples.png")

    ### create CEs of a given class from the image grid
    y = torch.full((x.shape[0], 1, 1), 5, dtype=torch.int).to(x.device)
    x_ces = generate_ces(cfg, model, scheduler, x, y, n_ces=1, guidance=15, ce_sigma=0.2, calc_likelihood=False)
    x_ces_npy = get_ims_npy(x_ces.squeeze())
    ces_images = []
    for x_elem in x_ces_npy:
        ces_images.append(Image.fromarray(x_elem))
    # Make a grid out of the images
    image_grid = make_image_grid(ces_images, rows=4, cols=4)
    # Save the images
    test_dir = os.path.join(cfg.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/eval_ce_samples.png")
    print("Done Sampling.")


    # Evaluate on Some Test Images
    begin = 15
    n_test = 5
    x_test, y_test = [], []
    for i in range(begin, begin+n_test):
        x, y = test_dataset[i]
        x_test.append(x[None, :, :, :])
        y_test.append(y)
    
    x_test = torch.cat(x_test)
    x_test_npy = get_ims_npy(x_test)
    x_test = x_test.cuda()
    x_ces, logits = generate_ces(cfg, model, scheduler, x_test, n_ces=3, guidance=15, ce_sigma=0.2, calc_likelihood=True)

    # prepare CEs for display
    x_ces_flat = x_ces.reshape((-1,) + x_ces.shape[-3:])
    x_ces_npy = np.reshape(get_ims_npy(x_ces_flat), x_ces.shape[:3] + x_ces.shape[-2:] + (x_ces.shape[3],))

    # prepare attributions for Display and yield reduced likelihoods
    probs = F.softmax(logits, dim=1)
    
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=n_test, ncols=1 + cfg.class_condition_dim)

    for i in range(n_test):
        axs[i][0].set_title("{}".format(y_test[i]))
        axs[i][0].imshow(x_test_npy[i])
        axs[i][0].xaxis.set_visible(False)
        axs[i][0].yaxis.set_visible(False)

        for j in range(cfg.class_condition_dim):
            axs[i][j+1].set_title("{:1.2f}".format(probs[i][j]))
            axs[i][j+1].imshow(x_ces_npy[i][j][0])
            axs[i][j+1].xaxis.set_visible(False)
            axs[i][j+1].yaxis.set_visible(False)

    fig.savefig(f"{test_dir}/eval_ces.png")


    # fig, axs = plt.subplots(nrows=n_test, ncols=1 + cfg.class_condition_dim)

    # for i in range(n_test):
    #     axs[i][0].set_title("{}".format(y_test[i]))
    #     axs[i][0].imshow(x_test_npy[i])
    #     axs[i][0].xaxis.set_visible(False)
    #     axs[i][0].yaxis.set_visible(False)

    #     for j in range(cfg.class_condition_dim):
    #         axs[i][j+1].set_title("{:1.2f}".format(probs[i][j]))
    #         axs[i][j+1].imshow(ll_sc_npy[i][j])
    #         axs[i][j+1].xaxis.set_visible(False)
    #         axs[i][j+1].yaxis.set_visible(False)

    # fig.savefig(f"{test_dir}/eval_ces_ll.png")
    
import torch
from configs.base_config import BaseTrainingConfig
from typing import Union, Tuple
from diffusers import DDPMScheduler, UNet2DModel
from torch.autograd import grad as grad_fn



# create routine to generate CEs of a given class using a scheduler, diffusion model, 

def generate_ces(
        cfg: BaseTrainingConfig,
        model: UNet2DModel,
        scheduler: DDPMScheduler,
        x: torch.FloatTensor,
        y: Union[torch.Tensor, None] = None,
        n_ces: int = 1,
        guidance: float = 0.,
        ce_sigma: float = 0.2,
        calc_likelihood: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create CEs with output shape (batch_size, n_class, n_ces, chan, height, width)
    """

    scheduler.set_timesteps(
        cfg.num_inference_steps
        )

    # assert defaults for now
    assert isinstance(model, UNet2DModel)
    assert isinstance(scheduler, DDPMScheduler)
    assert isinstance(n_ces, int) and n_ces > 0

    if y is None: # default to creating a CE for each class
        y = torch.arange(cfg.class_condition_dim, dtype=torch.long, device=x.device)[None, :, None].tile(
            x.shape[0], 1, n_ces
        )

    assert isinstance(y, torch.Tensor) and y.dim() == 3
    assert y.shape[0] == x.shape[0]
    assert x.dim() == 4

    # duplicate and flatten along batch dimension
    x_dup = x[:, None, None, :, :, :].tile(1, y.shape[1], n_ces, 1, 1, 1)
    ce_shape = x_dup.shape
    x_dup = x_dup.view((-1,) + x.shape[-3:]) # flatten along batch dimension

    y = torch.cat([y.view(-1), torch.full_like(y.view(-1), cfg.class_condition_dim)]) # duplicate for classifier-free guidance

    # initialize z with random noise, scale with scheduler
    z = torch.randn(x_dup.shape, device="cuda") * scheduler.init_noise_sigma

    # if calc_likelihood, create storage for pixel-level attributions. shape (batch_size, n_cond, C, H, W)
    if calc_likelihood: # initialize as the log-partition function of the prior. i.e., 0
        x_llike = torch.zeros_like(x_dup)

    for t_ind, t in enumerate(scheduler.timesteps):

        # assume that output type is 'epsilon'
        # calculate the CE score
        act = scheduler.alphas_cumprod[t]
        var = 1 - act
        g_var = var + ce_sigma**2
        scale = act**0.5
        g_mu = scale*x_dup
        ce_score = (g_mu - z) / (g_var)
        gauss_sc = 1. - t / scheduler.timesteps[0] # smooth transition from prior to density product distribution

        
        if calc_likelihood:
           z.requires_grad_(True)
           z_guide = scheduler.scale_model_input(torch.cat([z, z]), timestep=t)
           noisy_residual = model(z_guide, t, class_labels=y).sample
        else: # no grad
            with torch.no_grad():
                z_guide = scheduler.scale_model_input(torch.cat([z, z]), timestep=t)
                noisy_residual = model(z_guide, t, class_labels=y).sample
        pred_cond, pred_uncond = noisy_residual.chunk(2)
        pred = pred_cond + guidance * (pred_cond - pred_uncond)

        #### LIKELIHOOD CALCULATION ####
        if calc_likelihood and t_ind < len(scheduler.timesteps) - 1:
            t_prev = scheduler.timesteps[t_ind+1]
            gauss_sc_prev = 1. - t_prev / scheduler.timesteps[0]
            # estimate the log likelihood difference for the Gaussians, weighted by gauss_sc
            act_prev = scheduler.alphas_cumprod[t_prev]
            var_prev = (1. - act_prev)
            g_var_prev = var_prev + ce_sigma**2
            scale_prev = act_prev**0.5
            g_mu_prev = scale_prev*x_dup
            lld_gauss = gauss_sc_prev * (-0.5 * (g_var_prev.log() + (z - g_mu_prev).square().div(g_var_prev))) \
                - gauss_sc * (-0.5 * (g_var.log() + (z - g_mu).square().div(g_var)))

            # # estimate the log likelihood difference of DM
            dt = (t_prev - t) # / scheduler.timesteps[0] # should be negative TODO - double-check math here
            ode = -0.5 * scheduler.betas[t] * (z + -pred/(var**0.5)) # probability flow ode
            rad = torch.randint_like(ode, 2) * 2. - 1. # rademacher random variables
            div_prereduce = rad*grad_fn((rad*ode).sum(), z)[0] 
            z_prev_ode = z + ode*dt # flow the ode backwards in time (to lower noise level)
            # print((ode*dt).view(ode.shape[0], -1).norm(p=2, dim=1))
            # evaluate score at z_prev_ode with t_prev
            with torch.no_grad():
                z_prev_ode_guide = scheduler.scale_model_input(torch.cat([z_prev_ode, z_prev_ode]), timestep=t_prev)
                pred_ode_cond, pred_ode_uncond = model(z_prev_ode_guide, t_prev, class_labels=y).sample.chunk(2)
                pred_ode_prev = pred_ode_cond + guidance * (pred_ode_cond - pred_ode_uncond)
                score_prereduce = (z - z_prev_ode) * (-pred_ode_prev/(var_prev**0.5))
            
            x_llike = (x_llike + lld_gauss + div_prereduce*dt + score_prereduce).detach_()
                   
        ################################
        pred = pred - gauss_sc * (var**0.5) * ce_score

        z = scheduler.step(pred, t, z).prev_sample.detach_()
        # Add extra variance
        # z = z + torch.randn_like(z) * (scheduler._get_variance(t)**0.5)

        if t % 20 == 0:
            print(t.item())


    # reshape z
    z = z.reshape(ce_shape)
    if calc_likelihood:
       x_llike = x_llike.reshape(z.shape)
       return z, x_llike

    return z
    





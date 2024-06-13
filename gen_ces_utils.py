from accelerate import Accelerator

import torch
from configs.base_config import BaseTrainingConfig
from typing import Union, Tuple
from diffusers import DDPMScheduler, UNet2DModel
from torch.autograd import grad as grad_fn
from math import log, pi as PI
import numpy as np


# create routine to generate CEs of a given class using a scheduler, diffusion model, 

def generate_ces(
        accelerator: Accelerator,
        cfg: BaseTrainingConfig,
        model: UNet2DModel,
        scheduler: DDPMScheduler,
        x: torch.FloatTensor,
        y: Union[torch.Tensor, None] = None,
        n_ces: int = 1,
        guidance: float = 0.,
        ce_sigma: float = 0.2,
        calc_likelihood: bool = False,
        T: int = 1000
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create CEs with output shape (batch_size, n_class, n_ces, chan, height, width)
    """

    device = accelerator.device

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

    y = y.view(-1) # duplicate for classifier-free guidance
    y_uncond =  torch.full_like(y, cfg.class_condition_dim)

    # initialize z with random noise, scale with scheduler
    z = torch.randn(x_dup.shape, device=accelerator.device) * scheduler.init_noise_sigma

    N = np.prod(ce_shape[:3]) # batch_size x n_class x n_examples

    log_partition_diff = torch.zeros((N, len(scheduler.timesteps) - 1), device=z.device, dtype=torch.float) # base partition function (batch_size x n_class x n_examples, n_time - 1)

    # corral the data into a structure that will partition across processes correctly
    dist_inp = dict(
        z = z.reshape((-1,) + ce_shape[-3:]), # flatten along batch_dimension (batch_size x n_class x n_examples, LC, LH, LW)
        x_dup = x_dup.reshape((-1,) + ce_shape[-3:]), # flatten along batch_dimension (batch_size x n_class x n_examples, C, H, W)
        y = y.to(device), # (batch_size x n_class x n_examples, n_seq, n_feat)
        y_uncond = y_uncond.to(device), # (batch_size x n_class x n_examples, n_seq, n_feat)
        log_partition_diff = log_partition_diff # flatten along batch_dimension (batch_size x n_class x n_examples, n_time - 1)
    )


    accelerator.wait_for_everyone()

    # partition data with context manager
    with accelerator.split_between_processes(dist_inp, apply_padding=True) as inp:
        z, x_dup, y, y_uncond, log_partition_diff = inp["z"], inp["x_dup"], inp["y"], inp["y_uncond"], inp["log_partition_diff"]
        y = torch.cat([y, y_uncond], dim=0)
        
        for t_ind, t in enumerate(scheduler.timesteps):

            # assume that output type is 'epsilon'
            # calculate the CE score
            act = scheduler.alphas_cumprod[t]
            var = 1 - act
            g_var = ce_sigma**2
            g_mu = x_dup
            ce_score = (g_mu - z) / (g_var)
            gauss_sc = act # 1. - t / T # smooth transition from prior to density product distribution

            
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
                # estimate the log likelihood difference for the Gaussians, weighted by gauss_sc
                act_prev = scheduler.alphas_cumprod[t_prev]
                var_prev = (1. - act_prev)
                gauss_sc_prev = act_prev
                lld_gauss = (gauss_sc_prev - gauss_sc) * (-0.5 * (log(2.* PI) + log(g_var) + (z - g_mu).square().div(g_var))).sum(dim=(1,2,3))

                # estimate the log likelihood difference of DM
                dt = (t_prev - t) # / scheduler.timesteps[0] # should be negative TODO - double-check math here
                ode = -0.5 * scheduler.betas[t] * (z + -pred/(var**0.5)) # probability flow ode
                rad = torch.randint_like(ode, 2) * 2. - 1. # rademacher random variables
                div_prereduce = rad*grad_fn((rad*ode).sum(), z)[0] 
                z_prev_ode = z + ode*dt # flow the ode backwards in time (to lower noise level)
                # # evaluate score at z_prev_ode with t_prev
                with torch.no_grad():
                    z_prev_ode_guide = scheduler.scale_model_input(torch.cat([z_prev_ode, z_prev_ode]), timestep=t_prev)
                    pred_ode_cond, pred_ode_uncond = model(z_prev_ode_guide, t_prev, class_labels=y).sample.chunk(2)
                    pred_ode_prev = pred_ode_cond + guidance * (pred_ode_cond - pred_ode_uncond)
                    score_prereduce = (z - z_prev_ode) * (-pred_ode_prev/(var_prev**0.5))
                
                d_logits_prereduce = lld_gauss + div_prereduce.sum(dim=(1,2,3))*dt + score_prereduce.sum(dim=(1,2,3))
                log_partition_diff[:, t_ind] = d_logits_prereduce
                    
            ################################
            pred = pred - gauss_sc * (var**0.5) * ce_score

            z = scheduler.step(pred, t, z).prev_sample.detach_()

    # combine everything useful back together. truncate from padding during gather?
    accelerator.wait_for_everyone()
    z = accelerator.gather(z)[:N]
    log_partition_diff = accelerator.gather(log_partition_diff)[:N]
    log_partition_diff = log_partition_diff.reshape(ce_shape[:3] + (len(scheduler.timesteps) - 1,))
    log_partition_diff = log_partition_diff.logsumexp(dim=2) - np.log(n_ces)
    log_partition_diff = log_partition_diff.mean(dim=-1) # sum across all timesteps. yields (batch_size, n_class)
    # reshape z
    z = z.reshape(ce_shape)
    
    return z, log_partition_diff
    


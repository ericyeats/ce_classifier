import torch
from configs.base_config import BaseTrainingConfig
from typing import Union, Tuple
from diffusers import DDPMScheduler, UNet2DModel
from torch.autograd import grad as grad_fn
from math import log, pi as PI
import numpy as np


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
        calc_likelihood: bool = False,
        T: int = 1000
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
        logits = torch.zeros(ce_shape[:2], device=x_dup.device) # (batch_size, n_class, n_ces)

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
            # # print((ode*dt).view(ode.shape[0], -1).norm(p=2, dim=1))
            # # evaluate score at z_prev_ode with t_prev
            with torch.no_grad():
                z_prev_ode_guide = scheduler.scale_model_input(torch.cat([z_prev_ode, z_prev_ode]), timestep=t_prev)
                pred_ode_cond, pred_ode_uncond = model(z_prev_ode_guide, t_prev, class_labels=y).sample.chunk(2)
                pred_ode_prev = pred_ode_cond + guidance * (pred_ode_cond - pred_ode_uncond)
                score_prereduce = (z - z_prev_ode) * (-pred_ode_prev/(var_prev**0.5))
            
            d_logits_prereduce = lld_gauss + div_prereduce.sum(dim=(1,2,3))*dt + score_prereduce.sum(dim=(1,2,3))
            d_logits_prereduce = d_logits_prereduce.view(ce_shape[:3]) # (batch_size, n_class, n_ces)
            logits = (logits + torch.logsumexp(d_logits_prereduce, dim=2) - log(n_ces)).detach_()
                   
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
       return z, logits

    return z
    




# ##################### Fokker - Planck

# # create routine to generate CEs of a given class using a scheduler, diffusion model, 

# def generate_ces_fokker_planck(
#         cfg: BaseTrainingConfig,
#         model: UNet2DModel,
#         scheduler: DDPMScheduler,
#         x: torch.FloatTensor,
#         y: Union[torch.Tensor, None] = None,
#         n_ces: int = 1,
#         guidance: float = 0.,
#         ce_sigma: float = 0.2,
#         calc_likelihood: bool = False
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#     """
#     Create CEs with output shape (batch_size, n_class, n_ces, chan, height, width)
#     """

#     T = 1000

#     scheduler.set_timesteps(
#         cfg.num_inference_steps
#         )

#     # assert defaults for now
#     assert isinstance(model, UNet2DModel)
#     assert isinstance(scheduler, DDPMScheduler)
#     assert isinstance(n_ces, int) and n_ces > 0

#     if y is None: # default to creating a CE for each class
#         y = torch.arange(cfg.class_condition_dim, dtype=torch.long, device=x.device)[None, :, None].tile(
#             x.shape[0], 1, n_ces
#         )

#     assert isinstance(y, torch.Tensor) and y.dim() == 3
#     assert y.shape[0] == x.shape[0]
#     assert x.dim() == 4

#     # duplicate and flatten along batch dimension
#     x_dup = x[:, None, None, :, :, :].tile(1, y.shape[1], n_ces, 1, 1, 1)
#     ce_shape = x_dup.shape
#     M = np.prod(ce_shape[-3:])
#     x_dup = x_dup.view((-1,) + x.shape[-3:]) # flatten along batch dimension

#     y = torch.cat([y.view(-1), torch.full_like(y.view(-1), cfg.class_condition_dim)]) # duplicate for classifier-free guidance

#     # initialize z with random noise, scale with scheduler
#     z = torch.randn(x_dup.shape, device="cuda") * scheduler.init_noise_sigma

#     # if calc_likelihood, create storage for pixel-level attributions. shape (batch_size, n_cond, C, H, W)
#     if calc_likelihood: # initialize as the log-partition function of the prior. i.e., 0
#         logits = torch.zeros(ce_shape[:2], device=x_dup.device) # (batch_size, n_class,)

#     x_dup_dist = torch.distributions.Normal(loc=x_dup, scale=ce_sigma)

#     for t_ind, t in enumerate(scheduler.timesteps):

#         # assume that output type is 'epsilon'
#         # calculate the CE score
#         act = scheduler.alphas_cumprod[t]
#         var = 1 - act
#         g_var = ce_sigma**2
#         g_mu = x_dup
#         ce_score = (g_mu - z) / (g_var)
#         gauss_sc = 1. - t / T # smooth transition from prior to density product distribution

#         dlp = 0.
#         if calc_likelihood:
#            z.requires_grad_(True)
#            z_guide = scheduler.scale_model_input(torch.cat([z, z]), timestep=t)
#            noisy_residual = model(z_guide, t, class_labels=y).sample
#         else: # no grad
#             with torch.no_grad():
#                 z_guide = scheduler.scale_model_input(torch.cat([z, z]), timestep=t)
#                 noisy_residual = model(z_guide, t, class_labels=y).sample
#         pred_cond, pred_uncond = noisy_residual.chunk(2)
#         pred = pred_cond + guidance * (pred_cond - pred_uncond)

#         gauss_log_prob = x_dup_dist.log_prob(z)

#         #### LIKELIHOOD CALCULATION ####
#         if calc_likelihood and t_ind < len(scheduler.timesteps) - 1:
#             t_prev = scheduler.timesteps[t_ind+1]
#             dt = (t_prev - t) # / scheduler.timesteps[0] # should be negative TODO - double-check math here
#             score_guidance = -pred.float()/(var**0.5)
#             score_cond = -pred_cond.float()/(var**0.5)
#             score_uncond = -pred_uncond.float()/(var**0.5)
#             rad = torch.randint_like(score_guidance, 2) * 2. - 1. # rademacher random variables
#             div = torch.sum(rad*torch.autograd.grad((rad*score_guidance).sum(), z)[0], dim=(1, 2, 3))
#             fokker_planck = 0.5 * scheduler.betas[t] * (M + (score_guidance*z).sum(dim=(1,2,3)) + div \
#                                 + (guidance+1)*score_cond.square().sum(dim=(1,2,3)) - guidance*score_uncond.square().sum(dim=(1,2,3)))
#             dlp += fokker_planck*dt
#             # estimate the log likelihood difference for the Gaussians, weighted by gauss_sc
#             dlp += gauss_log_prob.float().sum(dim=(1,2,3)) * (t - t_prev) / T

#             logits += dlp.detach_().view(ce_shape[:3]).logsumexp(dim=2) - log(n_ces)

#         ################################
#         pred = pred - gauss_sc * (var**0.5) * ce_score

#         z = scheduler.step(pred, t, z).prev_sample.detach_()

#         if t % 20 == 0:
#             print(t.item())


#     # reshape z
#     z = z.reshape(ce_shape)
#     if calc_likelihood:
#        return z, logits

#     return z
import torch
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

def define_timesteps(base, num_inference_steps, device):
    # STEP 4: TIMESTEPS
    base.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    timesteps = base.scheduler.timesteps
    return timesteps

def adds(base, height, width, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
         device, batch_size, num_images_per_prompt, do_classifier_free_guidance):
    # STEP 7: PREPARE ADDED TIME IDS & EMBEDDINGS
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0,0)

    add_text_embeds = pooled_prompt_embeds

    add_time_ids = base._get_add_time_ids(
        original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype)

    
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    return prompt_embeds, add_text_embeds, add_time_ids

def adds_sld(base, height, width, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
         safety_embeds, safety_pooled_embeds,
         device, batch_size, num_images_per_prompt, do_classifier_free_guidance, enable_safety_guidance):
    # STEP 7: PREPARE ADDED TIME IDS & EMBEDDINGS
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0,0)
    negative_crops_coords_top_left = (0,0)

    add_text_embeds = pooled_prompt_embeds
    print(f'original_size {original_size}')
    print(f'crops_coords_top_left {crops_coords_top_left}')
    print(f'target_size {target_size}')
    print(f'prompt_embeds {prompt_embeds.dtype}')

    negative_original_size = None
    negative_target_size = None
    if negative_original_size is None:
            negative_original_size = original_size
    if negative_target_size is None:
            negative_target_size = target_size

    add_text_embeds = pooled_prompt_embeds

    if base.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
            text_encoder_projection_dim = base.text_encoder_2.config.projection_dim
    
    aesthetic_score = 6.0
    negative_aesthetic_score = 2.5
    add_time_ids = base._get_add_time_ids(
        original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,text_encoder_projection_dim=text_encoder_projection_dim)


    #add_text_embeds = pooled_prompt_embeds
    #if self.text_encoder_2 is None:
    #    text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    #else:
    #    text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

    #add_time_ids = self._get_add_time_ids(
    #        original_size,
    #        crops_coords_top_left,
    #        target_size,
    #        dtype=prompt_embeds.dtype,
    #        text_encoder_projection_dim=text_encoder_projection_dim,
       # )


        
    print(f'add_time_ids {add_time_ids}')
    #add_time_ids, add_neg_time_ids = base._get_add_time_ids(
    #        original_size,
    #        crops_coords_top_left,
    #        target_size,
    #        aesthetic_score,
    #        negative_aesthetic_score,
    #        negative_original_size,
    #        negative_crops_coords_top_left,
    #        negative_target_size,
    #        dtype=prompt_embeds.dtype,
    #        text_encoder_projection_dim=text_encoder_projection_dim,
    #    )
    


    if do_classifier_free_guidance:
        if enable_safety_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, safety_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds, safety_pooled_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids], dim=0)
        else:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    return prompt_embeds, add_text_embeds, add_time_ids


def define_cutoff_timesteps(base, timesteps, denoising_ratio):
        # 8.1 TIMESTEPS CUTOFF FOR DENOISING ON EACH EXPERT
        if denoising_ratio is not None and type(denoising_ratio) == float and denoising_ratio > 0 and denoising_ratio < 1:
                discrete_timestep_cutoff = int(round(
                        base.scheduler.config.num_train_timesteps - (denoising_ratio * base.scheduler.config.num_train_timesteps)))
                
                cutoff_num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
                cutoff_timesteps = timesteps[:cutoff_num_inference_steps]

        return cutoff_timesteps, cutoff_num_inference_steps


def denoising_loop(base, num_inference_steps, timesteps, cutoff_timesteps, cutoff_num_inference_steps, latents,
                   add_text_embeds, add_time_ids, prompt_embeds, output_type, guidance_scale, do_classifier_free_guidance):
    # local variable
    cross_attention_kwargs = None
    guidance_rescale = 0.0
    callback = None
    callback_steps = 1

    # STEP 6: PREPARE EXTRA STEP kwargs. I dont think it is used on SDXL default scheduler. eta applies only to DDIM scheduler.
    # "This prepares extra kwargs for the scheduler step, since not all schedulers have the same signature".
    extra_step_kwargs = base.prepare_extra_step_kwargs(generator=None, eta=0.0)

    # STEP 8.0 WARMUP STEPS - not sure how this is used.
    num_warmup_steps = max(len(timesteps) - num_inference_steps * base.scheduler.order, 0)

    # 8.2 DENOISING LOOP
    with torch.no_grad():
        with base.progress_bar(total=cutoff_num_inference_steps) as progress_bar:
            for i, t in enumerate(cutoff_timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = base.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = base.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = base.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(cutoff_timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % base.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

    latents = base.image_processor.postprocess(latents, output_type=output_type)
    SDXLPipeOutput = StableDiffusionXLPipelineOutput(images=latents)
    return SDXLPipeOutput, latents
import torch
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput


# Step 5 function: Timesteps
def prepare_refiner_timesteps(
    sdxl_refiner_pipeline,
    num_inference_steps,
    refiner_device,
    strength,
    denoising_start,
    batch_size,
    num_images_per_prompt,
):
    def denoising_value_valid(dnv):
        return isinstance(denoising_start, float) and 0 < dnv < 1

    sdxl_refiner_pipeline.scheduler.set_timesteps(
        num_inference_steps, device=refiner_device
    )
    timesteps, num_inference_steps = sdxl_refiner_pipeline.get_timesteps(
        num_inference_steps,
        strength,
        refiner_device,
        denoising_start=denoising_start if denoising_value_valid else None,
    )
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    add_noise = True if denoising_start is None else False

    return timesteps, num_inference_steps, latent_timestep, add_noise


# Dependency function for Step 8
def jk_get_add_time_ids(
    self,
    original_size,
    crops_coords_top_left,
    target_size,
    aesthetic_score,
    negative_aesthetic_score,
    negative_original_size,
    negative_crops_coords_top_left,
    negative_target_size,
    dtype,
):
    if self.config.requires_aesthetics_score:
        add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
        add_neg_time_ids = list(
            negative_original_size
            + negative_crops_coords_top_left
            + (negative_aesthetic_score,)
        )
    else:
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_neg_time_ids = list(
            negative_original_size + crops_coords_top_left + negative_target_size
        )

    passed_add_embed_dim = (
        self.unet.config.addition_time_embed_dim * len(add_time_ids)
        + self.text_encoder_2.config.projection_dim
    )
    expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

    if (
        expected_add_embed_dim > passed_add_embed_dim
        and (expected_add_embed_dim - passed_add_embed_dim)
        == self.unet.config.addition_time_embed_dim
    ):
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
        )
    elif (
        expected_add_embed_dim < passed_add_embed_dim
        and (passed_add_embed_dim - expected_add_embed_dim)
        == self.unet.config.addition_time_embed_dim
    ):
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
        )
    elif expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

    return add_time_ids, add_neg_time_ids


# Step 8 function: Adds
def prepare_refiner_embeddings(
    sdxl_refiner_pipeline,
    latents,
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
    refiner_device,
    batch_size,
    num_images_per_prompt,
    use_classifier_free_guidance,
):
    negative_original_size = None
    negative_target_size = None
    crops_coords_top_left = (0, 0)
    negative_crops_coords_top_left = (0, 0)
    aesthetic_score = 6.0
    negative_aesthetic_score = 2.5

    # STEP 7
    height, width = latents.shape[-2:]
    height = height * sdxl_refiner_pipeline.vae_scale_factor
    width = width * sdxl_refiner_pipeline.vae_scale_factor
    original_size = (height, width)
    target_size = (height, width)

    if negative_original_size is None:
        negative_original_size = original_size
    if negative_target_size is None:
        negative_target_size = target_size

    add_text_embeds = pooled_prompt_embeds
    add_time_ids, add_neg_time_ids = jk_get_add_time_ids(
        sdxl_refiner_pipeline,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype=prompt_embeds.dtype,
    )
    add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

    if use_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_neg_time_ids = add_neg_time_ids.repeat(
            batch_size * num_images_per_prompt, 1
        )
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(refiner_device)
    add_text_embeds = add_text_embeds.to(refiner_device)
    add_time_ids = add_time_ids.to(refiner_device)

    return prompt_embeds, add_text_embeds, add_time_ids


# Step 9 function: Denoising Loop
def execute_refiner_denoising(
    sdxl_refiner_pipeline,
    timesteps,
    num_inference_steps,
    denoising_start,
    latents,
    add_text_embeds,
    add_time_ids,
    prompt_embeds,
    use_classifier_free_guidance,
    cfg_guidance_scale,
):
    # local variable
    callback = None
    callback_steps = 1
    return_dict = True
    output_type = "pil"

    # 7. Prepare extra step kwargs.
    extra_step_kwargs = sdxl_refiner_pipeline.prepare_extra_step_kwargs(
        generator=None, eta=0.0
    )

    # 9. Denoising loop
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * sdxl_refiner_pipeline.scheduler.order, 0
    )

    with torch.no_grad():
        with sdxl_refiner_pipeline.progress_bar(
            total=num_inference_steps
        ) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if use_classifier_free_guidance
                    else latents
                )

                latent_model_input = sdxl_refiner_pipeline.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                noise_pred = sdxl_refiner_pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if use_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg_guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = sdxl_refiner_pipeline.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % sdxl_refiner_pipeline.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(
                            sdxl_refiner_pipeline.scheduler, "order", 1
                        )
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = (
                sdxl_refiner_pipeline.vae.dtype == torch.float16
                and sdxl_refiner_pipeline.vae.config.force_upcast
            )

            if needs_upcasting:
                sdxl_refiner_pipeline.upcast_vae()
                latents = latents.to(
                    next(
                        iter(sdxl_refiner_pipeline.vae.post_quant_conv.parameters())
                    ).dtype
                )

            image = sdxl_refiner_pipeline.vae.decode(
                latents / sdxl_refiner_pipeline.vae.config.scaling_factor,
                return_dict=False,
            )[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                sdxl_refiner_pipeline.vae.to(dtype=torch.float16)
        else:
            image = latents
            final_image = StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        if sdxl_refiner_pipeline.watermark is not None:
            image = sdxl_refiner_pipeline.watermark.apply_watermark(image)

        image = sdxl_refiner_pipeline.image_processor.postprocess(
            image, output_type=output_type
        )

        # Offload all models
        if (
            hasattr(sdxl_refiner_pipeline, "final_offload_hook")
            and sdxl_refiner_pipeline.final_offload_hook is not None
        ):
            sdxl_refiner_pipeline.final_offload_hook.offload()

        if not return_dict:
            final_image = (image,)

        final_image = StableDiffusionXLPipelineOutput(images=image)

    return final_image[0][0], latents

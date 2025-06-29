import torch

from debiasing_main.decodi_pipeline_base_breakdown import (
    adds_sld,
    define_cutoff_timesteps,
    define_timesteps,
)
from debiasing_main.decodi_pipeline_refiner_breakdown import (
    execute_refiner_denoising,
    prepare_refiner_embeddings,
    prepare_refiner_timesteps,
)


def sdxl_debias_cfg(
    base,
    refiner,
    prompt,
    num_inference_steps,
    denoising_ratio,
    guidance_scale,
    do_classifier_free_guidance,
    num_images_per_prompt,
    device,
    output_type,
    freeze_random_latents,
    decodi_text_concept,
    enable_decodi_guidance,
    decodi_guidance_scale,
    decodi_threshold,
    decodi_warmup_steps,
    decodi_momentum_scale,
    decodi_mom_beta,
    text_encoder_lora_scale,
    batch_size,
    refiner_device,
    denoising_start,
    strength,
    verbose=False,
):
    ################# BASE #################
    # STEP 0: Image Height and Width
    height = base.default_sample_size * base.vae_scale_factor
    width = base.default_sample_size * base.vae_scale_factor
    if verbose:
        print(f"height {height} and width {width}")

    # STEP 3: TEXT ENCODER
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = base.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        lora_scale=text_encoder_lora_scale,
    )

    if verbose and type(negative_prompt_embeds) is not type(None):
        print(f"""Prompt shapes: embeds {prompt_embeds.shape} | neg embeds {negative_prompt_embeds.shape} | pooled embeds {pooled_prompt_embeds.shape} | 
        neg pooled embeds {negative_pooled_prompt_embeds.shape}""")

    # STEP 3: BREAKDOWN 3 CREATE PROMPT ENCODER FOR decodi
    # Encode the decodi concept text
    if enable_decodi_guidance:
        decodi_concept_input = base.tokenizer(
            decodi_text_concept,
            padding="max_length",
            max_length=base.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        decodi_embeddings = base.text_encoder(
            decodi_concept_input.input_ids.to(base.device), output_hidden_states=True
        )
        decodi_embeddings = decodi_embeddings.hidden_states[-2]
        if verbose:
            print(f"decodi_embeddings: {decodi_embeddings.shape}")

        # duplicate decodi embeddings for each generation per prompt, using mps friendly method
        seq_len = decodi_embeddings.shape[1]
        decodi_embeddings = decodi_embeddings.repeat(
            batch_size, num_images_per_prompt, 1
        )
        if verbose:
            print(f"REPEAT decodi_embeddings: {decodi_embeddings.shape}")
        decodi_embeddings = decodi_embeddings.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        if verbose:
            print(f"VIEW decodi_embeddings: {decodi_embeddings.shape}")

        ########################## TOKENIZER_2 and ENCODER_2 ########################
        decodi_concept_input2 = base.tokenizer_2(
            decodi_text_concept,
            padding="max_length",
            max_length=base.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        decodi_embeddings2 = base.text_encoder_2(
            decodi_concept_input2.input_ids.to(base.device), output_hidden_states=True
        )

        ### POOLED FROM SECOND ENCODER ###
        decodi_pooled_embeds = decodi_embeddings2[0]
        if verbose:
            print(f"decodi_pooled_embeds: {decodi_pooled_embeds.shape}")
        ### POOLED FROM SECOND ENCODER ###

        decodi_embeddings2 = decodi_embeddings2.hidden_states[-2]
        if verbose:
            print(f"decodi_embeddings2: {decodi_embeddings2.shape}")

        # duplicate decodi embeddings for each generation per prompt, using mps friendly method
        seq_len = decodi_embeddings2.shape[1]
        decodi_embeddings2 = decodi_embeddings2.repeat(
            batch_size, num_images_per_prompt, 1
        )
        if verbose:
            print(f"REPEAT decodi_embeddings2: {decodi_embeddings2.shape}")
        decodi_embeddings2 = decodi_embeddings2.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        if verbose:
            print(f"VIEW decodi_embeddings2: {decodi_embeddings2.shape}")
        ########################## TOKENIZER_2 and ENCODER_2 ########################

        decodi_embeds = torch.cat([decodi_embeddings, decodi_embeddings2], dim=2)
        if verbose:
            print(f"CONCAT decodi_embeds: {decodi_embeds.shape}")
    else:
        decodi_embeds = None
        decodi_pooled_embeds = None

    # STEP 4: TIMESTEPS
    timesteps = define_timesteps(base, num_inference_steps, device)
    if verbose:
        print(f"Total timesteps {timesteps.shape}")

    # STEP 5: LATENT VARIABLES
    if not freeze_random_latents:
        num_channels_latents = base.unet.config.in_channels
        base_latents = base.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator=None,
            latents=None,
        )
        if verbose:
            print(f"Latent shape {base_latents.shape}")

    # STEP 7: PREPARE ADDED TIME IDS & EMBEDDINGS
    prompt_embeds, add_text_embeds, add_time_ids = adds_sld(
        base,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        decodi_embeds,
        decodi_pooled_embeds,
        device,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance,
        enable_decodi_guidance,
    )

    print(f"add_time_ids {add_time_ids}")
    if verbose:
        print(
            f"prompt_embeds {prompt_embeds.shape} | add_text_embeds {add_text_embeds.shape} | add_time_ids {add_time_ids.shape}"
        )

    # STEP 8.1: TIMESTEPS CUTOFF FOR DENOISING ON EACH EXPERT
    cutoff_timesteps, cutoff_num_inference_steps = define_cutoff_timesteps(
        base, timesteps, denoising_ratio
    )
    if verbose:
        print(f"Timesteps after cutoff {cutoff_timesteps.shape}")

    # 8.2 DENOISING LOOP
    from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

    latents = base_latents
    extra_step_kwargs = base.prepare_extra_step_kwargs(generator=None, eta=0.0)
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * base.scheduler.order, 0
    )
    callback = None
    callback_steps = 1

    with torch.no_grad():
        with base.progress_bar(total=cutoff_num_inference_steps) as progress_bar:
            for i, t in enumerate(cutoff_timesteps):
                if enable_decodi_guidance:
                    latent_model_input = (
                        torch.cat([latents] * 3)
                        if do_classifier_free_guidance
                        else latents
                    )
                else:
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )
                latent_model_input = base.scheduler.scale_model_input(
                    latent_model_input, t
                )

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                noise_pred = base.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                # print(f'FULL noise_pred: {noise_pred.shape}')

                # perform guidance
                if do_classifier_free_guidance:
                    if enable_decodi_guidance:
                        (
                            noise_pred_uncond,
                            noise_pred_text,
                            noise_pred_decodi_concept,
                        ) = noise_pred.chunk(3)
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    if enable_decodi_guidance:
                        # default classifier free guidance
                        noise_guidance = noise_pred_text - noise_pred_uncond
                        decodi_momentum = torch.zeros_like(noise_guidance)
                        # Equation 6
                        scale = torch.clamp(
                            torch.abs((noise_pred_text - noise_pred_decodi_concept))
                            * decodi_guidance_scale,
                            max=1.0,
                        )
                        # Equation 6
                        decodi_concept_scale = torch.where(
                            (noise_pred_text - noise_pred_decodi_concept)
                            >= decodi_threshold,
                            torch.zeros_like(scale),
                            scale,
                        )
                        # Equation 4
                        noise_guidance_decodi = torch.mul(
                            (noise_pred_decodi_concept - noise_pred_uncond),
                            decodi_concept_scale,
                        )
                        # Equation 7
                        noise_guidance_decodi = (
                            noise_guidance_decodi
                            + decodi_momentum_scale * decodi_momentum
                        )
                        # Equation 8
                        decodi_momentum = (
                            decodi_mom_beta * decodi_momentum
                            + (1 - decodi_mom_beta) * noise_guidance_decodi
                        )
                        if i >= decodi_warmup_steps:  # Warmup
                            # Equation 3
                            noise_guidance = noise_guidance - noise_guidance_decodi
                        noise_pred = noise_pred_uncond + guidance_scale * noise_guidance

                # compute the previous noisy sample x_t -> x_t-1
                latents = base.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(cutoff_timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % base.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

    latents = base.image_processor.postprocess(latents, output_type=output_type)
    SDXLPipeOutput = StableDiffusionXLPipelineOutput(images=latents)

    final_image_base = SDXLPipeOutput

    ################# REFINER #################
    # 3. Encode input prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = refiner.encode_prompt(
        prompt=prompt,
        device=refiner_device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        lora_scale=None,
    )

    if verbose and type(negative_prompt_embeds) is not type(None):
        print(f"""Prompt shapes: embeds {prompt_embeds.shape} | neg embeds {negative_prompt_embeds.shape}
                pooled embeds {pooled_prompt_embeds.shape} | neg pooled embeds {negative_pooled_prompt_embeds.shape}""")

    # 5. Prepare Refiner Timesteps
    timesteps, num_inference_steps, latent_timestep, add_noise = (
        prepare_refiner_timesteps(
            refiner,
            num_inference_steps,
            refiner_device,
            strength,
            denoising_start,
            batch_size,
            num_images_per_prompt,
        )
    )
    if verbose:
        print(
            f"Timesteps: {timesteps.shape} | num_inference_steps: {num_inference_steps} | latent_timestep: {latent_timestep[0]} | add_noise {add_noise}"
        )

    # 6. Prepare Latent variables
    image = refiner.image_processor.preprocess(final_image_base[0])
    latents = refiner.prepare_latents(
        image,
        latent_timestep,
        batch_size,
        num_images_per_prompt,
        prompt_embeds.dtype,
        refiner_device,
        generator=None,
        add_noise=add_noise,
    )
    if verbose:
        print(f"Latent shape {latents.shape}")

    # 8. Prepare added time ids & embeddings
    prompt_embeds, add_text_embeds, add_time_ids = prepare_refiner_embeddings(
        refiner,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        refiner_device,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance,
    )
    if verbose:
        print(
            f"add_text_embeds {add_text_embeds.shape} and add_time_ids {add_time_ids.shape}"
        )

    # Step 9 Denoising Loop
    final_image, latents = execute_refiner_denoising(
        refiner,
        timesteps,
        num_inference_steps,
        denoising_start,
        latents,
        add_text_embeds,
        add_time_ids,
        prompt_embeds,
        do_classifier_free_guidance,
        guidance_scale,
    )
    if verbose:
        print(f"Shape of Latents {latents.shape}")

    return final_image

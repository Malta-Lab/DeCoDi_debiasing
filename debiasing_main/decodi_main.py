import os
import pickle
import random
import sys

import numpy as np
import torch
from diffusers.pipelines import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

from debiasing_main.decodi_class_free_guidance import sdxl_debias_cfg

# Usage: python decodi_main.py <mode> <enable_safety_guidance> <default> [output_dir] [cuda_device] [num_images]
# Arguments:
#   mode: firefighter, nurse, or ceo
#   enable_safety_guidance: True or False (enable debiasing)
#   default: True or False (use default or custom hyperparameters)
#   output_dir: optional path to save images (default: base_dir/images)
#   cuda_device: optional CUDA device (default: cuda:0, e.g., cuda:1, cuda:2)
#   num_images: optional number of images to generate (default: all seeds in rnd_list.pkl)

# Defining generation mode
acceptable_modes = ["firefighter", "nurse", "ceo"]

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

cache_dir = os.path.join(base_dir, "sdxl_models")
os.makedirs(cache_dir, exist_ok=True)

if len(sys.argv) < 2 or sys.argv[1] not in acceptable_modes:
    print(f"Error! First argument must be one of the following: {acceptable_modes}")
    exit()

mode = sys.argv[1]  # gets the first argument from the command line

if len(sys.argv) > 2 and sys.argv[2].lower() in ["true", "false"]:
    enable_safety_guidance = sys.argv[2].lower() == "true"
else:
    print(
        'Error! Second argument must be "True" or "False" for enable_safety_guidance.'
    )
    exit()

if len(sys.argv) > 3 and sys.argv[3].lower() in ["true", "false"]:
    default = sys.argv[3].lower() == "true"
else:
    print('Error! Third argument must be "True" or "False" for default.')
    exit()

# Fourth argument: output directory (optional)
if len(sys.argv) > 4:
    output_dir = sys.argv[4]
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
else:
    output_dir = os.path.join(base_dir, "images")

# Fifth argument: CUDA device (optional, default is cuda:0)
if len(sys.argv) > 5:
    cuda_device = sys.argv[5]
    if not cuda_device.startswith("cuda:"):
        print(
            'Error! Fifth argument must be in format "cuda:X" (e.g., "cuda:0", "cuda:1")'
        )
        exit()
else:
    cuda_device = "cuda:0"

# Sixth argument: number of images to generate (optional)
if len(sys.argv) > 6:
    try:
        num_images_to_generate = int(sys.argv[6])
        if num_images_to_generate <= 0:
            print(
                "Error! Sixth argument must be a positive integer for number of images."
            )
            exit()
    except ValueError:
        print("Error! Sixth argument must be a valid integer for number of images.")
        exit()
else:
    num_images_to_generate = None  # Generate all images in the list


# Instanciating SDXL Expert: Base
base = None
try:
    base.device.type == "cuda"
    print("Model already on GPU!")
except AttributeError:
    print("Loading model to GPU...")
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=cache_dir,
    )
    base.to(cuda_device)

# Instanciating SDXL Expert: Refiner
refiner = None
try:
    refiner.device.type == "cuda"
    print("Model already on GPU!")
except AttributeError:
    print("Loading model to GPU...")
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=cache_dir,
    )
    refiner.to(cuda_device)

# BASE HYPERPARAMETERS
if mode == "firefighter":
    prompt = "the face of firefighter"
    safety_text_concept = "Caucasian, white, pale skin, Eurocentric, blonde, light eyes, light hair, fair complexion, European features, narrow nose, straight hair, light-skinned, Western traits, Nordic, Anglo-Saxon, European ethnicity"
    concept_type = "ethnicity"
elif mode == "nurse":
    prompt = "the face of a nurse"
    safety_text_concept = "woman, female, feminine, soft features, delicate, long hair, ponytail, makeup, slender build, gentle expression, nurturing demeanor, soft smile, maternal, traditional femininity, skirt, dress, apron, feminine posture, small hands, petite frame, caring woman, young female, lady nurse, lipstick, subtle makeup, earrings, jewelry"
    concept_type = "gender"
elif mode == "ceo":
    prompt = "the face of a ceo"
    safety_text_concept = "old, elderly, senior, wrinkles, gray hair, balding, receding hairline, aging, aged, mature face, crow's feet, sagging skin, white hair, glasses, stooped posture, frail, age spots, middle-aged, older man, older woman, senior executive, aging leader, late 50s, 60s, 70s, aged features"
    concept_type = "age"

if default:
    num_inference_steps = 50
    denoising_ratio = 0.99  # Defines what % of steps to be run on base expert
    guidance_scale = 7.5
    do_classifier_free_guidance = guidance_scale > 1.0
    num_images_per_prompt = 1
    device = base._execution_device
    output_type = "latent"  # 'pil' or 'latent'
    freeze_random_latents = False
    text_encoder_lora_scale = None
    batch_size = 1
    # DeCoDi parameters
    decodi_guidance_scale = 2000  # If < 1, safety guidance is disabled.
    decodi_threshold = 1  # Threshold that separates the hyperplane between appropriate and inappropriate images.
    decodi_warmup_steps = 7  # decodi is only be applied for diffusion steps greater than decodi_warmup_steps.
    decodi_momentum_scale = 0.5  # Scale of the decodi momentum to be added to the safety guidance at each diffusion step. If set to 0.0, momentum is disabled.
    decodi_mom_beta = 0.7  # Defines how safety guidance momentum builds up during warmup.Indicates how much of the previous momentum is kept.
    # REFINER HYPERPARAMETERS
    refiner_device = refiner._execution_device
    denoising_start = denoising_ratio
    strength = 0.01  # indicates how much to transform the reference 'image'. Adds more noise to the input image the larger the 'strength'.
    is_default = "default"
else:
    num_inference_steps = 50
    denoising_ratio = 0.99  # Defines what % of steps to be run on base expert
    guidance_scale = 12
    do_classifier_free_guidance = guidance_scale > 1.0
    num_images_per_prompt = 1
    device = base._execution_device
    output_type = "latent"  # 'pil' or 'latent'
    freeze_random_latents = False
    text_encoder_lora_scale = None
    batch_size = 1
    # DeCoDi parameters
    decodi_guidance_scale = 15000  # If < 1, safety guidance is disabled.
    decodi_threshold = 0.025  # Threshold that separates the hyperplane between appropriate and inappropriate images.
    decodi_warmup_steps = 7  # decodi is only be applied for diffusion steps greater than decodi_warmup_steps.
    decodi_momentum_scale = 0.5  # Scale of the decodi momentum to be added to the safety guidance at each diffusion step. If set to 0.0, momentum is disabled.
    decodi_mom_beta = 0.7  # Defines how safety guidance momentum builds up during warmup.Indicates how much of the previous momentum is kept.
    # REFINER HYPERPARAMETERS
    refiner_device = refiner._execution_device
    denoising_start = denoising_ratio
    strength = 0.01  # indicates how much to transform the reference 'image'. Adds more noise to the input image the larger the 'strength'.
    is_default = "custom"


# RUN DENOISING LOOP - KEEPING ORIGINAL SDXL

pickle_files_dir = os.path.join(base_dir, "pickle files")
pickle_file_path = os.path.join(pickle_files_dir, "rnd_list.pkl")

with open(pickle_file_path, "rb") as f:
    rnd_list = pickle.load(f)

# Limit the number of images to generate if specified
if num_images_to_generate is not None:
    rnd_list = rnd_list[:num_images_to_generate]
    print(f"Generating {len(rnd_list)} images (limited from total available)")
else:
    print(f"Generating all {len(rnd_list)} images from the seed list")

original_imgs_list = []

images_dir = output_dir

for i, rnd in enumerate(rnd_list):
    print(f"Generating image {i + 1}/{len(rnd_list)} (seed: {rnd})")

    # Control seeds
    torch.manual_seed(rnd)
    random.seed(rnd)
    np.random.seed(rnd)

    original_img = sdxl_debias_cfg(
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
        safety_text_concept,
        enable_safety_guidance,
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
    )

    mode_dir = os.path.join(
        images_dir,
        str(concept_type)
        + mode
        + "_safety_guidance_"
        + str(enable_safety_guidance)
        + "_"
        + is_default,
    )
    os.makedirs(mode_dir, exist_ok=True)

    if not enable_safety_guidance:
        name_image = "original"
    else:
        name_image = "debiased"

    file_name = f"{mode}_{name_image}_{is_default}_{rnd}.jpg"
    file_path = os.path.join(mode_dir, file_name)

    original_img.save(file_path, "JPEG")

    original_imgs_list.append(
        {
            "img_path": f"/static/images/{mode}/{file_name}",
            "rnd": rnd,
            "prompt": prompt,
            "safety_concept": safety_text_concept,
            "gender": "",
            "ethnicity": "",
            "apparent_age": "",
            "user_name": "",
        }
    )

with open(f"imgs_{concept_type}_{mode}_{name_image}.pkl", "wb") as f:
    pickle.dump(original_imgs_list, f)

import torch
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from diffusers.pipelines import StableDiffusionXLPipeline
from diffusers.pipelines import StableDiffusionXLImg2ImgPipeline
from diffusers.loaders import TextualInversionLoaderMixin
import sys
import os
import argparse
from sdxl_debias_class_free_guidance import sdxl_debias_cfg

# Defining generation mode
acceptable_modes = ['firefighter', 'nurse', 'business_leader']

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

cache_dir = os.path.join(base_dir,'sdxl_models')
os.makedirs(cache_dir,exist_ok=True)

if len(sys.argv) < 2 or sys.argv[1] not in acceptable_modes:
    print(f'Error! First argument must be one of the following: {acceptable_modes}')
    exit()

mode = sys.argv[1]  # gets the first argument from the command line

if len(sys.argv) > 2 and sys.argv[2].lower() in ['true', 'false']:
    enable_safety_guidance = sys.argv[2].lower() == 'true'
else:
    print('Error! Second argument must be "True" or "False" for enable_safety_guidance.')
    exit()

if len(sys.argv) > 3 and sys.argv[3].lower() in ['true', 'false']:
    default = sys.argv[3].lower() == 'true'
else:
    print('Error! Third argument must be "True" or "False" for default.')
    exit()


# Instanciating SDXL Expert: Base
try:
    base.device.type == 'cuda'
    print('Model already on GPU!')
except:
    print('Loading model to GPU...')
    base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, 
                                                variant="fp16", use_safetensors=True, cache_dir=cache_dir)
    base.to('cuda:0')

# Instanciating SDXL Expert: Refiner
try:
    refiner.device.type == 'cuda'
    print('Model already on GPU!')
except:
    print('Loading model to GPU...')
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", text_encoder_2=base.text_encoder_2,
        vae=base.vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir=cache_dir)
    refiner.to('cuda:0')

# BASE HYPERPARAMETERS
if mode == 'firefighter':
    prompt = "the face of firefighter"
    safety_text_concept = ("man traits, very manly male man, rougth man, man, male, man, male, man, male")
elif mode == 'nurse':
    prompt = "the face of a nurse"
    safety_text_concept = ("very womanly female woman, woman, female, woman, female, woman, female, woman, female")
elif mode == 'business_leader':
    prompt = "the face of a business leader"
    safety_text_concept = ("an old man, caucasian, male, man, man, man, male, man, male, man, manly male, man traits")
#prompt = "the face of firefighter" #1 "the face of firefighter" #2,3,4 "the face of a nurse" #5 "the face of a business leader"


if default == True:
    num_inference_steps = 50
    denoising_ratio = 0.99 # Defines what % of steps to be run on base expert
    guidance_scale = 5.0
    do_classifier_free_guidance = guidance_scale > 1.0
    num_images_per_prompt = 1
    device = base._execution_device
    output_type = 'latent' # 'pil' or 'latent'
    freeze_random_latents = False
    text_encoder_lora_scale = None
    batch_size = 1
    # SAFE LATENT DIFFUSION
    sld_guidance_scale = 1000 # If < 1, safety guidance is disabled.
    sld_threshold = 0.01 # Threshold that separates the hyperplane between appropriate and inappropriate images.
    sld_warmup_steps = 10 # SLD is only be applied for diffusion steps greater than sld_warmup_steps.
    sld_momentum_scale = 0.3 # Scale of the SLD momentum to be added to the safety guidance at each diffusion step. If set to 0.0, momentum is disabled.
    sld_mom_beta = 0.4 # Defines how safety guidance momentum builds up during warmup.Indicates how much of the previous momentum is kept.
    # REFINER HYPERPARAMETERS
    refiner_device = refiner._execution_device
    denoising_start = denoising_ratio
    strength = 0.01 #indicates how much to transform the reference 'image'. Adds more noise to the input image the larger the 'strength'.
    is_default = 'default'
else:
    num_inference_steps = 50
    denoising_ratio = 0.99 # Defines what % of steps to be run on base expert
    guidance_scale = 12
    do_classifier_free_guidance = guidance_scale > 1.0
    num_images_per_prompt = 1
    device = base._execution_device
    output_type = 'latent' # 'pil' or 'latent'
    freeze_random_latents = False
    text_encoder_lora_scale = None
    batch_size = 1
    # SAFE LATENT DIFFUSION
    sld_guidance_scale = 15000 # If < 1, safety guidance is disabled.
    sld_threshold = 0.025 # Threshold that separates the hyperplane between appropriate and inappropriate images.
    sld_warmup_steps = 7 # SLD is only be applied for diffusion steps greater than sld_warmup_steps.
    sld_momentum_scale = 0.5 # Scale of the SLD momentum to be added to the safety guidance at each diffusion step. If set to 0.0, momentum is disabled.
    sld_mom_beta = 0.7 # Defines how safety guidance momentum builds up during warmup.Indicates how much of the previous momentum is kept.
    # REFINER HYPERPARAMETERS
    refiner_device = refiner._execution_device
    denoising_start = denoising_ratio
    strength = 0.01 #indicates how much to transform the reference 'image'. Adds more noise to the input image the larger the 'strength'.
    is_default = 'jordan'


# RUN DENOISING LOOP - KEEPING ORIGINAL SDXL

pickle_files_dir = os.path.join(base_dir, 'pickle files')
pickle_file_path = os.path.join(pickle_files_dir, 'rnd_list_0_1000_50.pkl')

with open(pickle_file_path, 'rb') as f:
    rnd_list = pickle.load(f)

original_imgs_list = []

#'/home/jordankp/sdxl/UI_label_images/static/images/'
images_dir = os.path.join(base_dir, 'images')

for i, rnd in enumerate(rnd_list):
    # Control seeds
    torch.manual_seed(rnd)
    random.seed(rnd)
    np.random.seed(rnd)
    

    original_img = sdxl_debias_cfg(base, refiner, prompt, num_inference_steps, denoising_ratio, 
    guidance_scale, do_classifier_free_guidance, num_images_per_prompt, device, output_type, freeze_random_latents,
    safety_text_concept, enable_safety_guidance, sld_guidance_scale, sld_threshold, sld_warmup_steps, 
    sld_momentum_scale, sld_mom_beta, text_encoder_lora_scale, batch_size, refiner_device, 
    denoising_start, strength, verbose=False)
    
    mode_dir = os.path.join(images_dir, mode +"_safety_guidance_"+ str(enable_safety_guidance)+"_"+is_default)
    os.makedirs(mode_dir, exist_ok = True)

    if enable_safety_guidance == False:
        name_image = 'original'
    else:
        name_image = 'debiased'
        
    file_name = f"{mode}_{name_image}_{is_default}_{rnd}.jpg"
    file_path = os.path.join(mode_dir,file_name)

    original_img.save(file_path,'JPEG')

    original_imgs_list.append({
        'img_path': f'/static/images/{mode}/{file_name}',
        'rnd': rnd,
        'prompt': prompt,
        'safety_concept': safety_text_concept,
        'gender': '',
        'ethnicity': '',
        'apparent_age': '',
        'user_name': ''
    })


with open('original_imgs_'+mode+'.pkl', 'wb') as f:
    pickle.dump(original_imgs_list, f)
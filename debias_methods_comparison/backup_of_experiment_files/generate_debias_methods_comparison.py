from diffusers import DiffusionPipeline
import torch
import numpy as np
import random
from PIL import Image
from diffusers.utils import make_image_grid
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt


# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    cache_dir='/mnt/A-NVME/jordankp/sdxl_models/')
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    cache_dir='/mnt/A-NVME/jordankp/sdxl_models/')
refiner.to("cuda")
print('done')


# gets the list of random seeds used in CFG
for folder in os.listdir('/home/jordankp/sdxl/UI_label_images/labeled_images/'):
    for file in os.listdir('/home/jordankp/sdxl/UI_label_images/labeled_images/'+folder):
        #print('/home/jordankp/sdxl/UI_label_images/labeled_images/'+folder+'/'+file)
        with open('/home/jordankp/sdxl/UI_label_images/labeled_images/'+folder+'/'+file, 'rb') as f:
            imgs_list = pickle.load(f)
        
        if 'df' not in globals():
            df = pd.DataFrame(imgs_list)
        else:
            df = pd.concat([df, pd.DataFrame(imgs_list)], axis=0, ignore_index=True)
df.drop('index', axis=1, inplace=True)
df['orig_deb'] = df['img_path'].str.contains('original').replace(True,'original').replace(False,'debiased')

rnd_list = df['rnd'].unique()


# Define Hyper-Parameters
n_steps = 50
high_noise_frac = 0.99
generation_folder = 'firefighter'
prompt = 'the face of firefighter'
guidance_scale = 12
negative_prompt = 'man traits, very manly male man, rougth man, man, male, man, male, man, male'
prompt_eng = 'a single firefighter person, focused on face, female or male' # best prompt engineering to balance a single person between males and females
lookup_prompt = 'the face of a '+ random.choice(['male','female']) +' firefighter'

all_images = []
num_samples = 150

# selects X number of random seeds
rnd_list = rnd_list[0:num_samples]

for seed in rnd_list:
    for mode in range(5):
        if mode==0: # default prompt
            torch.manual_seed(seed)
            image = base(
                prompt=prompt,
                negative_prompt=None,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                guidance_scale=guidance_scale,
                output_type="latent"
            ).images
            image_default = refiner(
                prompt=prompt,
                negative_prompt=None,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image,
                guidance_scale=guidance_scale,
            ).images[0]
        elif mode==1: # negative prompt
            torch.manual_seed(seed)
            image = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                guidance_scale=guidance_scale,
                output_type="latent"
            ).images
            image_negative = refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image,
                guidance_scale=guidance_scale,
            ).images[0]
        elif mode==2: # prompt engineering
            torch.manual_seed(seed)
            image = base(
                prompt=prompt_eng,
                negative_prompt=None,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                guidance_scale=guidance_scale,
                output_type="latent"
            ).images
            image_prompt_eng = refiner(
                prompt=prompt_eng,
                negative_prompt=None,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image,
                guidance_scale=guidance_scale,
            ).images[0]
        elif mode==3: # Lookup table
            torch.manual_seed(seed)
            image = base(
                prompt=lookup_prompt,
                negative_prompt=None,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                guidance_scale=guidance_scale,
                output_type="latent"
            ).images
            image_lookup_table = refiner(
                prompt=lookup_prompt,
                negative_prompt=None,
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=image,
                guidance_scale=guidance_scale,
            ).images[0]
        elif mode==4:
            image_cfg = Image.open(
                '/home/jordankp/sdxl/UI_label_images/static/images/'+
                generation_folder+'/'+
                generation_folder+'_debiased_'
                +str(seed)+'.jpg')
    all_images.append([seed,image_default,image_cfg,image_negative,image_prompt_eng,image_lookup_table])

# Save list of images as a pickle file
with open('./all_images.pkl', 'wb') as file:
    pickle.dump(all_images, file)
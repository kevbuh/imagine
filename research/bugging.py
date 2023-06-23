from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import time
# model_id = "stabilityai/stable-diffusion-2"
# 48 mins with mps and without attention slicing

start_time = time.time()

# Use the Euler scheduler here instead
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cpu")

pipe.enable_attention_slicing()

prompt = "a photo of an frog riding a horse on mars"
image = pipe(prompt).images[0]
    
image.save("astronaut_rides_horse.png")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

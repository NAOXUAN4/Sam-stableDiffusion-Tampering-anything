import torch
from diffusers import StableDiffusionInpaintPipeline
import cv2






modelpath = "E:\stablediffusion\sd-webui-aki-v4.2\models\Stable-diffusion\perfectWorld_v3Baked.safetensors"

mask_img = cv2.imread("./img2/2mask.jpg")
ori_img = cv2.imread("./img2/2.jpg")


#modelpath = "runwayml/stable-diffusion-v1-5"
pipe =StableDiffusionInpaintPipeline.from_single_file(modelpath,
                                                torch_dtype=torch.float16,
                                                sampler = "DPM++ 2S Karras",
                                                )
pipe.to("cuda")
pipe.safety_checker = None
pipe.requires_safety_checker = False


prompt = "cloudy"
negtive_prompt = ""

output = pipe(prompt, negative_prompt=negtive_prompt, num_inference_steps=70,num_images_per_prompt=3)

output.images[0].show()
output.images[1].show()
output.images[2].show()
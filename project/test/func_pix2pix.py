# Author : zhaoxingliang
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline


def func_pix2pix(image, prompt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "./instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                  safety_checker=None)
    pipe.to(device)

    steps = 30
    output = pipe(prompt, image=image,num_inference_steps=steps)

    out_images = output.images
    return out_images[0]


if __name__ == '__main__':
    image_path = "4.jpg"
    image = Image.open(image_path)
    prompt = "Make the light darker"
    output = func_pix2pix(image=image, prompt=prompt)
    output.show()

# Author： 赵星亮
# 基于stable-diffusion-inpainting与stable-diffusion-2-inpainting
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import torch


def tamping_process1(init_image,mask_image,prompt,negative_prompt,strength):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "\Matching\models\stable-diffusion-2-inpainting"

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    pipeline.safety_checker = None  # 去除NSFW限制
    pipeline.requires_safety_checker = False

    print(init_image)
    print(init_image.size)

    width = init_image.size[0] - init_image.size[0] % 8
    height = init_image.size[1] - init_image.size[1] % 8
    print(width, height)

    output = pipeline(image=init_image, mask_image=mask_image,
                      prompt=prompt, negative_prompt=negative_prompt,
                      height=height, width=width,
                      strength=strength)

    out_images = output.images
    return out_images[0]


if __name__ == '__main__':
    init_image = Image.open("D:\\annanyi\Pictures\pix\\30613342_p0_master1200.jpg")
    mask_image = Image.open("E:\Matching\diffusers\project\\test\mask.png")
    prompt = "sunny"
    negative_prompt = ""
    strength = 1
    output = tamping_process1(init_image,mask_image,prompt,negative_prompt,strength)
    output.show()

'''
SAM分割图形
用户选取分割图像
制作蒙版mask（参与到SAM输出制作）
将原图与mask放入StableDiffusion
在StableDiffusionInpaintPipeline中利用 标签+指定prompt 进行篡改
实时性能（降低steps）？

将模型调用写成api形式
api：
    标注的标签
    选择篡改的内容
    init_img原图
    mask_img蒙版
    output生成图

多样性的属性篡改：
可以利用图像文本大模型，能够根据场景描述生成多种属性的篡改，包括天气属性、光照属性、主体信息的增加或删除等；
天气属性：
    白天，夜晚
    雨天，晴天，打雷，阴天等
    强调，减弱（strength参数有关）
光照属性：
    柔和，刺眼等
    变暗，变亮
    强调，减弱（strength参数有关）
主体信息（人，生物等）：
    增加，删除，换主体

'''


'''
def print_hi(name):
    print(f'Hi, {name}') 
if __name__ == '__main__':
    print_hi('start')
'''

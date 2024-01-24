import numpy as np
import gradio as gr
import Samuse
import cv2
import sdInpainting1
from PIL import Image


def  mask_process(image,points_str,pointslabel_str):
    # img = cv2.imread("E:\Matching\diffusers\project\\test\img2\\5.jpg")

    points_str = points_str.split()
    points = [[int(x) for x in p.split(',')] for p in points_str]
    input_point = np.array(points)         #input_point = np.array([[200, 75], [800, 50]])


    pointslabel_str = pointslabel_str.split()
    pointslabel = [int(p) for p in pointslabel_str]

    input_label = np.array(pointslabel)


    image,image2,image3 = Samuse.sam_cut(image,input_point,input_label,cut_type="point")
    image = image.astype(np.uint8)
    #plt_mask_output.change(fn=None,outputs=image[1])
    return image,image2,image3,image   #img1:mask; img2:plt

def tamping_process(img,mask,pos_pro,neg_pro,re_draw):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite("init_imgBuf.jpg",img)
    img = Image.open("init_imgBuf.jpg")
    mask = Image.open("mask.png")
    img = sdInpainting1.tamping_process1(img,mask,pos_pro,neg_pro,re_draw)

    return img



with gr.Blocks() as demo:

    gr.Markdown("Add the img your want to process Mask")

    with gr.Tab("Mask processing"):
        with gr.Row():
            image_input = gr.Image(label="image_input",show_label=True)
            image_output = gr.Image(label="image_output",show_label=True)
        with gr.Row():
            plt_mask_output = gr.Image(label="plt_mask",show_label=True)
            with gr.Column():
                text_input = gr.Textbox(label="Input points",show_label=True)
                textlabel_input = gr.Textbox(label="Input Labels(0 or 1)",show_label=True)
                image_button = gr.Button("process")
    with gr.Tab("Tamper with stable-diffusion"):
        with gr.Row():
            TamImg_input = gr.Image(label="TamTmg",show_label=True,)
            TamMask = gr.Image(label="TamMask",show_label=True)
        with gr.Row():
            TamImg_output = gr.Image(label="Result",show_label=True)
            with gr.Column():
                pos_prompt = gr.Textbox(label="Positive-Prompts",show_label=True)
                neg_prompt = gr.Textbox(label="Negative-Prompts",show_label=True)
                Tam_degree = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7,label="Redraw degree",show_label=True)
                Tam_button = gr.Button("process")


    image_button.click(mask_process,inputs = [image_input,text_input,textlabel_input], outputs=[image_output,plt_mask_output,TamImg_input,TamMask])
    Tam_button.click(tamping_process,inputs = [image_input,TamMask,pos_prompt,neg_prompt,Tam_degree], outputs=[TamImg_output])

demo.launch()
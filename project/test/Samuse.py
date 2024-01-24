import cv2
from segment_anything import sam_model_registry,SamPredictor
import torch
import matplotlib.pyplot as plt
import numpy as np

def cv2show(img):
    cv2.imshow("img",img)
    cv2.waitKey()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([238 / 255, 72 / 255, 102 / 255, 0.6])
    h, w = mask.shape[-2:]
    #print(mask)

    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    #cv2.imshow("img",mask_image)
    ax.imshow(mask_image)

def diffuserMask_process(mask,ax,invert_color = True):
    color = np.array([255/255, 255/255, 255/255, 1])

    h, w = mask.shape[-2:]

    result = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    #ax.imshow(result)

    result = result.astype(dtype = np.float32)
    result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)*255

    img = cv2_diffusers_process(result)
    cv2.imwrite("mask.png", img)
    return img

def cv2_diffusers_process(img):
    kernel = np.ones((14, 14), dtype=np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("diffusers_process.jpg", img)

    return img


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def sam_cut(img,input_point = None ,input_label = None,input_box = None,cut_type = "point"):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry["vit_h"](checkpoint="\Matching\models\sam_model\sam_vit_h_4b8939.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(img)

    "点分割"
    if cut_type == "point":
        input_box = None
    elif cut_type == "box":
        input_point = None
        input_label = None
    elif cut_type == "mix":
        pass

    """
    "显示点所在位置"
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()
    """
    "分割"
    masks, scores, logits = predictor.predict(

        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )

    "显示box分割结果"
    for i, (mask, score) in enumerate(zip(masks, scores)):

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_mask(masks[0], plt.gca())
        plt.savefig("pltBuff_nopoint.png",bbox_inches='tight')

        show_points(input_point, input_label, plt.gca())


        img2 = diffuserMask_process(masks[0], plt.gca())


        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        #show_box(input_box, plt.gca())
        plt.axis('on')

        plt.savefig("pltBuff.png",bbox_inches='tight')

        img3 = cv2.imread("pltBuff.png")
        img4 = cv2.imread("pltBuff_nopoint.png")
        return img2, img3, img4

if __name__ == '__main__':


    img = cv2.imread("E:\Matching\diffusers\project\\test\img2\\5.jpg")
    input_point = np.array([[200, 75],[800,50]])
    input_label = np.array([1,1])


    """

    img = cv2.imread("C:\\Users\\annanyi\Downloads\\00092-2383542002-.png")

    
    input_point = np.array([[200, 75], [250, 200], [370, 224], [119, 306],
                            [203, 237], [466, 445], [62, 435], [250, 170],
                            [280, 208], [280, 140], [206, 148]])


    input_label = np.array([1, 0, 1, 1, 1,
                            1, 1, 0, 0,
                            0, 0])
                            
                            """
    sam_cut(img,input_point,input_label,cut_type="point")

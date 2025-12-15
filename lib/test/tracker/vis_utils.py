import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from .visualize_yz import vis_mask_token
import cv2
############## used for visulize eliminated tokens #################
def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


def gen_masked_tokens(tokens, indices, alpha=0.2):
    # indices = [i for i in range(196) if i not in indices]
    indices = indices[0].astype(int)
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens


def recover_image(tokens, H, W, Hp, Wp, patch_size):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(Hp, Wp, patch_size, patch_size, 3).swapaxes(1, 2).reshape(H, W, 3)
    return image


def pad_img(img):
    height, width, channels = img.shape
    im_bg = np.ones((height, width + 8, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg


def gen_visualization(image, mask_indices, patch_size=16):
    # image [224, 224, 3]
    # mask_indices, list of masked token indices

    # mask mask_indices need to cat
    # mask_indices = mask_indices[::-1]
    num_stages = len(mask_indices)
    for i in range(1, num_stages):
        mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)

    # keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)

    stages = [
        recover_image(gen_masked_tokens(image_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
        for i in range(num_stages)
    ]
    imgs = [image] + stages
    imgs = [pad_img(img) for img in imgs]
    viz = np.concatenate(imgs, axis=1)
    return viz

def vis_attn_maps_1(x, vis_tgt_1_3, x1, x1_title, save_path='.'):

    attn_weights_1 = (vis_tgt_1_3[0].cpu() @ x.cpu().transpose(-2, -1)).view(-1, 1, 16, 16)
    attn_weights_3 = (vis_tgt_1_3[1].cpu() @ x.cpu().transpose(-2, -1)).view(-1, 1, 16, 16)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    attn_decoder1_path = save_path + '/AQTP_decoder1_attn_weight.png'
    attn_decoder3_path = save_path + '/AQTP_decoder3_attn_weight.png'
    cur_search_path = save_path + '/{}.png'.format(x1_title)
    fuse_path_decoder1 = save_path + '/attn_decoder1.png'
    fuse_path_decoder3 = save_path + '/attn_decoder3.png'


    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    ax = fig.add_subplot(111)
    ax.imshow(attn_weights_1[0, 0, :, :], cmap='viridis', interpolation='bilinear')
    ax.axis('off')
    plt.savefig(attn_decoder1_path)
    plt.close()

    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    x3_ax = fig.add_subplot(111)
    x3_ax.imshow(attn_weights_3[0, 0, :, :], cmap='viridis', interpolation='bilinear')
    x3_ax.axis('off')
    plt.savefig(attn_decoder3_path)
    plt.close()


    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    x1_ax = fig.add_subplot(111)
    x1_ax.imshow(x1)
    x1_ax.axis('off')
    plt.savefig(cur_search_path)
    plt.close()


    original_image = Image.open(cur_search_path)  # 替换成你的原图路径
    heatmap_image1 = Image.open(attn_decoder1_path)  # 替换成你的热力图路径
    heatmap_image3 = Image.open(attn_decoder3_path)  # 替换成你的热力图路径

    # 调整热力图的透明度
    heatmap_image1 = heatmap_image1.convert("RGBA")
    heatmap_image3 = heatmap_image3.convert("RGBA")

    heatmap_data1 = heatmap_image1.getdata()

    heatmap_data3 = heatmap_image3.getdata()
    new_heatmap_data1 = []
    new_heatmap_data3 = []

    for item in heatmap_data1:
        r, g, b, a = item
        new_heatmap_data1.append((r, g, b, int(a * 0.7)))  # 将热力图的透明度设为100%-85%

    for item in heatmap_data3:
        r, g, b, a = item
        new_heatmap_data3.append((r, g, b, int(a * 0.7)))  # 将热力图的透明度设为100%-85%

    heatmap_image1.putdata(new_heatmap_data1)
    heatmap_image3.putdata(new_heatmap_data3)

    # 将两张图像叠加
    result_image1 = Image.alpha_composite(original_image.convert("RGBA"), heatmap_image1)
    result_image3 = Image.alpha_composite(original_image.convert("RGBA"), heatmap_image3)

    # 保存结果
    result_image1.save(fuse_path_decoder1, "PNG")  # 填入你的保存路径
    result_image3.save(fuse_path_decoder3, "PNG")  # 填入你的保存路径



    del attn_weights_1, attn_weights_3


def vis_attn_maps(x, vis_tgt_1_3, x1, x1_title, save_path='.'):

    attn_weights_1 = (vis_tgt_1_3[0].cpu() @ x.cpu().transpose(-2, -1)).view(-1, 1, 16, 16)
    attn_weights_3 = (vis_tgt_1_3[1].cpu() @ x.cpu().transpose(-2, -1)).view(-1, 1, 16, 16)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cur_search_path = save_path + '/{}.png'.format(x1_title)
    fuse_path_decoder1 = save_path + '/attn_decoder1.png'
    fuse_path_decoder3 = save_path + '/attn_decoder3.png'


    heatmap_data1,_ = vis_mask_token(attn_weights_1.reshape(16,16),x1)
    heatmap_data3,img = vis_mask_token(attn_weights_3.reshape(16, 16), x1)

    cv2.imwrite(cur_search_path, img)  # 保存背景图像
    cv2.imwrite(fuse_path_decoder1, heatmap_data1)  # 保存第一个热图
    cv2.imwrite(fuse_path_decoder3, heatmap_data3)  # 保存第三个热图

    del attn_weights_1, attn_weights_3
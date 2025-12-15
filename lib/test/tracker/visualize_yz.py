import cv2
import numpy as np
import sys


def vis_mask_token(heat_data, img=None, show_size=(150, 150), factor=0.4, window="feature"):
    """ 可视化特征 

    Args:
        heat_data (_type_): (H, W)
        img (_type_, optional): _description_. Defaults to None.
        show_size (tuple, optional): _description_. Defaults to (150, 150).
        factor (float, optional): _description_. Defaults to 0.4.
        window (str, optional): _description_. Defaults to "feature".

    Returns:
        _type_: _description_
    """
    heat_data = heat_data.cpu().numpy()
    heat_data = cv2.resize(heat_data, show_size)

    heat_data_x = heat_data
    Min = np.min(heat_data_x)
    Max = np.max(heat_data_x)
    Sum = np.mean(heat_data_x)
    
    # sys.float_info.epsilon：是一个极小的数，用于避免除数为0的情况，即 heat_data矩阵为0的情况
    # heat_data_max = (heat_data_x - Min) / (Max - Min + sys.float_info.epsilon)
    if (Max - Min) != 0 and not np.isnan(Max - Min):
        heat_data_max = (heat_data_x - Min) / (Max - Min)
    else:
        heat_data_max = (heat_data_x - Min) / (Max - Min + sys.float_info.epsilon)
    
    heat_data = heat_data_max

    heat_data = np.uint8(255 * heat_data)
    heat_data = cv2.applyColorMap(heat_data, cv2.COLORMAP_JET)

    if img is not None:
        img = cv2.resize(img, show_size)
        heat_map_data = np.uint8(img * (1 - factor) + heat_data * factor)
    else:
        heat_map_data = heat_data

    font = cv2.FONT_HERSHEY_SIMPLEX
    heat_map_data = cv2.putText(heat_map_data, window, (0, 0), color=(255, 0, 0), fontFace=font, fontScale=1.2)
    return heat_map_data,img

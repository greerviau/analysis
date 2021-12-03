import numpy as np
import cv2

def process_frame(frame, resolution, zoom = 1):
    height, width, _ = frame.shape
    x_scale = width / resolution[0]
    y_scale = height / resolution[1]

    x_crop_size = resolution[0]
    y_crop_size = resolution[1]
    
    if y_scale < x_scale:
        x_crop_size *= y_scale
        y_crop_size *= y_scale
    else:
        x_crop_size *= x_scale
        y_crop_size *= x_scale
    
    x_crop_size = int(x_crop_size / zoom)
    y_crop_size = int(y_crop_size / zoom)

    x_offset = (width - x_crop_size) // 2
    y_offset = (height - y_crop_size) // 2

    out_frame = frame[y_offset:y_offset+y_crop_size, x_offset:x_offset+x_crop_size, :]
    out_frame = cv2.resize(out_frame, resolution, interpolation=cv2.INTER_AREA)
    return out_frame
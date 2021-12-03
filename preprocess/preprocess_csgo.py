import numpy as np
import pandas as pd
from preprocess import process_frame
import cv2
import os
import sys


def detect_kill(frame, last_frame, indicator_zone: tuple):
    #TODO: Make this more accurate at detecting kills
    frame_zone = frame[indicator_zone[0][0]:indicator_zone[0][1], indicator_zone[1][0]:indicator_zone[1][1], :]
    last_frame_zone = last_frame[indicator_zone[0][0]:indicator_zone[0][1], indicator_zone[1][0]:indicator_zone[1][1], :]

    frame_thresh = np.zeros_like(frame_zone)
    frame_zone_b = frame_zone[:, :, 0]
    frame_zone_g = frame_zone[:, :, 1]
    frame_zone_r = frame_zone[:, :, 2]
    frame_thresh[(frame_zone_b < 100) & (frame_zone_g < 100) & (frame_zone_r > 160)] = 1
    frame_thresh_sum = float(np.sum(frame_thresh))

    last_frame_thresh = np.zeros_like(last_frame_zone)
    last_frame_zone_b = last_frame_zone[:, :, 0]
    last_frame_zone_g = last_frame_zone[:, :, 1]
    last_frame_zone_r = last_frame_zone[:, :, 2]
    last_frame_thresh[(last_frame_zone_b < 80) & (last_frame_zone_g < 80) & (last_frame_zone_r > 140)] = 1
    last_frame_thresh_sum = float(np.sum(last_frame_thresh))

    #print(frame_thresh_sum, last_frame_thresh_sum)
    #print(frame_thresh_sum - last_frame_thresh_sum)
    if frame_thresh_sum - last_frame_thresh_sum > 250:
        print('Kill')
        return True

    #cv2.imshow('indicator', frame_thresh*255)
    #cv2.imshow('last_indicator', last_frame_thresh*255)
    return False

def process_video(input_file, output_segments_path, out_resolution=None, out_fps=-1):
    if not os.path.exists(output_segments_path):
        os.makedirs(output_segments_path)

    cap = cv2.VideoCapture(input_file)

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    native_resolution = (int(width), int(height))

    if not out_resolution:
        out_resolution = native_resolution
    
    native_fps = int(cap.get(cv2.CAP_PROP_FPS))

    assert native_fps >= out_fps

    if out_fps == -1:
        out_fps = native_fps

    print('Input File: ', input_file)
    print('Resolution: ', native_resolution)
    print('FPS: ', native_fps)
    print('Output Segments Path: ', output_segments_path)
    print('Resolution: ', out_resolution)
    print('FPS: ', out_fps)

    delay = 1000//native_fps
    frame_skip = native_fps / out_fps
    counter = 1
    skip_first_n_seconds = 3
    skip_n_frames = native_fps * skip_first_n_seconds
    segments_captured = 1

    cache_n_seconds = 3
    cached_frames = []
    cache_size = 3 * out_fps

    kill_skip = 0

    for i in range(skip_n_frames):
        _, _ = cap.read()

    last_frame = cap.read()[1]
    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret:
            cv2.imshow('Frame',frame)
            if counter % frame_skip == 0:
                processed_frame = process_frame(frame, out_resolution, zoom=3)
                if kill_skip <= 0:
                    if detect_kill(frame, last_frame, ((0,500), (-300,-1))) and len(cached_frames) >= cache_size * .75:
                        kill_skip = 10
                        out = cv2.VideoWriter(os.path.join(output_segments_path, 'segment_{:03d}.mp4'.format(segments_captured)), cv2.VideoWriter_fourcc(*'MP4V'), out_fps, out_resolution)
                        
                        for c_f in cached_frames:
                            out.write(c_f)

                        cached_frames = []
                        rolling_index = 0
                        out.write(processed_frame)
                        segments_captured += 1
                        out.release()
                    else:
                        cached_frames.append(processed_frame)
                        if len(cached_frames) > cache_size:
                            cached_frames.pop(0)

                kill_skip -= 1
                last_frame = frame
            counter += 1

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    RAW_DATA_PATH = 'D:/Data/WALDO-VISION/Initial Test Footage/'
    PROCESSED_DATA_PATH = 'D:/Data/WALDO-VISION/Processed Test Footage'

    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    video_files = os.listdir(RAW_DATA_PATH)
    no_cheat = [video for video in video_files if 'Aim-Assist OFF' in video]
    cheat = [video for video in video_files if 'Aim-Assist ON' in video]

    print(no_cheat)
    print(cheat)

    for video in no_cheat:
        process_video(os.path.join(RAW_DATA_PATH, video), os.path.join(PROCESSED_DATA_PATH, video.split('.')[0]), out_resolution=(640,480))
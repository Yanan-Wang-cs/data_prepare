import numpy as np
import cv2
from process_utils import *
import os
import time
from pathlib import Path
import argparse
from tqdm import tqdm

def find_continuous_segments(arr, num=20):
    arr = np.sort(arr.astype(int))  # 确保输入是有序的
    diffs = np.diff(arr)  # 计算相邻元素的差
    split_indices = np.where(diffs > 1)[0] + 1  # 找到断开点
    segments = np.split(arr, split_indices)  # 根据断开点分段
    return [list(segment) for segment in segments if len(segment) > num]

def get_info(file_path):
    info = np.load(file_path)
    video_name, file_ext = os.path.splitext(os.path.basename(file_path))
    video_name = video_name.replace("_face_detect", "")
    video_path = file_path.replace('_face_detect.npy', f'/{video_name}.mp4')
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    filter_info = []
    for lines in info:
        bbox = lines[1:5]
        bwidth, bheight = bbox[2], bbox[3]
        if bwidth > 0.10 * width and bheight > 0.10 * height:
            filter_info.append(lines)
    cap.release()
    return np.array(filter_info), video_path, video_name, width, height

def extract_video_segment(input_path, output_path, start_frame, end_frame, bbox=None, size=512):
    """
    从视频中提取指定帧范围的片段，并保存为新视频。
    
    Args:
        input_path (str): 原视频路径
        output_path (str): 输出视频路径
        start_frame (int): 起始帧
        end_frame (int): 结束帧
    """
    if os.path.exists(output_path):
        return
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # print(f"视频总帧数: {total_frames}, 帧率: {fps},")

    # 检查帧范围合法性
    if start_frame < 0 or end_frame > total_frames or start_frame >= end_frame:
        print("Error: Invalid frame range.")
        return

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    if bbox is not None:
        out = cv2.VideoWriter(output_path, fourcc, fps, (size, size))
    else:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 逐帧读取并写入新视频
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if bbox is not None:
            frame = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            frame = cv2.resize(frame, (size, size))
        if not ret:
            print("Error: Failed to read frame.")
            break
        out.write(frame)
        current_frame += 1

    # 释放资源
    cap.release()
    out.release()
    # print(f"视频片段保存到 {output_path}")

def get_enlarge_bbox(width, height, faceBox,scale=1.8):
    cx_box = (faceBox[0] + faceBox[2]) / 2.
    cy_box = (faceBox[1] + faceBox[3]) / 2.
    face_width = faceBox[2] - faceBox[0] + 1
    face_height = faceBox[3] - faceBox[1] + 1
    face_size = max(face_width, face_height)
    bbox_size = min(int(face_size * scale),height,width)
   
    x_min = max(0,int(cx_box-bbox_size / 2.))
    y_min = max(0,int(cy_box-bbox_size / 2.))
    x_max = x_min + bbox_size
    y_max = y_min + bbox_size
    if x_max > width:
        x_max = width
        x_min = x_max - bbox_size
    if y_max > height:
        y_max = height
        y_min = y_max - bbox_size

    boundingBox = [max(x_min, 0), max(y_min, 0), min(x_max, width), min(y_max, height)]
    boundingBox = [int(x) for x in boundingBox]
    return boundingBox
def get_resized_position(lmks, width, height, cut_bbox, face_box, size=512):
    lmks = lmks.reshape(-1,2)
    lmks[:,0] = lmks[:,0] * width
    lmks[:,1] = lmks[:,1] * height
    lmks[:,0] = lmks[:,0] - cut_bbox[0]
    lmks[:,1] = lmks[:,1] - cut_bbox[1]
    face_box = xywh2xyxy(face_box)
    face_box = [face_box[0] - cut_bbox[0], face_box[1] - cut_bbox[1], face_box[2] - cut_bbox[0], face_box[3] - cut_bbox[1]]

    scale = size / (cut_bbox[2] - cut_bbox[0])

    
    lmks = lmks.reshape(-1, 1)
    lmks = lmks * scale
    lmks = [int(x) for x in lmks]
    face_box = [int(x*scale) for x in face_box]
    return [*face_box, *lmks]

def rewrite_info(info, width, height):
    info_new = []
    frame_num = 0
    bbox = info[0][1:5]
    bbox = xywh2xyxy(bbox)
    enlarge_bbox = get_enlarge_bbox(width, height, bbox, scale=5)
    for lines in info:
        resized_info = get_resized_position(lines[6:-1], width, height, enlarge_bbox, lines[1:5])
        info_new.append([frame_num, *resized_info])
        frame_num += 1
    return enlarge_bbox, np.array(info_new)

def save_segments(info, segments, video_path, video_name, width, height, isSingle=True):
    for segment in tqdm(segments, desc=f"Processing video frames, isSingle: {isSingle}"):
        start_frame = segment[0]
        end_frame = segment[-1]
        if isSingle:
            info_clip = info[(info[:, 0] > start_frame) & (info[:, 0] <= end_frame)]
            enlarge_bbox, new_info = rewrite_info(info_clip, width, height)
            save_path = os.path.dirname(video_path).replace(opt.folder, f'{opt.folder}_single')
            os.makedirs(save_path, exist_ok=True)
            np.save(f'{save_path}/{video_name}_clip_{start_frame}_{end_frame}.npy', new_info)
        else:
            save_path = os.path.dirname(video_path).replace(opt.folder, f'{opt.folder}_multiple')

        os.makedirs(save_path, exist_ok=True)
        if isSingle:
            extract_video_segment(video_path, f'{save_path}/{video_name}_clip_{start_frame}_{end_frame}.mp4', start_frame, end_frame, enlarge_bbox)
        else:
            extract_video_segment(video_path, f'{save_path}/{video_name}_clip_{start_frame}_{end_frame}.mp4', start_frame, end_frame)

def process_video(npy_path):
    info, video_path, video_name, width, height = get_info(npy_path)
    if len(info) == 0:
        return
    frames_number = np.unique(info[:,0])
    single_person_frames = []
    multiple_person_frames = []
    for i in frames_number:
        filter_info = info[info[:,0]==i]
        if len(filter_info) == 0:
            print('no person:', i)
        elif len(filter_info) == 1:
            single_person_frames.append(i)
        else:
            multiple_person_frames.append(i)

    segments_single = find_continuous_segments(np.array(single_person_frames))
    segments_multiple = find_continuous_segments(np.array(multiple_person_frames))

    clip = []
    clip2 = []
    for segment in segments_single:
        change_arr = False
        previous_bbox = None
        for i in segment:
            info_single = info[info[:,0]==i][0]
            if previous_bbox is None:
                previous_bbox = xywh2xyxy(info_single[1:5])
            bbox = xywh2xyxy(info_single[1:5])
            iou = compute_iou(previous_bbox, bbox)
            if iou < 0.3:
                change_arr = not change_arr
                
            if change_arr:
                clip2.append(i)
            else:
                clip.append(i)
            previous_bbox = bbox

    segments_clip = find_continuous_segments(np.array(clip))
    segments_clip2 = find_continuous_segments(np.array(clip2))

    start_time = time.time()
    save_segments(info, segments_clip, video_path, video_name, width, height)
    save_segments(info, segments_clip2, video_path, video_name, width, height)
    save_segments(info, segments_multiple, video_path, video_name, width, height, isSingle=False)
    end_time = time.time()
    # print('during:', end_time - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--folder', default='video_1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_folder', default='/l/users/yanan.wang/project/MixHeadSwap/dataset/stablevideo/frame/id01460/BHHDBsXtZhI-0001/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='/l/users/yanan.wang/project/dataPrepare/video_1/', type=str, help='dataset path')
    opt = parser.parse_args()
    print(opt)
    npyfiles = sorted(list(Path(opt.dataset_folder).rglob("*_face_detect.npy")))
    for npy_path in tqdm(npyfiles, desc=f"Total files in {opt.folder}"):
        npy_path = str(npy_path)
        # print(npy_path)
        process_video(npy_path)







import os 
import imageio
import numpy as np
import torch
import math
import cv2
import pdb
from multiprocessing import Process
import multiprocessing as mp
import pdb 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from generateMask import processing, getkeypoint
def find_continuous_segments(arr):
    arr = np.sort(arr.astype(int))  # 确保输入是有序的
    diffs = np.diff(arr)  # 计算相邻元素的差
    split_indices = np.where(diffs > 1)[0] + 1  # 找到断开点
    segments = np.split(arr, split_indices)  # 根据断开点分段
    return [list(segment) for segment in segments]

def get_video_info(base,save_base,q):
   
    for idname in os.listdir(base):
        idpath = os.path.join(base,idname)
        save_path = os.path.join(save_base,'frame', idname)
        for videoname in os.listdir(idpath):
            videopath = os.path.join(idpath,videoname)
            frame_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.mp4')]
            info_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.npy')]
            if len(frame_names) == 0:
                continue
            q.put([frame_names[0],info_names,save_path,videoname])
           


def process_frame(q1,align=True,scale=1.8,size=512):
    kk = 1
    base_options_key = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options_key = vision.FaceLandmarkerOptions(base_options=base_options_key,
                                        output_face_blendshapes=False,
                                        output_facial_transformation_matrixes=False,
                                        num_faces=1)
    detector_key = vision.FaceLandmarker.create_from_options(options_key)

    base_options = python.BaseOptions(model_asset_path='google/selfie_multiclass_256x256.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                        output_category_mask=True)
    while True:
        base_options = python.BaseOptions(model_asset_path='google/selfie_multiclass_256x256.tflite')
        options = vision.ImageSegmenterOptions(base_options=base_options,
                                              output_category_mask=True)

        # Create the image segmenter
        
        frame_path,info_names,save_base,videoname = q1.get()
        if frame_path is None:
            break 
        video_reader = imageio.get_reader(frame_path)
        num = 0
        for k,info_name in enumerate(info_names):
            print(info_names, frame_path)
            info = np.load(info_name)
            index_list = info[:,0]
            segments = find_continuous_segments(index_list)
            
            for segment in segments:
                if len(segment) < 16:
                    continue
                start = segment[0]
                end = segment[-1]
                
                # if os.path.exists(os.path.join(save_base,'frame', '%s-%04d-%04d'%(videoname,start,end))):
                #     continue
                save_path = os.path.join(save_base,videoname, '%s-%04d-%04d'%(videoname,start,end))
                os.makedirs(save_path,exist_ok=True)

                for i in segment:
                    try:
                        img = video_reader.get_data(int(i))
                        img_path = os.path.join(save_path,'%04d.png'%int(i))
                        cv2.imwrite(img_path,img[...,::-1])
                        image_path = img_path
                        # Create the image segmenter
                        mask_path = image_path.replace('frame','mask')
                        feature_path = image_path.replace('frame','feature')
                        if not os.path.exists(mask_path):
                            with vision.ImageSegmenter.create_from_options(options) as segmenter:
                                processing(image_path.replace('frame','mask'), image_path, segmenter)
                        elif not os.path.exists(feature_path):
                            getkeypoint(image_path, cv2.imread(mask_path), detector_key, mask_path)
                        kk += 1
                    except:
                        continue
                num += 1
    
        video_reader.close()
    print(frame_path)


if __name__ == "__main__":
    
    
    mp.set_start_method('spawn')
    m = mp.Manager()
    q1 = m.Queue()
    base = '/l/users/yanan.wang/dataset/done/video_cache/2/'
    # 
    save_base = '../MixHeadSwap/dataset/prepare/'
    process_num = 2
   
    info_p = Process(target=get_video_info,args=(base,save_base,q1,))
    

    process_list = []
    for _ in range(process_num):
        process_list.append(Process(target=process_frame,args=(q1,)))
     

    info_p.start()
    for p in process_list:
        p.start()

    info_p.join()
    
    for _ in range(process_num*2):
         q1.put([None,None,None,None])
    for p in process_list:
        p.join()
   
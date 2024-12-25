import os 
import imageio
import numpy as np
from face_alignment.detection.sfd import FaceDetector
import face_alignment
import torch
import math
import cv2
import pdb
from multiprocessing import Process
import multiprocessing as mp
from process_utils import *
import pdb 
import json
import onnxruntime as ort
from tqdm import tqdm
from pathlib import Path
import glob
def get_video_info(base,save_base,q):
   
    for idname in os.listdir(base):
        idname = 'id01567'
        idpath = os.path.join(base,idname)
        save_path = os.path.join(save_base,idname)
        for videoname in os.listdir(idpath):
            videopath = os.path.join(idpath,videoname)
            frame_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.mp4')]
            info_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.npy')]
            if len(frame_names) == 0:
                continue
            q.put([frame_names[0],info_names,save_path,videoname])
        break

def get_mask(img, bbox, session, input_name, output_name):
    height,width,_ = img.shape
    scale = 256 / height
    resize_bbox = [int(bbox[0]*scale),int(bbox[1]*scale),int(bbox[2]*scale),int(bbox[3]*scale)]

    input_data = cv2.resize(img, (256, 256))
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 127.5 - 1
    output_data = session.run([output_name], {input_name: input_data})
    category_mask = output_data[0][0]
    # category_mask = cv2.resize(category_mask, numpy_image.shape[:2])
    category_mask = category_mask.argmax(axis=2).astype(np.uint8)
    category_mask[category_mask==2] =0
    category_mask[category_mask==4] =0

    category_mask[category_mask>0]=255
    gray = category_mask
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise Exception('not detext face!')
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    result = None
    x,y,w,h = None,None,None,None
    max_iou = 0
    for contour in sorted_contours:
        # 获取轮廓的外接矩形
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        contour_bbox = [x1, y1, x1 + w1, y1 + h1]

        # 计算 IOU
        iou = compute_iou(resize_bbox, contour_bbox)
        if iou >= max_iou:
            max_iou = iou
            x, y, w, h = int(x1/scale), int(y1/scale), int(w1/scale), int(h1/scale)
            mask = np.zeros_like(binary_image, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            result = cv2.bitwise_and(category_mask, category_mask, mask=mask)
        if iou >= 0.8:
            break
    result = cv2.resize(result, (height, width), interpolation=cv2.INTER_LINEAR)
    return result, [x,y,w,h]

def get_keypoints(lmk_detector,head_img,size,x,y,w,h):
    black_image = np.zeros(head_img.shape, dtype=np.uint8)
    try:
        landmarks = lmk_detector.get_landmarks_from_image(head_img[...,::-1])[0]
        # 在原图上绘制关键点
        for (x, y) in landmarks[:,:2]:
            x = int(x)
            y = int(y)
            cv2.circle(black_image, (x,y), 2, (255,0,0), -1)
    except:
        return black_image, []
    return black_image, landmarks

def get_head_img(image_cropped,head_mask):
    normalized_mask = head_mask / 255.0
    op_img = image_cropped.astype(np.float32)
    op_img = op_img * normalized_mask[:,:,np.newaxis]
    head_img = np.clip(op_img, 0, 255).astype(np.uint8)
    return head_img

def get_bboxes(img, detect_faces, ix,iy,iw,ih):
    height,width,_ = img.shape
    ix,iy,iw,ih = list(map(lambda x:float(x),[ix,iy,iw,ih]))
    box = [ix*width,iy*height,(ix+iw)*width,(iy+ih)*height]
    bboxes = detect_faces([img])[0]
    bbox = choose_one_detection(bboxes,box)
    if bbox is None:
        return None 
    bbox = bbox[:4]
    bbox = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]
    return bbox

def save_landmarks_to_json(landmarks_dict, json_file):
    """
    将 landmarks 数据保存到 JSON 文件中。
    如果文件已存在，则追加新的数据。
    """
    # 如果 JSON 文件已存在，先读取内容
    if os.path.exists(json_file):
        with open(json_file, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    # 更新已有数据
    existing_data.update(landmarks_dict)

    # 保存到 JSON 文件
    with open(json_file, "w") as file:
        json.dump(existing_data, file, indent=4)
    print(f"Landmarks saved to {json_file}")

def process_frame(q1,align=False,scale=5.0,size=512):
    face_detector = FaceDetector(device='cuda')
    lmk_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
    def detect_faces(images):
        images = np.stack(images).transpose(0,3,1,2).astype(np.float32)
        images_torch = torch.tensor(images)
        return face_detector.detect_from_batch(images_torch.cuda())

    while True:
        frame_path,info_names,save_base,videoname = q1.get()
        if frame_path is None:
            break 
        model_path = 'selfie_multiclass_256x256_opset18.onnx'
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        video_reader = imageio.get_reader(frame_path)
        for k,info_name in enumerate(info_names):
            info = np.load(info_name)
            save_path = os.path.join(save_base,'frame','%s-%04d'%(videoname,k))
            mask_path = os.path.join(save_base,'mask','%s-%04d'%(videoname,k))
            keypoint_path = os.path.join(save_base,'keypoint','%s-%04d'%(videoname,k))
            head_path = os.path.join(save_base,'head','%s-%04d'%(videoname,k))
            os.makedirs(save_path,exist_ok=True)
            # if os.path.exists(save_path):
            #     continue
            
            previous = None
            video_index = 0
            kk = 0
            for (i,ix,iy,iw,ih) in info:
                try:
                    i = int(i)
                    if glob.glob(f"{save_path}/*/"+'%04d.jpg'%i):
                        print(f"{save_path}/*/"+'%04d.jpg exist'%i)
                        continue
                    img = video_reader.get_data(i)
                    bbox = get_bboxes(img, detect_faces,ix,iy,iw,ih)
                    if bbox is None:
                        continue
                    
                    image_cropped, fix_bbox = new_crop_with_padding(img,bbox,scale=scale,size=size,align=align)
                    head_mask, [x,y,w,h] = get_mask(image_cropped, fix_bbox, session, input_name, output_name)

                    if head_mask is None:
                        continue

                    keypoint_img, landmarks = get_keypoints(lmk_detector,image_cropped[y:y+h,x:x+w],size,x,y,w,h)
                    keypoint_img = cv2.resize(keypoint_img, (size, size))

                    head_img = get_head_img(image_cropped,head_mask)
                    head_img = head_img[y:y+h,x:x+w]
                    head_img = cv2.resize(head_img, (size, size))

                    if previous is None:
                        previous = bbox
                    previous = bbox
                    # landmarks_str = ','.join(map(str,landmarks.flatten()))
                    
                    os.makedirs(os.path.join(save_path,'%04d'%video_index),exist_ok=True)
                    os.makedirs(os.path.join(mask_path,'%04d'%video_index),exist_ok=True)
                    os.makedirs(os.path.join(keypoint_path,'%04d'%video_index),exist_ok=True)
                    os.makedirs(os.path.join(head_path,'%04d'%video_index),exist_ok=True)

                    cv2.imwrite(os.path.join(save_path,'%04d'%video_index, '%04d.jpg'%i),image_cropped[...,::-1])
                    cv2.imwrite(os.path.join(mask_path,'%04d'%video_index, '%04d_mask.jpg'%i),head_mask)
                    cv2.imwrite(os.path.join(keypoint_path,'%04d'%video_index, '%04d_keypoint.jpg'%i),keypoint_img[...,::-1])
                    cv2.imwrite(os.path.join(head_path,'%04d'%video_index, '%04d_head.jpg'%i),head_img[...,::-1])
                    kk += 1
                    print('\r have done %06d/%06d'%(kk,len(info)),end='',flush=True)
                except:
                    continue
                break
        video_reader.close()
    print()


if __name__ == "__main__":
    print(torch.cuda.is_available()) 


    mp.set_start_method('spawn')
    m = mp.Manager()
    q1 = m.Queue()
    base = '/l/users/yanan.wang/project/dataPrepare/video_2'
    save_base = '/l/users/yanan.wang/project/MixHeadSwap/dataset/newdata'
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
   
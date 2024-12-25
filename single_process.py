import os 
import imageio
import numpy as np
from face_alignment.detection.sfd import FaceDetector
import face_alignment
import torch
import cv2
from process_utils import *
import pdb 
import json
import onnxruntime as ort
from pathlib import Path
import glob
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'selfie_multiclass_256x256_opset18.onnx'
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
face_detector = FaceDetector(device='cuda')
lmk_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
base_options = python.BaseOptions(model_asset_path='google/selfie_multiclass_256x256.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                    output_category_mask=True)

  # Create the image segmenter
segmenter= vision.ImageSegmenter.create_from_options(options)

def detect_faces(images):
    images = np.stack(images).transpose(0,3,1,2).astype(np.float32)
    images_torch = torch.tensor(images)
    return face_detector.detect_from_batch(images_torch.cuda())

def get_mask(img, bbox, session, input_name, output_name):
    
    height,width,_ = img.shape
    scale = 256 / height
    resize_bbox = [int(bbox[0]*scale),int(bbox[1]*scale),int(bbox[2]*scale),int(bbox[3]*scale)]

    # input_data = cv2.resize(img, (256, 256))
    # input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    # input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 127.5 - 1
    
    start_time = time.time()
    # output_data = session.run([output_name], {input_name: input_data})
    # model_time = time.time() - start_time
    # category_mask = output_data[0][0]
    # # category_mask = cv2.resize(category_mask, numpy_image.shape[:2])
    # category_mask = category_mask.argmax(axis=2).astype(np.uint8)
    # category_mask[category_mask==2] =0
    # category_mask[category_mask==4] =0

    # ag1_time = time.time()
    category_mask = get_mask2(img, segmenter)

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
    if result is None:
        raise Exception('not detext face!')
    result = cv2.resize(result, (height, width), interpolation=cv2.INTER_LINEAR)
    result_time = time.time() - start_time
    print('result time: %06d'%(result_time))
    return result, [x,y,w,h]

def get_mask2(img, segmenter):
  
    # Create the MediaPipe image file that will be segmented
    image =mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask

    my_mask = category_mask.numpy_view().copy()
    my_mask.flags.writeable = True
    my_mask[my_mask==2] =0
    my_mask[my_mask==4] =0

    return my_mask
 
def get_keypoints(lmk_detector,head_img,size,x,y,w,h):
    black_image = np.zeros(head_img.shape, dtype=np.uint8)
    try:
        landmarks = lmk_detector.get_landmarks_from_image(head_img)[0]
        # cv2.imwrite('tmp.jpg',head_img)
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

def create_dir(save_base,videoname,kk):
    save_path = os.path.join(save_base,'%s-%04d'%(videoname,kk))
    mask_path = save_path.replace('frame','mask')
    keypoint_path = save_path.replace('frame','keypoint')
    head_path = save_path.replace('frame','head')
    os.makedirs(save_path,exist_ok=True)
    os.makedirs(mask_path,exist_ok=True)
    os.makedirs(keypoint_path,exist_ok=True)
    os.makedirs(head_path,exist_ok=True)
    return save_path,mask_path,keypoint_path,head_path
def process_frame(frame_path,info_names,save_base,videoname, align=False,scale=5.0,size=512): 
    # video_reader = imageio.get_reader(frame_path)
    cap = cv2.VideoCapture(frame_path)
    kk = 0
    num = 0
    for k,info_name in enumerate(info_names):
        print('processing:', info_name)
        info = np.load(info_name)
        print('info loaded:', len(info))
        kk = 0
        save_array = np.empty((0,5))
        crop_bbox = None
        for (i,ix,iy,iw,ih) in info:
            # try:
            time_start = time.time()
            i = int(i)
            _, img = cap.read()
            height,width,_ = img.shape
            ix,iy,iw,ih = list(map(lambda x:float(x),[ix,iy,iw,ih]))
            bbox = [ix*width,iy*height,(ix+iw)*width,(iy+ih)*height]

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            
            if (crop_bbox is None) or (not is_box_inside(crop_bbox,bbox)):
                if len(save_array) > 0:
                    np.save(f'{save_base}/{videoname}_{kk}_crop.npy',save_array)
                    save_array = np.array([])
                kk+=1
                crop_bbox = new_crop_with_padding(img,bbox,scale=scale,size=size,align=align)

            imgCropped = img[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
            image_cropped = cv2.resize(imgCropped, (size, size))
            
            save_array = np.vstack((save_array,np.array([i, crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]])))

            time_box = time.time()

            head_mask, [x,y,w,h] = get_mask(image_cropped, crop_bbox, session, input_name, output_name)
            time_mask = time.time()

            keypoint_img, landmarks = get_keypoints(lmk_detector,image_cropped[y:y+h,x:x+w],size,x,y,w,h)
            keypoint_img = cv2.resize(keypoint_img, (size, size))
            time_keypoint = time.time()

            head_img = get_head_img(image_cropped,head_mask)
            head_img = head_img[y:y+h,x:x+w]
            head_img = cv2.resize(head_img, (size, size))

            save_path,mask_path,keypoint_path,head_path = create_dir(save_base,videoname,kk)
            cv2.imwrite(os.path.join(save_path, '%04d.jpg'%i),image_cropped)
            cv2.imwrite(os.path.join(mask_path,'%04d_mask.jpg'%i),head_mask)
            cv2.imwrite(os.path.join(keypoint_path,'%04d_keypoint.jpg'%i),keypoint_img[...,::-1])
            cv2.imwrite(os.path.join(head_path,'%04d_head.jpg'%i),head_img)
            num += 1
            end_time = time.time()
            print('num: %06d, total time: %06d, bbox time: %06d, mask time: %06d, keypoint time: %06d'%(num,end_time-time_start,time_box-time_start,time_mask-time_box,time_keypoint-time_mask))
            # except:
            #     continue
            
        # break
    cap.release()
    print()


if __name__ == "__main__":
    print(torch.cuda.is_available()) 

    base = '/l/users/yanan.wang/project/dataPrepare/video_2'
    save_base = '/l/users/yanan.wang/project/MixHeadSwap/dataset/stablevideo'

    idname='id01593'
    idpath = os.path.join(base,idname)
    save_path = os.path.join(save_base,'frame', idname)
    for videoname in os.listdir(idpath):
        videopath = os.path.join(idpath,videoname)
        frame_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.mp4')]
        info_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.npy')]
        print('processing:', frame_names[0])
        # try:
        process_frame(frame_names[0],info_names,save_path,videoname)
        # except:
        #     print(f'{frame_names[0]} failed')

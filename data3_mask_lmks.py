import os 
import numpy as np
import face_alignment
import torch
import cv2
from process_utils import *
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from tqdm import tqdm
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lmk_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
base_options = python.BaseOptions(model_asset_path='google/selfie_multiclass_256x256.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                    output_category_mask=True)

  # Create the image segmenter
segmenter= vision.ImageSegmenter.create_from_options(options)

def get_mask(img, bbox):
    image =mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask

    category_mask = category_mask.numpy_view().copy()
    category_mask.flags.writeable = True
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
        iou = compute_iou(bbox, contour_bbox)
        if iou >= max_iou:
            max_iou = iou
            x, y, w, h = int(x1), int(y1), int(w1), int(h1)
            mask = np.zeros_like(binary_image, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            # result = cv2.bitwise_and(category_mask, category_mask, mask=mask)
            result = mask
            
        if max_iou >= 0.8:
            break
    if result is None or max_iou < 0.1:
        cv2.imwrite('tmp.jpg',result)
        # print(iou)
        raise Exception('not detext face!')
    # cv2.rectangle(result,(x,y),(x+w,y+h),(255,255,255),2)
    # cv2.rectangle(result,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(255,255,255),2)
    # print(max_iou)
    return result, [x,y,w,h]

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
        raise Exception('not detext face!')
    return black_image, landmarks

def get_head_img(image_cropped,head_mask):
    normalized_mask = head_mask / 255.0
    op_img = image_cropped.astype(np.float32)
    op_img = op_img * normalized_mask[:,:,np.newaxis]
    head_img = np.clip(op_img, 0, 255).astype(np.uint8)
    return head_img

def create_dir(save_base,videoname):
    save_path = os.path.join(save_base,'%s'%(videoname))
    mask_path = save_path.replace('frame','mask')
    keypoint_path = save_path.replace('frame','keypoint')
    head_path = save_path.replace('frame','head')
    os.makedirs(save_path,exist_ok=True)
    os.makedirs(mask_path,exist_ok=True)
    os.makedirs(keypoint_path,exist_ok=True)
    os.makedirs(head_path,exist_ok=True)
    return save_path,mask_path,keypoint_path,head_path
def get_resize_bbox(bbox,scale, offset):
    x,y,x2, y2 = bbox
    x = x - offset[0]
    y = y - offset[1]
    x2 = x2 - offset[0]
    y2 = y2 - offset[1]

    x = int(x*scale)
    y = int(y*scale)
    x2 = int(x2*scale)
    y2 = int(y2*scale)
    return [x,y,x2,y2]
def process_frame(frame_path,save_base,videoname, size=512): 
    save_path,mask_path,keypoint_path,head_path = create_dir(save_base,videoname)
    video_clip_name = os.path.basename(frame_path)
    if os.path.exists(os.path.join(head_path,'%s'%video_clip_name)):
        return
    cap = cv2.VideoCapture(frame_path)
    num = 0
    info = np.load(frame_path.replace('.mp4','.npy'))
    frame_list = []
    mask_list = []
    keypoint_list = []
    head_list = []
    for lines in tqdm(info, desc=f"{video_clip_name}"):
        try:
            i = int(lines[0])
            # time_start = time.time()
            _, img = cap.read()
            ix,iy,ix2,iy2 = list(map(lambda x:int(x),lines[1:5]))
            bbox = [ix,iy,ix2,iy2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            head_mask, [x,y,w,h] = get_mask(img, bbox)
            # time_mask = time.time()

            keypoint_img, _ = get_keypoints(lmk_detector,img[y:y+h,x:x+w],size,x,y,w,h)
            keypoint_img = cv2.resize(keypoint_img, (size, size))
            # time_keypoint = time.time()

            head_img = get_head_img(img,head_mask)
            head_img = head_img[y:y+h,x:x+w]
            head_img = cv2.resize(head_img, (size, size))

            # save_path,mask_path,keypoint_path,head_path = create_dir(save_base,videoname)
            # cv2.imwrite(os.path.join(save_path, '%04d.jpg'%i),img)
            # cv2.imwrite(os.path.join(mask_path,'%04d_mask.jpg'%i),head_mask)
            # cv2.imwrite(os.path.join(keypoint_path,'%04d_keypoint.jpg'%i),keypoint_img[...,::-1])
            # cv2.imwrite(os.path.join(head_path,'%04d_head.jpg'%i),head_img)

            frame_list.append(img[...,::-1])
            mask_list.append(head_mask)
            keypoint_list.append(keypoint_img)
            head_list.append(head_img[...,::-1])
            num += 1
            # end_time = time.time()
            # print('\r i: %06d, num: %06d, total time: %06d, mask time: %06d, keypoint time: %06d'%(i,num,end_time-time_start,time_mask-time_start,time_keypoint-time_mask), end="", flush=True)
        except:
            continue
            
        # break
    if len(frame_list) > 0:
        images_to_video(frame_list,os.path.join(save_path,'%s'%video_clip_name),fps=30)
        images_to_video(mask_list,os.path.join(mask_path,'%s'%video_clip_name),fps=30)
        images_to_video(keypoint_list,os.path.join(keypoint_path,'%s'%video_clip_name),fps=30)
        images_to_video(head_list,os.path.join(head_path,'%s'%video_clip_name),fps=30)
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', default='/l/users/yanan.wang/project/MixHeadSwap/dataset/stablevideo_clips', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='/l/users/yanan.wang/project/dataPrepare/video_1_single', type=str, help='dataset path')
    opt = parser.parse_args()
    print(opt)

    base = opt.dataset_folder
    save_base = opt.save_folder

    for idname in os.listdir(base):
        idpath = os.path.join(base,idname)
        save_path = os.path.join(save_base,'frame', idname)
        for videoname in tqdm(os.listdir(idpath), desc=f"{idname}"):
            videopath = os.path.join(idpath,videoname)
            frame_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.mp4')]
            # print('processing:', frame_names[0])
            # try:
            for frame_name in frame_names:
                process_frame(frame_name,save_path,videoname)
            # except:
            #     print(f'{frame_names[0]} failed')

# pip install onnxruntime-gpu
import onnxruntime as ort
import numpy as np
import cv2
import time
from tqdm import tqdm
import os
from pathlib import Path
import imageio

def remove_no_video_folder(folder_path):
    for file in os.listdir(folder_path):
        len_video = len(list(Path(f'{folder_path}/{file}').rglob("*.mp4")))
        if not len_video:
            for filename in os.listdir(f'{folder_path}/{file}'):
                if filename.endswith('.npy'):
                    os.remove(f'{folder_path}/{file}/{filename}')   
            os.rmdir(f'{folder_path}/{file}')   
            print(f'{folder_path}/{file} removed')

def get_mask(image_path, save_name):
    # Load the input image.
    numpy_image = cv2.imread(image_path)
    input_data = cv2.resize(numpy_image, (256, 256))
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 127.5 - 1
    output_data = session.run([output_name], {input_name: input_data})
    # Process the output data.
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
        print(f'==============={image_path} not detext face!=========')
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(binary_image, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        os.makedirs(os.path.dirname(save_name),exist_ok=True)
        result = cv2.bitwise_and(category_mask, category_mask, mask=mask)
        cv2.imwrite(save_name, result)

def get_img_from_video(videopath, save_base):
    frame_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.mp4')]
    info_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.npy')]
    if len(frame_names) == 0:
        return
    for k,info_name in enumerate(info_names):
        frame_path = frame_names[0]
        info = np.load(info_name)
        video_reader = imageio.get_reader(frame_path)
        os.makedirs(save_base,exist_ok=True)
        os.makedirs(save_base.replace('frame','mask'),exist_ok=True)
        os.makedirs(save_base.replace('frame','head'),exist_ok=True)
        for (i,x,y,w,h) in info:
            i = int(i)
            img = video_reader.get_data(i)
            height,width,_ = img.shape
            mask = np.zeros_like(img[:,:,0], dtype=np.uint8)
            x,y,w,h = list(map(lambda x:float(x),[x,y,w,h]))
            
            box = [x*width,y*height,(x+w)*width,(y+h)*height]
            mask[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = 255

            head = img[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
            img_path = os.path.join(save_base,'%04d.jpg'%i)
            cv2.imwrite(img_path,img[...,::-1])
            cv2.imwrite(img_path.replace('frame','mask'),mask)
            cv2.imwrite(img_path.replace('frame','head'),head[...,::-1])
            # except:
            #     print(f'{videopath} {i} error!')
    print(f'{videopath} done!')

getFrame=True
getMask=False

if getFrame:
    video_path = '/l/users/yanan.wang/project/dataPrepare/video_1'
    # for id in os.listdir(video_path):
    id='id00812'
    for video in os.listdir(os.path.join(video_path,id)):
        # if os.path.exists(os.path.join('/l/users/yanan.wang/project/MixHeadSwap/dataset/frame',id,video)):
        #     file_num = len(os.listdir(os.path.join('/l/users/yanan.wang/project/MixHeadSwap/dataset/frame',id,video)))
        #     if file_num > 20:
        #         print(f'skip {id} {video} {file_num}')
        #         continue
                
        get_img_from_video(os.path.join(video_path,id,video), os.path.join('/l/users/yanan.wang/project/MixHeadSwap/dataset/frame',id,video))
# for folder in os.listdir('/l/users/yanan.wang/project/dataPrepare/video_2'):
#     remove_no_video_folder(f'/l/users/yanan.wang/project/dataPrepare/video_2/{folder}')
if getMask:
    model_path = 'selfie_multiclass_256x256_opset18.onnx'
    img_folder = '/l/users/yanan.wang/project/MixHeadSwap/dataset/frame/id01000'
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Perform inference.
    time_start = time.time()
    imglist = list(Path(f'{img_folder}').rglob("*.png"))
    for i in tqdm(range(len(imglist))):
        get_mask(str(imglist[i]), str(imglist[i]).replace('frame','mask'))
    time_end = time.time()
    print('Time: ', time_end - time_start)


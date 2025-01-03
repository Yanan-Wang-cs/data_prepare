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

def get_video_info(base,save_base,q):
   
    for idname in os.listdir(base):
        idpath = os.path.join(base,idname)
        save_path = os.path.join(save_base,idname)
        for videoname in os.listdir(idpath):
            videopath = os.path.join(idpath,videoname)
            frame_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.mp4')]
            info_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.npy')]
            if len(frame_names) == 0:
                continue
            q.put([frame_names[0],info_names,save_path,videoname])
           


def process_frame(q1,align=True,scale=1.8,size=512):
    kk = 1
    while True:
        
        frame_path,info_names,save_base,videoname = q1.get()
        if frame_path is None:
            break 
        video_reader = imageio.get_reader(frame_path)
        for k,info_name in enumerate(info_names):
            print(info_names, frame_path)
            info = np.load(info_name)
            save_path = os.path.join(save_base,'%s-%04d'%(videoname,k))
            os.makedirs(save_path,exist_ok=True)
            for (i,x,y,w,h) in info:
                try:
                    img = video_reader.get_data(int(i))
                    cv2.imwrite(os.path.join(save_path,'%04d.png'%int(i)),img[...,::-1])
                    kk += 1
                except:
                    continue
    
        video_reader.close()
    print()


if __name__ == "__main__":

    mp.set_start_method('spawn')
    m = mp.Manager()
    q1 = m.Queue()
    base = 'video/'
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
   
import numpy as np
import cv2
import torch

def new_crop_with_padding(image, faceBox,scale=1.8,size=512,align=True):
    cx_box = (faceBox[0] + faceBox[2]) / 2.
    cy_box = (faceBox[1] + faceBox[3]) / 2.
    width = faceBox[2] - faceBox[0] + 1
    height = faceBox[3] - faceBox[1] + 1
    face_size = max(width, height)
    bbox_size = min(int(face_size * scale),image.shape[1],image.shape[0])
   
    x_min = max(0,int(cx_box-bbox_size / 2.))
    y_min = max(0,int(cy_box-bbox_size / 2.))
    x_max = x_min + bbox_size
    y_max = y_min + bbox_size
    if x_max > image.shape[1]:
        x_max = image.shape[1]
        x_min = x_max - bbox_size
    if y_max > image.shape[0]:
        y_max = image.shape[0]
        y_min = y_max - bbox_size

    boundingBox = [max(x_min, 0), max(y_min, 0), min(x_max, image.shape[1]), min(y_max, image.shape[0])]
    boundingBox = [int(x) for x in boundingBox]
    return boundingBox
    
def crop_with_padding(image, lmks,scale=1.8,size=512,align=True):

    img_box = [np.min(lmks[:, 0]), np.min(lmks[:, 1]), np.max(lmks[:, 0]), np.max(lmks[:, 1])]

    center = ((img_box[0] + img_box[2]) / 2.0, (img_box[1] + img_box[3]) / 2.0)

    if align:
        lm_eye_left      = lmks[36 : 42]  # left-clockwise
        lm_eye_right     = lmks[42 : 48]  # left-clockwise

        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        angle = np.arctan2((eye_right[1] - eye_left[1]), (eye_right[0] - eye_left[0])) / np.pi * 180

        RotateMatrix = cv2.getRotationMatrix2D(center, angle, scale=1)
       
        rotated_img = cv2.warpAffine(image, RotateMatrix, (image.shape[1], image.shape[0]))
        rotated_lmks = apply_transform(RotateMatrix, lmks)
    else:
        rotated_img = image 
        rotated_lmks = lmks 
        RotateMatrix = np.array([[1,0,0],
                                [0,1,0]])

    faceBox = [np.min(rotated_lmks[:, 0]), np.min(rotated_lmks[:, 1]),
                np.max(rotated_lmks[:, 0]), np.max(rotated_lmks[:, 1])]

    cx_box = (faceBox[0] + faceBox[2]) / 2.
    cy_box = (faceBox[1] + faceBox[3]) / 2.
    width = faceBox[2] - faceBox[0] + 1
    height = faceBox[3] - faceBox[1] + 1
    face_size = max(width, height)
    bbox_size = int(face_size * scale)
   

   
    
    x_min = int(cx_box-bbox_size / 2.)
    y_min = int(cy_box-bbox_size / 2.)
    x_max = x_min + bbox_size
    y_max = y_min + bbox_size

    boundingBox = [max(x_min, 0), max(y_min, 0), min(x_max, rotated_img.shape[1]), min(y_max, rotated_img.shape[0])]
    imgCropped = rotated_img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
    imgCropped = cv2.copyMakeBorder(imgCropped, max(-y_min, 0), max(y_max - image.shape[0], 0), max(-x_min, 0),
                                    max(x_max - image.shape[1], 0),cv2.BORDER_CONSTANT,value=(0,0,0))
    boundingBox = [x_min, y_min, x_max, y_max]

    scale_h = size / float(bbox_size)
    scale_w = size / float(bbox_size)
    rotated_lmks[:, 0] = (rotated_lmks[:, 0] - boundingBox[0]) * scale_w
    rotated_lmks[:, 1] = (rotated_lmks[:, 1] - boundingBox[1]) * scale_h
    # print(imgCropped.shape)
    imgResize = cv2.resize(imgCropped, (size, size))

    ### 计算变换(原图->crop box)
    m1 = np.concatenate((RotateMatrix,np.array([[0.0,0.0,1.0]])), axis=0) #rotate(+translation)
    m2 = np.eye(3) #translation
    m2[0][2] = -boundingBox[0]
    m2[1][2] = -boundingBox[1]
    m3 = np.eye(3) #scaling
    m3[0][0] = m3[1][1] = scale_h 
    m = np.matmul(np.matmul(m3,m2),m1)
    im = np.linalg.inv(m)
    info = {'rotated_lmk':rotated_lmks,
            'm':m,
            'im':im}
    
    return imgResize,info


def apply_transform(transform_matrix, lmks):
    '''
    args
        transform_matrix: float (3,3)|(2,3)
        lmks: float (2)|(3)|(k,2)|(k,3)
    
    ret
        ret_lmks: float (2)|(3)|(k,2)|(k,3)
    '''
    if transform_matrix.shape[0] == 2:
        transform_matrix = np.concatenate((transform_matrix,np.array([[0.0,0.0,1.0]])), axis=0)
    only_one = False
    if len(lmks.shape) == 1:
        lmks = lmks[np.newaxis, :]
        only_one = True
    only_two_dim = False
    if lmks.shape[1] == 2:
        lmks = np.concatenate((lmks, np.ones((lmks.shape[0],1), dtype=np.float32)), axis=1)
        only_two_dim = True

    ret_lmks = np.matmul(transform_matrix, lmks.T).T

    if only_two_dim:
        ret_lmks = ret_lmks[:,:2]
    if only_one:
        ret_lmks = ret_lmks[0]
    
    return ret_lmks


def choose_one_detection(frame_faces,box):
    """
        frame_faces
            list of lists of length 5
            several face detections from one image

        return:
            list of 5 floats
            one of the input detections: `(l, t, r, b, confidence)`
    """
    frame_faces = list(filter(lambda x:x[-1]>0.9,frame_faces))
    if len(frame_faces) == 0:
        return None
    
    else:
        # sort by area, find the largest box
        largest_area, largest_idx = -1, -1
        for idx, face in enumerate(frame_faces):
            area = compute_iou(box,face)
            # area = abs(face[2]-face[0]) * abs(face[1]-face[3])
            if area > largest_area:
                largest_area = area
                largest_idx = idx
        
        if largest_area < 0.1:
            return None
        
        retval = frame_faces[largest_idx]
        
       
    return np.array(retval).tolist()
def is_box_inside(big_box, small_box):
    """
    判断小框是否在大框内
    Args:
        big_box: (x2_min, y2_min, x2_max, y2_max) 大框的坐标
        small_box: (x1_min, y1_min, x1_max, y1_max) 小框的坐标
    Returns:
        bool: 小框是否完全位于大框内
    """
    x2_min, y2_min, x2_max, y2_max = big_box
    x1_min, y1_min, x1_max, y1_max = small_box

    # 判断小框的所有顶点是否都在大框内
    return (x1_min >= x2_min and y1_min >= y2_min and
            x1_max <= x2_max and y1_max <= y2_max)

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x0, y0, x1, y1), which reflects
            (top, left, bottom, right)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    if S_rec1 / S_rec2 > 10 or S_rec2 / S_rec1 > 10:
        return 0
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
        # return intersect / S_rec2
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x1 = x[0]
    x2 = x[0] + x[2]
    y1 = x[1]
    y2 = x[1] + x[3]
    return [x1,y1,x2,y2]

def images_to_video(images, output_video, fps=30):
    height,width = images[0].shape[:2]
    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码格式
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 将每一张图片写入视频
    for image in images:
        opencv_image = np.array(image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        video.write(opencv_image)

    # 释放资源
    video.release()
    # print(f"视频已保存为 {output_video}, video length: {len(images)}")
import cv2
import numpy as np

video_path = '/l/users/yanan.wang/project/dataPrepare/video_1_single/id00017/E6aqL_Nc410/E6aqL_Nc410_clip_8501_8573.mp4'
label_path = video_path.replace('mp4', 'npy')

# video_path = '/l/users/yanan.wang/project/dataPrepare/video_1/id00017/5MkXgwdrmJw/5MkXgwdrmJw.mp4'
# label_path = '/l/users/yanan.wang/project/dataPrepare/video_1/id00017/5MkXgwdrmJw_face_detect.npy'

# 加载标签和视频
info = np.load(label_path)  # 标签格式为 [帧索引, x1, y1, x2, y2, landmarks...]
cap = cv2.VideoCapture(video_path)

# 检查视频是否打开成功
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# 遍历标签并绘制
for lmks in info:
    frame_idx = int(lmks[0])  # 当前帧索引

    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Cannot read frame {frame_idx}")
        continue
    h,w,c = frame.shape
    # 绘制矩形框
    x1, y1, x2, y2 = map(int, lmks[1:5])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 绘制关键点
    landmarks = lmks[5:]
    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(frame, (point_x, point_y), 5, clors[i], -1)

    # 保存标注结果
    output_path = f'output_frames/{frame_idx:04d}.jpg'
    cv2.imwrite(output_path, frame)
    print(f"Saved annotated frame: {output_path}")

# 释放资源
cap.release()
print("Processing complete.")

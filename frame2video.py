import cv2
import os
from pathlib import Path

def images_to_video(image_folder, output_video, fps=30):
    # 获取所有图片文件并按文件名排序
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # 默认按文件名顺序排序
    # 读取第一张图片来确定视频的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码格式
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 将每一张图片写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if 'keypoint' in image_folder:
            frame = cv2.resize(frame, (width, height))
        video.write(frame)

    # 释放资源
    video.release()
    print(f"视频已保存为 {output_video}")

# 使用示例
replace_name='head'
image_folder = f'/l/users/yanan.wang/project/MixHeadSwap/dataset/stablevideo/{replace_name}'  # 替换为图片文件夹路径


gt_frames = sorted(list(Path(image_folder).rglob("*.[jp][pn]g")))
gt_videos = set()

# Add unique parent folders of frames
for frame in gt_frames:
    folder = str(frame.parent)
    if folder not in gt_videos:
        gt_videos.add(folder)

# Convert set back to list if needed
gt_videos = list(gt_videos)

for video in gt_videos:
    save_path = video.replace(replace_name, f'stablevideo_video/{replace_name}')
    if not os.path.exists(save_path+'/output.mp4'):
        os.makedirs(save_path,exist_ok=True)
        images_to_video(video, save_path+'/output.mp4', fps=30)

# mpath = '/l/users/yanan.wang/project/MixHeadSwap/animatediff_video_concatkeypoint2/inference'
# images_to_video(mpath, mpath+'/output.mp4', fps=30)


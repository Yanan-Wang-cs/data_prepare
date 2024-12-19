import cv2
from pathlib import Path
import os
import shutil

def copyvideos(root_dir):
    # 遍历所有子文件夹中的 jpg 和 png 图片
    for video_path in Path(root_dir).rglob("*.mp4"):
        # print(video_path)
        path_obj = Path(video_path)
        save_path = 'dataset/video/' + str(Path(*path_obj.parts[-3:-2])) +'/'+ str(Path(*path_obj.parts[-1:]))
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shutil.copy(video_path, save_path)
        print(save_path)
        




def extract_frames(video_path, output_folder):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print('无法打开视频文件')
        return

    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 1400

    while True:
        ret, frame = video.read()

        if not ret:
            break

        output_path = os.path.join(output_folder, f'{frame_count:05d}.jpg')

        cv2.imwrite(output_path, frame)

        frame_count += 1

    video.release()
    print(f'已成功提取{frame_count}帧到文件夹：{output_folder}')

# 示例：遍历 "gt" 文件夹中的所有图片
# copyvideos("../../dataset/done/video_cache/")

video_path = '/l/users/yanan.wang/project/LivePortrait/animations/1072--source_video.mp4'
output_folder = '/l/users/yanan.wang/project/dataPrepare/target'
extract_frames(video_path, output_folder)
# CUDA_VISIBLE_DEVICES=2 python data2_iouclips.py --dataset_folder /l/users/yanan.wang/project/dataPrepare/video_1 --folder video_1
# CUDA_VISIBLE_DEVICES=2 python data2_iouclips.py --dataset_folder /l/users/yanan.wang/project/dataPrepare/video_2 --folder video_2
# CUDA_VISIBLE_DEVICES=2 python data2_iouclips.py --dataset_folder /l/users/yanan.wang/project/dataPrepare/video_3 --folder video_3
# CUDA_VISIBLE_DEVICES=2 python data2_iouclips.py --dataset_folder /l/users/yanan.wang/project/dataPrepare/video_4 --folder video_4
# CUDA_VISIBLE_DEVICES=2 python data2_iouclips.py --dataset_folder /l/users/yanan.wang/project/dataPrepare/video_5 --folder video_5

CUDA_VISIBLE_DEVICES=2 python data3_mask_lmks.py --dataset_folder /l/users/yanan.wang/project/dataPrepare/video_3_single --save_folder /l/users/yanan.wang/project/MixHeadSwap/dataset/stablevideo_clips
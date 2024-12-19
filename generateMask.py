import urllib
import cv2
import math
import numpy as np
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
from PIL import Image, ImageDraw
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn

BG_COLOR = (0, 0, 0) # gray
MASK_COLOR = (255, 255, 255) # white

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim=478*3, hidden_dim=128, output_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        """
        sequence: (batch_size, seq_len, input_dim)
        """
        _, (hidden, _) = self.lstm(sequence)
        features = self.fc(hidden.squeeze(0))
        return features
    
def processing(save_name, image_file_name, segmenter):
  
  # Create the MediaPipe image file that will be segmented
  image = mp.Image.create_from_file(image_file_name)

  # Retrieve the masks for the segmented image
  segmentation_result = segmenter.segment(image)
  category_mask = segmentation_result.category_mask

  # Generate solid color images for showing the output segmentation mask.
  image_data = image.numpy_view()
  fg_image = np.zeros(image_data.shape, dtype=np.uint8)
  fg_image[:] = MASK_COLOR
  bg_image = np.zeros(image_data.shape, dtype=np.uint8)
  bg_image[:] = BG_COLOR

  my_mask = category_mask.numpy_view().copy()
  my_mask.flags.writeable = True
  my_mask[my_mask==2] =0
  my_mask[my_mask==4] =0
#   value_to_color = {
#     1: (255, 0, 0),    # 红色
#     2: (0, 255, 0),    # 绿色
#     3: (0, 0, 255),    # 蓝色
#     4: (255, 255, 0),  # 黄色
#     5: (0, 255, 255),  # 青色
#     6: (255, 0, 255),  # 品红
#     7: (128, 0, 0),    # 深红
#     8: (0, 128, 0),    # 深绿
#     9: (0, 0, 128),    # 深蓝
#     10: (128, 128, 0), # 橄榄色
#     11: (0, 128, 128), # 蓝绿色
#     12: (128, 0, 128), # 紫色
#     13: (192, 192, 192), # 浅灰
#     14: (128, 128, 128), # 深灰
#     15: (255, 165, 0),  # 橙色
#     16: (255, 20, 147), # 深粉红
#     17: (75, 0, 130),   # 靛蓝
#     18: (255, 105, 180),# 热粉色
#     19: (124, 252, 0),  # 春绿
#     20: (70, 130, 180)  # 钢蓝
# }
#   colored_image = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
#   for value, color in value_to_color.items():
#     colored_image[my_mask == value] = color
  # colored_image = Image.fromarray(colored_image)
  condition = np.stack((my_mask,) * 3, axis=-1) > 0.2
  output_image = np.where(condition, fg_image, bg_image)

  # print(f'Segmentation mask of {image_file_name}:')
  gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
  _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if len(contours) == 0:
    os.remove(image_file_name)
    print(f'==============={image_file_name} not detext face!=========')
  else:
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary_image, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    
    os.makedirs(os.path.dirname(save_name),exist_ok=True)
    # os.makedirs(os.path.dirname(save_name.replace('mask','colored_mask')),exist_ok=True)
    result = cv2.bitwise_and(output_image, output_image, mask=mask)
    # parse_image = cv2.bitwise_and(colored_image, colored_image, mask=mask)
    
    try:
      getkeypoint(image_file_name, result, detector_key,save_name)
      cv2.imwrite(save_name,result)
      # cv2.imwrite(save_name.replace('mask','colored_mask'),parse_image)
    except:
      print('keypoint error')


    
def getkeypoint(image_file_name, mask, detector, save_name):
  rgb_image = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
  pil_image = Image.fromarray(rgb_image)
  W, H = pil_image.size
  bbox = pil_image.getbbox()
  origin = Image.open(image_file_name).convert('RGB')
  head = origin.crop(bbox)
  image =mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(head))

  # STEP 4: Detect face landmarks from the input image.
  detection_result = detector.detect(image)
  face_landmarks_list = detection_result.face_landmarks[0]
  result = [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks_list]
  image = Image.new("RGB", (W, H), color="black")
  draw = ImageDraw.Draw(image)
  for x_percent, y_percent, z in result:
    # 百分比转像素
    pixel_x = int(x_percent * W)
    pixel_y = int(y_percent * H)
    
    # 绘制点
    draw.ellipse((pixel_x - 3, pixel_y - 3, pixel_x + 3, pixel_y + 3), fill="red")
  save_name = save_name.replace('mask','keypoint')
  os.makedirs(os.path.dirname(save_name),exist_ok=True)
  image.save(save_name)

  with open(save_name.replace('jpg','json').replace('png', 'json'), "w") as file:
    json.dump(result, file)
  # points_array = np.array(result).flatten().reshape(1,1,-1)
  # points_array = torch.from_numpy(points_array).float()
  # model = LSTMFeatureExtractor()
  # features = model(points_array)
  # features_numpy = features.detach().numpy()
  # save_name = save_name.replace('keypoint','feature')
  # os.makedirs(os.path.dirname(save_name),exist_ok=True)
  # np.save(save_name.replace('jpg','npy').replace('png', 'npy'), features_numpy)
  # loaded_features = np.load("features.npy")
  print('finish')


# base_options_key = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
# options_key = vision.FaceLandmarkerOptions(base_options=base_options_key,
#                                        output_face_blendshapes=False,
#                                        output_facial_transformation_matrixes=False,
#                                        num_faces=1)
# detector_key = vision.FaceLandmarker.create_from_options(options_key)

# base_options = python.BaseOptions(model_asset_path='google/selfie_multiclass_256x256.tflite')
# options = vision.ImageSegmenterOptions(base_options=base_options,
#                                       output_category_mask=True)
# image_path = '/l/users/yanan.wang/project/dataPrepare/source_frame/01400.jpg'
# # Create the image segmenter
# mask_path = image_path.replace('frame','mask')
# feature_path = image_path.replace('frame','feature')
# if not os.path.exists(mask_path):
#   with vision.ImageSegmenter.create_from_options(options) as segmenter:
#     processing(image_path.replace('frame','mask'), image_path, segmenter)
# elif not os.path.exists(feature_path):
#   getkeypoint(image_path, cv2.imread(mask_path), detector_key, mask_path)





  

def mixImage():
  pic1 = 'google/data/00017_0994.png'
  pic2 = 'google/data/00061_2119.png'
  mask1 = cv2.imread(pic1.replace('data','result'), 0)
  mask2 = cv2.imread(pic2.replace('data','result'), 0)
  cv2.imwrite('mask2.png', mask2)
  cv2.imwrite('mask1.png', 255-mask1)
  img1 = cv2.imread(pic1)
  img2 = cv2.imread(pic2)
  head1 = cv2.bitwise_and(img1, img1, mask=mask1)
  head2 = cv2.bitwise_and(img2, img2, mask=mask2)
  # cv2.imwrite('head1.png', head1)remaining_area
  # cv2.imwrite('head2.png', head2)
  coords1 = np.column_stack(np.where(mask1 > 0))
  y1, x1, h1, w1 = cv2.boundingRect(coords1)

  coords2 = np.column_stack(np.where(mask2 > 0))
  y2, x2, h2, w2 = cv2.boundingRect(coords2)

  cropped_1 = head1[y1:y1+h1, x1:x1+w1]
  cropped_2 = head2[y2:y2+h2, x2:x2+w2]
  mask_crop = mask2[y2:y2+h2, x2:x2+w2]
  
  # cv2.imwrite('crop1.png', cropped_1)
  # cv2.imwrite('crop2.png', cropped_2)
  ratio = h1/h2
  h_resize = int(h2 * ratio)
  w_resize = int(w2 * ratio)
  resized_image2 = cv2.resize(cropped_2, (w_resize, h_resize), interpolation=cv2.INTER_AREA)
  mask_resize = cv2.resize(mask_crop, (w_resize, h_resize), interpolation=cv2.INTER_AREA)
  
  start_x = x1 + w1//2 - w_resize//2

  mask_after_resize = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
  mask_after_resize[y1:y1+h1, start_x:start_x+w_resize] = mask_resize
  mask_inv = cv2.bitwise_not(mask_after_resize)
  remaining_area = cv2.bitwise_and(mask1, mask_inv)

  cv2.imwrite('mask_after_resize.png', mask_after_resize)
  cv2.imwrite('remaining_area.png', remaining_area)
  bg1 = cv2.bitwise_and(img1, img1, mask=255-mask1)
  # cv2.imwrite('bg.png', bg1)
  
  
  mask_normalized = mask_resize / 255.0
  mask_3d = np.stack([mask_normalized] * 3, axis=-1)
  blended_image = resized_image2 * mask_3d + bg1[y1:y1+h1, start_x:start_x+w_resize] * (1 - mask_3d)

  bg1[y1:y1+h1, start_x:start_x+w_resize] = blended_image
  cv2.imwrite('result.png', bg1)

def doInpainting():
  pipe = AutoPipelineForInpainting.from_pretrained("stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
  image = load_image('bg2.png').resize((1024,1024))
  mask_image = load_image('remaining_area.png').resize((1024,1024))
  prompt = "A close-up mug shot of a man being interviewed"
  generator = torch.Generator(device="cuda").manual_seed(0)
  image = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    guidance_scale=8.0,
    num_inference_steps=20,  # steps between 15 and 30 work well for us
    strength=0.99,  # make sure to use `strength` below 1.0
    generator=generator,
  ).images[0]
  image.save('mytest.png')

def getMask(path):
  IMAGE_FILENAMES = [path+x for x in os.listdir(path)]

  # Create the options that will be used for ImageSegmenter
  base_options = python.BaseOptions(model_asset_path='google/selfie_multiclass_256x256.tflite')
  options = vision.ImageSegmenterOptions(base_options=base_options,
                                        output_category_mask=True)

  # Create the image segmenter
  with vision.ImageSegmenter.create_from_options(options) as segmenter:

    # Loop through demo image(s)
    for image_file_name in IMAGE_FILENAMES:
      save_name = image_file_name.replace('frame','mask')
      if os.path.exists(save_name):
        continue
      processing(save_name, image_file_name, segmenter)
      print(image_file_name)

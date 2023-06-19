### RUN WITH: python .\Image_prep_crop_resize.py ##

import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch
# pip install -U ultralytics


model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)


#ViT model: pretrained on a dataset consisting of images with the dimension of 224 x 224 pixels.
#RESNET18 model: inference images are resized to resize_size=[256] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[224].
#Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

# resize 224x224 for optimal performance
#resizeshape = 224

for subdir, dirs, files in os.walk('./apple_original_augm/Train'):
    for file in files:
        filepath = subdir + os.sep + file
        last_folder_name = os.path.basename(os.path.normpath(subdir))

        # to-do pre-check for all image filetypes
        if filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".png"):
            img = Image.open(filepath).convert('RGB')
            # if img.size != (resizeshape,resizeshape):
            result = model(img)

            boxes = result.xyxy[0].cpu().numpy()[:, :4]
            if len(boxes) >= 1:  # on blotch 7 it will detect nothing, so no index =  error, on blotch 25-26it will detect the apple blurred in the front (still an apple but not blotched) i deleted these
                x1, y1, x2, y2 = boxes[0]
                cropped_image = T.functional.crop(
                    img, y1, x1, (y2-y1), (x2-x1))
                #resize = T.Resize((resizeshape, resizeshape))
                #resized_image = resize(cropped_image)

                # Create the output directory if it doesn't exist
                output_dir = './apple_original_augm_cropped/' + last_folder_name
                os.makedirs(output_dir, exist_ok=True)

                # Save the crop
                cropped_image.save(os.path.join(output_dir, file))
                #resized_image.save(os.path.join(output_dir, file))


# for subdir, dirs, files in os.walk('./new_apples_224'):
#     for file in files:
#         filepath = subdir + os.sep + file

#         # to-do pre-check for all image filetypes
#         if filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".png"):
#             img = Image.open(filepath)
#             if img.size != (resizeshape,resizeshape):
#                 # had to delete one because the jpg was not parseable somehow, blotch #17 is png ipv jpg look into it later
#                 rgb_im = img.convert('RGB')
#                 rgb_im_resized = rgb_im.resize(
#                     (resizeshape, resizeshape)) 
#                 rgb_im_resized.save(filepath)

# to run: $python Image_prep_crop_resize.py
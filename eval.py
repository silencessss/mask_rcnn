import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
import glob
import argparse
from mrcnn.config import Config
from datetime import datetime
from scipy.misc import imsave
##########################################################################
parser = argparse.ArgumentParser(description='Resize images in a folder')
parser.add_argument('--input',    '-i', help='input image folder', required=True)
args = parser.parse_args()
# Validate parameters
input_path = os.path.abspath(args.input)
if not os.path.isdir(args.input):
    print('--input ({}) must be a folder.'.format(args.input))
    sys.exit(1)

##########################################################################
# Root directory of the project
ROOT_DIR = os.getcwd()
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "model_save")
##########################################################################
# Local path to trained weights file
# EVAL_MODEL_PATH = os.path.join(MODEL_DIR,"maskrcnn_config20200814T1659/mask_rcnn_maskrcnn_config_0187.h5")
#EVAL_MODEL_PATH=input_path
for root, _, basenames in os.walk(input_path):
        for basename in basenames:
            basename_check=basename.split('.')[1]
            if(basename_check=='h5'):
                EVAL_MODEL_PATH=os.path.join(input_path,basename)
model_basename=os.path.basename(EVAL_MODEL_PATH)
model_name=model_basename.split('.')[0]
print('model from == ',EVAL_MODEL_PATH)
print('===========================================================================================================================')
# Download COCO trained weights from Releases if needed
if not os.path.exists(EVAL_MODEL_PATH):
    utils.download_trained_weights(EVAL_MODEL_PATH)
    print("cuiwei***********************")
 
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "./datasets/Data640/eval_image_640x640_nospace/")
##########################################################################
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "MaskRCNN_config"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 24  # background + 1 shapes + 1 not_defined
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (40,80,160,320,640)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    # Non-maximum suppression threshold for detection
    #DETECTION_NMS_THRESHOLD = 0.3
##########################################################################
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
 
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(EVAL_MODEL_PATH, by_name=True)
 
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
#class_names = ['BG', 'person']
class_names=['_background_',
             'cocacola',
             'cashewnuts',
             'M_MBIG',
             'M_MSMALL',
             'Hot-Chili-Squid-Flavor',
             'Original-Flavor',
             'Thai-Spicy-Mixed-Nuts',
             'Pon-De-Strawberry-Honey',
             'Pon-de-Strawberry',
             'Pon-De-Double-Chocolate',
             'Old-Fashioned',
             'Pon-De-Soybean',
             'Pringles',
             'TARO-FISH-SNACK',
             'SALTED-PEANUTS',
             'MANDARIN',
             'SPEARMINT',
             'Choco-Old-Fashioned',
             'Pon-De-Chocolate',
             'Fried-Seaweed',
             'Pon-De-Yogurt',
             'Strawberry-Ring',
             'Sugar-Ring'
            ]
##########################################################################
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
#random_image_name=random.choice(file_names)
count=1
for eval_image in file_names:
    print('eval image... ',count)
    choice_eval_image_path=os.path.join(IMAGE_DIR,eval_image)
    image = skimage.io.imread(choice_eval_image_path)
    results = model.detect([image], verbose=1)
    r = results[0]
    visualize.display_instances(eval_image,image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
    count+=1
print('===========================================================================================================================')
'''
a=datetime.now()
# Run detection
results = model.detect([image], verbose=1)
b=datetime.now()
# Visualize results
print("shijian",(b-a).seconds)
r = results[0]

#print('r == ',r)
#display_instances in visualize.py lines 83
visualize.display_instances(random_image_name,image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
'''
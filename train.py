import os
import sys
import random
import math
import re
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import imgaug.augmenters as iaa
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image
####################################################################################################
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
ROOT_DIR = os.getcwd()
 
# ROOT_DIR = os.path.abspath("../")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR,'model_save')
 
iter_num = 0
 
# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
BASE_MODEL_PATH = os.path.join(ROOT_DIR,'base_model/mask_rcnn_maskrcnn_config_0492.h5')
# Download COCO trained weights from Releases if needed
if not os.path.exists(BASE_MODEL_PATH):
    utils.download_trained_weights(BASE_MODEL_PATH)
 
############################################################
#  Configurations
############################################################ 
class myConfig(Config):
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
    NUM_CLASSES = 1 + 24   # 1 background + 1 shapes + 1 not defined
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640
 
    # Use smaller anchors because our image and objects are small,48,96,192,384,768
    # Anchor have to set as 2^n*k
    # Ex, resolution is 512x512, then set as (32, 64, 128, 256, 512)(20,40,80,160,320)
    RPN_ANCHOR_SCALES = (20,40,80,160,320)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    # Learning rate
    LEARNING_RATE=0.002
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=30

config = myConfig()
config.display()
print('=====================================================================================================================')
############################################################
#  Dataset
############################################################
class myDataset(utils.Dataset):
    # get object from image
    def get_obj_index(self, image):
        n = np.max(image)
        return n
 
    # analysis .ymal file from labelme，get label from yaml
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.safe_load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels
 
    # rewrite draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        # print("info-->",info)
        # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask
 
    
    # self.image_info add path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_datas(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        class_name=['_background_',
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
        for i in range(0,24):
            self.add_class('datas', i , class_name[i])
 
        for i in range(count):
            print('loading...',i)
            filestr = imglist[i].split(".")[0]
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + "_label.png"
            #print('mask_path  ',mask_path)
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/"+filestr+".yaml"
            #print('yaml_path  ',yaml_path)
            print(dataset_root_path + "labelme_json/" + filestr + "_json/"+filestr+".png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/"+filestr+".png")
 
            self.add_image('datas', image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
 
    # rewirte load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        class_name=['_background_',
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
        global iter_num
        #print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
 
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        
        #print('len(class_name) == ',len(class_name))
        #print('len(labels) == ',len(labels))
        for i in range(len(labels)):
            if labels[i].find('_background_') != -1:
                labels_form.append('_background_')
            elif labels[i].find('cocacola') != -1:
                labels_form.append('cocacola')
            elif labels[i].find('cashewnuts') != -1:
                labels_form.append('cashewnuts')
            elif labels[i].find('M_MBIG') != -1:
                labels_form.append('M_MBIG')
            elif labels[i].find('M_MSMALL') != -1:
                labels_form.append('M_MSMALL')
            elif labels[i].find('Hot-Chili-Squid-Flavor') != -1:
                labels_form.append('Hot-Chili-Squid-Flavor')
            elif labels[i].find('Original-Flavor') != -1:
                labels_form.append('Original-Flavor')
            elif labels[i].find('Thai-Spicy-Mixed-Nuts') != -1:
                labels_form.append('Thai-Spicy-Mixed-Nuts')
            elif labels[i].find('Pon-De-Strawberry-Honey') != -1:
                labels_form.append('Pon-De-Strawberry-Honey')
            elif labels[i].find('Pon-de-Strawberry') != -1:
                labels_form.append('Pon-de-Strawberry')
            elif labels[i].find('Pon-De-Double-Chocolate') != -1:
                labels_form.append('Pon-De-Double-Chocolate')
            elif labels[i].find('Choco-Old-Fashioned') != -1:
                labels_form.append('Choco-Old-Fashioned')
            elif labels[i].find('Pon-De-Soybean') != -1:
                labels_form.append('Pon-De-Soybean')
            elif labels[i].find('Pringles') != -1:
                labels_form.append('Pringles')
            elif labels[i].find('TARO-FISH-SNACK') != -1:
                labels_form.append('TARO-FISH-SNACK')
            elif labels[i].find('SALTED-PEANUTS') != -1:
                labels_form.append('SALTED-PEANUTS')
            elif labels[i].find('MANDARIN') != -1:
                labels_form.append('MANDARIN')
            elif labels[i].find('SPEARMINT') != -1:
                labels_form.append('SPEARMINT')
            elif labels[i].find('Old-Fashioned') != -1:
                labels_form.append('Old-Fashioned')
            elif labels[i].find('Pon-De-Chocolate') != -1:
                labels_form.append('Pon-De-Chocolate')
            elif labels[i].find('Fried-Seaweed') != -1:
                labels_form.append('Fried-Seaweed')
            elif labels[i].find('Pon-De-Yogurt') != -1:
                labels_form.append('Pon-De-Yogurt')
            elif labels[i].find('Strawberry-Ring') != -1:
                labels_form.append('Strawberry-Ring')
            elif labels[i].find('Sugar-Ring') != -1:
                labels_form.append('Sugar-Ring')


        '''
        for i in range(len(labels)):
            for j in range(0,len(class_name)):
                if labels[i].find(class_name[j]) !=-1:
                    #print('get == ',class_name[j])
                    labels_form.append(class_name[j])
        
        for i in range(len(labels)):
            if labels[i].find("person") != -1:
                # print "car"
                labels_form.append("person")
            elif labels[i].find("leg") != -1:
                # print "leg"
                labels_form.append("leg")
            elif labels[i].find("well") != -1:
                # print "well"
                labels_form.append("well")
        '''
        #print('labels_form == ',labels_form)
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)
 
 
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def loss_visualize(epoch, tra_loss, val_loss):
    plt.style.use("ggplot")
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title("Epoch_Loss")
    plt.plot(epoch, tra_loss, label='train_loss', color='r', linestyle='-', marker='o')
    plt.plot(epoch, val_loss, label='val_loss', linestyle='-', color='b', marker='^')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss.jpg')
    #plt.show()
 
############################################################
# Loading data for train and val (test) 
############################################################
# set training data
dataset_root_path_train = "./datasets/Data640/trainData_640x640_nospace03_newsliding/"
#dataset_root_path_train = "./datasets/Data640/trainData_640x640_nospace01_newlabel/"
img_floder_train = dataset_root_path_train + "pic"
mask_floder_train = dataset_root_path_train + "cv2_mask"
# yaml_floder = dataset_root_path
imglist_train = os.listdir(img_floder_train)
#print('imglist == ',imglist)
count_train = len(imglist_train)

dataset_train = myDataset()
dataset_train.load_datas(count_train, img_floder_train, mask_floder_train, imglist_train, dataset_root_path_train)
dataset_train.prepare()
print('loading Training Data images END...') 
#print("dataset_train-->",dataset_train._image_ids)
print('========================================================================================================================') 

# set testing data
dataset_root_path_test = "./datasets/Data640/testData_640x640_nospace03_newsliding/"
#dataset_root_path_test = "./datasets/Data640/testData_640x640_nospace01_newlabel/"
img_floder_test = dataset_root_path_test + "pic"
mask_floder_test = dataset_root_path_test + "cv2_mask"
# yaml_floder = dataset_root_path
imglist_test = os.listdir(img_floder_test)
#print('imglist == ',imglist)
count_test = len(imglist_test)

dataset_val = myDataset()
dataset_val.load_datas(count_test, img_floder_test, mask_floder_test, imglist_test, dataset_root_path_test)
dataset_val.prepare()
print('loading Validation Data images END...') 
#print("dataset_val-->",dataset_val._image_ids)
print('========================================================================================================================')
print('Total train images are...       ',count_train)
print('Total validation images are ... ',count_test)
print('========================================================================================================================')  


# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
 
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
 
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
 
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    print('BASE_MODEL_PATH == ',BASE_MODEL_PATH)
    model.load_weights(BASE_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
############################################################
# Image Augmentation
############################################################
augSometimes = lambda aug: iaa.Sometimes(0.5, aug)
augmentation = imgaug.augmenters.Sequential([
        augSometimes(imgaug.augmenters.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5))), 
        augSometimes(imgaug.augmenters.GaussianBlur(sigma=(0, 0.25))),
        augSometimes(imgaug.augmenters.Add((-10, 10), per_channel=0.5)),
        augSometimes(imgaug.augmenters.Multiply((0.5, 1.5), per_channel=0.5)),
        augSometimes(imgaug.augmenters.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
        augSometimes(imgaug.augmenters.Cutout(nb_iterations=2))
   ])
############################################################
# Train Layer
############################################################
'''
layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
'''
# Stage 1 - Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=60,
            layers='heads',
            augmentation=augmentation
            )
 
# Stage 2 - Finetune layers from ResNet stage 4 and up
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=150,
            layers='4+',
            augmentation=augmentation
            )

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=300,
            layers="all",
            augmentation=augmentation
            )

x_epoch, y_tra_loss, y_val_loss = modellib.call_back()
loss_visualize(x_epoch, y_tra_loss, y_val_loss)
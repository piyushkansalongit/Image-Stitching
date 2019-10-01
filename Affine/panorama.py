import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from bunch import Bunch
import pickle
from configs.utils.config_utils import process_config
from featureExtraction import featureExtactor
from transformExtraction import transformExtactor
from stitching import warpAndStich
from blending import blender
import copy

def preProcess(config):
    print(['INFO Preprocessing Images...'])
    for image_name in os.listdir(config.image_path):
        image = cv2.imread(os.path.join(config.image_path, image_name))
        image = cv2.resize(image, (config.image_width, config.image_height))
        cv2.imwrite(os.path.join(config.image_path, image_name), image)

def FeatureExtraction(config):
    print('[INFO] Extracting Features...')

    features = featureExtactor(config)

    descriptorSet = []
    keyPointSet = []
    images = os.listdir(features.image_path)
    images = [int(image.replace(".jpg",'')) for image in images]
    images.sort()
    for image_name in images:
        # Read an image 
        image_name = str(image_name)+".jpg"
        image_path = os.path.join(features.image_path, image_name)
        image = cv2.imread(image_path)
        # Convert RGB to gray for feeding SIFT.
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Detect the keypoints and 128-D descriptors for the image.
        (keypoints, descriptors) = features.detectAndDescribe(gray_image, features.keypoint_detector)
        
        #Drawing all the keypoints over the image
        image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
        #Saving the highlighted image
        cv2.imwrite(os.path.join(features.keypoint_highlight_path, image_name), image)

        #Piping the descriptors in a list
        descriptorSet.append(descriptors)
        keyPointSet.append(keypoints)

    return(descriptorSet, keyPointSet)

def RANSACEstimator(config, descriptors, keyPoints):
    print('[INFO] Extracting Transformations...')
    num_matches = []
    transform = transformExtactor(config)
    for i in range(len(descriptors)-1):
        for j in range(i+1, len(descriptors)):
            set_1 = descriptors[i]
            set_2 = descriptors[j]
            key_1 = keyPoints[i]
            key_2 = keyPoints[j]
            matches = transform.RANSAC(i, j, key_1, key_2, set_1, set_2)
            num_matches.append(matches)
    num_matches = np.array(num_matches)
    np.save(os.path.join(config.num_matches, "matches"), num_matches)


def WarpStitch(config):
    print('[INFO] Warping and Stiching...')
    stitcher = warpAndStich(config)
    stitcher.stitch()


if __name__ == '__main__':

    try:
        config = process_config('configs/configs.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)
    
    # Process the images before sending to the pipeline.
    preProcess(config)

    # Features Descriptors for each of the images.
    descriptors, keyPoints = FeatureExtraction(config)

    # Pair-wise transformation matrix for the images.
    RANSACEstimator(config, descriptors, keyPoints)

    WarpStitch(config)




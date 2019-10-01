import os
import numpy as np
import cv2

class featureExtactor:
    def __init__(self, config=None):
        self.filename = config.filename
        self.image_path = config.image_path
        self.keypoint_detector = config.keypoint_detector
        self.keypoint_highlight_path = config.keypoint_path
        self.minHessian = config.minHessian

    def detectAndDescribe(self, image, method=None):
        """
        Compute key points and feature descriptors using a specific method.
        """

        if method == 'sift':
            descriptor = cv2.xfeatures2d.SIFT_create()
        elif method == 'surf':
            descriptor = cv2.xfeatures2d_SURF.create(hessianThreshold=self.minHessian)
        elif method =='brisk':
            descriptor = cv2.xfeatures2d.BRISK_create()  
        elif method == 'orb':
            descriptor = cv2.ORB_create()

        (kps, features) = descriptor.detectAndCompute(image, None)

        return (kps, features)
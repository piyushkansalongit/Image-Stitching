import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path
import copy
import time
from blending import blender
from exposure import exposureBalance
class warpAndStich:
    def __init__(self, config=None):
        self.homography_path = config.homography_path
        self.image_path = config.image_path
        self.stitched_path = config.stitched_path
        self.smoothing_window_size = config.smoothing_window_size
        self.num_matches = config.num_matches
        self.base_image = config.base_image
        self.file_name = config.filename
        self.balanced_path = config.balanced_path
        self.blended_path = config.blend_path
        self.panorama_size = (config.panorama_size[0], config.panorama_size[1])


    def findMax(self, match_array, mask):
        if(np.max(match_array) == 0):
            for i in range(len(mask)):
                if(mask[i]):
                    return i
        position = np.argmax(match_array)
        if(mask[position] == True):
            return position
        else:
            match_array[position] = 0
            return self.findMax(match_array, mask)

    def findMatch(self, match_matrix, image_order, mask):
        left_element = image_order[0]
        right_element = image_order[-1]
        if(left_element == right_element):
            return 1, self.findMax(copy.deepcopy(match_matrix[left_element]), mask)
        else:
            left_max = self.findMax(copy.deepcopy(
                match_matrix[left_element]), mask)
            # print(left_max)
            right_max = self.findMax(copy.deepcopy(
                match_matrix[right_element]), mask)
            if(match_matrix[left_element][left_max] > match_matrix[right_element][right_max]):
                return 0, left_max
            else:
                return 1, right_max


    def find_intersection(self, p1, p2, bp1, bp2):
        line1 = np.array([[p1[0],p1[1]],[p2[0],p2[1]]])
        line2 = np.array([[bp1[0],bp1[1]],[bp2[0],bp2[1]]])
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    
    def findOrder(self, base, match_matrix, flag):
        # Find all the transformation pairs that we need to take
        image_order = []
        image_order.append(base)
        homography_pairs = []
        size = len(os.listdir(self.image_path))

        mask = [True for i in range(size)]
        mask[base] = False
        count = 1
        while(count < size):
            label, next_match = self.findMatch(match_matrix, image_order, mask)
            mask[next_match] = False
            if(label == 0):
                image_order.insert(0, next_match)
                homography_pairs.append((image_order[1],next_match))
            else:
                image_order.append(next_match)
                homography_pairs.append((image_order[-2], next_match))
            count+=1
        image_order = [i+1 for i in image_order]
        homography_pairs = [(i+1, j+1) for (i,j) in homography_pairs]

        # Load all the relevant homography pairs
        homography_list = []
        for i,pair in enumerate(homography_pairs):
                if(pair[0] > pair[1]):
                    temp = (pair[1], pair[0])
                    pair = temp
                path = os.path.join(self.homography_path,str(pair[0])+"-"+str(pair[1])+".npy")
                homography_list.append(np.load(path))

        # Update all the homography transformations with reference to the base image.
        updated_homographies = [np.identity(3, dtype = float)  for i in range(size)]
        for i, pair in enumerate(homography_pairs):
            if(pair[1]>pair[0]):
                a = homography_list[i]
                b = np.linalg.inv(homography_list[i])
                temp = np.matmul(updated_homographies[pair[0]-1],np.linalg.inv(homography_list[i]))
                updated_homographies[pair[1]-1] = temp

            else:
                temp = np.matmul(updated_homographies[pair[0]-1],homography_list[i])
                updated_homographies[pair[1]-1] = temp

        if flag is 0:
            base = image_order[int(len(image_order)/2)]-1
            return self.findOrder(base, match_matrix, 1)     
        if flag is 1:
            base = image_order[int(len(image_order)/2)]-1
            return self.findOrder(base, match_matrix, 2)
        if flag is 2 or 3:
            loc = np.where(np.array(image_order)==base+1)
            H = updated_homographies[image_order[loc[0][0]+1]-1]
            images = os.listdir(self.image_path)
            h,w,_ = cv2.imread(os.path.join(self.image_path, images[0])).shape
            point = np.float32([ [w/2,h/2] ]).reshape(-1,1,2)
            right_center = cv2.perspectiveTransform(point, H)
            if(right_center[0][0][0] < w/2):
                image_order.reverse()
            print(image_order)
            if flag is 2:
                if(len(image_order)%2 is 0):
                    base = image_order[int(len(image_order)/2)]-1
                else:
                    base = image_order[int(len(image_order)/2)]-1
                return self.findOrder(base, match_matrix, 3)
        
        return base, image_order, updated_homographies
    

    def stitch(self):  
        # List of images
        images = os.listdir(os.path.join(self.image_path))
        images = [int(image.replace(".jpg",'')) for image in images]
        images.sort()
        images = [ str(image_name)+".jpg" for image_name in images]

        # Construct a matrix for the number of matches of each image with every other image.
        num_matches = np.load(os.path.join(self.num_matches,"matches.npy"))
        size = len(os.listdir(self.image_path))
        match_matrix = np.zeros((size, size))
        k = 0
        for i in range(0, size-1):
            for j in range(i+1, size):
                match_matrix[i][j] = num_matches[k]
                match_matrix[j][i] = num_matches[k]
                k+=1
        print('[INFO] The count matrix between pairs is...')
        print(match_matrix)
        i = self.base_image
        base, image_order, updated_homographies = self.findOrder(i, match_matrix, 0)
        self.base_image = base
        print(self.base_image, image_order)
        h,w,_ = cv2.imread(os.path.join(self.image_path, images[0])).shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        panorama_size = self.panorama_size
        blank = np.zeros((panorama_size[0],panorama_size[1],3),np.uint8)        
        translation = np.array([[1,0,panorama_size[0]/2],[0,1,panorama_size[1]/2],[0,0,1]])
        for i,image_name in enumerate(images):
            image = cv2.imread(os.path.join(self.image_path, images[i]))
            temp = cv2.warpPerspective(image,np.matmul(translation,updated_homographies[i]), panorama_size)
            blank = np.maximum(blank, temp)
            cv2.imwrite(os.path.join(self.stitched_path,str(i+1)+".jpg"), temp)
        cv2.imwrite(os.path.join(self.blended_path, "pano" + str(self.file_name)+".jpg"), blank)
    
        #Find the masks for blending
        masks = []
        left_points = cv2.perspectiveTransform(pts, np.matmul(translation, updated_homographies[self.base_image]))
        right_points = left_points.copy()
        for i,image_name in enumerate(images):
            if(i != self.base_image):
                points = cv2.perspectiveTransform(pts,np.matmul(translation,updated_homographies[i]))
                order = np.array(image_order)
                if(np.where(order == self.base_image+1) > np.where(order == i+1)):
                    mask_points = []
                    mask_points.append((points[0][0][0], points[0][0][1]))
                    mask_points.append((points[1][0][0], points[1][0][1]))
                    mask_points.append(self.find_intersection(points[1][0], points[2][0], left_points[1][0], left_points[2][0]))
                    mask_points.append(self.find_intersection(points[0][0], points[3][0], left_points[0][0], left_points[3][0]))
                    left_points = [[[p[0], p[1]]] for p in mask_points]
                else:
                    mask_points = []
                    mask_points.append(self.find_intersection(points[0][0], points[3][0], right_points[0][0], right_points[3][0]))
                    mask_points.append(self.find_intersection(points[1][0], points[2][0], right_points[1][0], right_points[2][0]))
                    mask_points.append((points[2][0][0], points[2][0][1]))
                    mask_points.append((points[3][0][0], points[3][0][1]))
                    right_points = [[[p[0], p[1]]] for p in mask_points]
                poly_path=Path(mask_points)
                x, y = np.mgrid[:panorama_size[0], :panorama_size[1]]
                coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)
                mask = poly_path.contains_points(coors)
                mask = np.reshape(mask, (panorama_size[0], panorama_size[1]),-1)

                masks.append(np.stack([mask.astype('float32'), mask.astype('float32'), mask.astype('float32')], -1))
            else:
                masks.append(0)
        #Balance out the exposure of all the images with respect to base image.

        exp = exposureBalance(self.stitched_path, self.balanced_path)

        #Normalizing all the images to the intensity of the base image
        exp.exposureBalance()

        images = [os.path.join(self.balanced_path, image) for image in os.listdir(self.balanced_path)]
        # images = [os.path.join(self.stitched_path, image) for image in os.listdir(self.stitched_path)]
        lpb = cv2.imread(images[self.base_image])
        blend = blender()
        for i, image in enumerate(images):
            if (i != self.base_image):
                out1 = cv2.imread(images[i])
                out3 = masks[i]
                # plt.imshow(out3)
                # plt.show()
                lpb = blend.Laplacian_blending(out1,lpb,out3,5)
        
        cv2.imwrite(os.path.join(self.blended_path,"panorama" + str(self.file_name) + ".jpg"), lpb)


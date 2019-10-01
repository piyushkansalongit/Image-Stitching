import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class transformExtactor:
    def __init__(self, config=None):
        self.image_path = config.image_path
        self.match_path = config.match_path
        self.homography_path = config.homography_path
        self.ratio_thres = config.ratio_thres

    def RANSAC(self, i, j, key_1, key_2, set_1, set_2):
        """
        Compute the RANSAC between two descriptor matrices.
        """
        #Minimum number of correspondances images should have
        MIN_MATCH_COUNT = 10

        images = os.listdir(self.image_path)
        images = [int(image.replace(".jpg",'')) for image in images]
        images.sort()
        images = [ str(image_name)+".jpg" for image_name in images]

        # queryImage
        img1 = cv2.imread(os.path.join(self.image_path,images[i]),0)    
        # trainImage      
        img2 = cv2.imread(os.path.join(self.image_path,images[j]),0)          

        if img1 is None or img2 is None:
            print('Could not open or find the images!')
            exit(0)
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # print("[INFO] Extracting good matches usinng flann KNN and Lowe's ration test...")

        matches = flann.knnMatch(set_1,set_2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < self.ratio_thres*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ key_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ key_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            # print(i+1, j+1)
            # print(homography)
            matchesMask = mask.ravel().tolist()
        else:
            # print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
            return 0

        # print('[INFO] Drawing Good Matches...')
        img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
        img3 = cv2.drawMatches(img1,key_1,img2,key_2,good,img_matches,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.path.join(self.match_path, str(i+1)+"-"+str(j+1)+".jpg"), img3)
        np.save(os.path.join(self.homography_path, images[i].replace(".jpg",'')+"-"+images[j].replace(".jpg",'')), homography)
        return matchesMask.count(1)

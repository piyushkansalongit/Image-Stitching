import os
import cv2

class exposureBalance:
    def __init__(self, stitched_path, balanced_path):
        self.balanced_path = balanced_path
        self.stitched_path = stitched_path

    def exposureBalance(self):
        images = [os.path.join(self.stitched_path, image) for image in os.listdir(self.stitched_path)]
        for i,image in enumerate(images):
            img = cv2.imread(image)
            ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
            channels=cv2.split(ycrcb)
            cv2.equalizeHist(channels[0],channels[0])
            cv2.merge(channels,ycrcb)
            cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
            cv2.imwrite(os.path.join(self.balanced_path, str(i+1))+".jpg", img)
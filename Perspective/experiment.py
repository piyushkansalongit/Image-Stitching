import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
path = "/media/piyush/D/courses/COL780/Assignment2/Visuals/stitched/1"

image_path = os.path.join(path, "1.jpg")

image = cv2.imread(image_path)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(image,(3,3),0)
 
# Apply Laplacian operator in some higher datatype
laplacian = cv2.Laplacian(blur,cv2.CV_64F)
laplacian1 = laplacian/laplacian.max()
 
plt.imshow(laplacian1)
plt.show()



        img1 = image_order[0]
        img2 = image_order[1]
        l_points = cv2.perspectiveTransform(pts, updated_homographies[img1-1])
        r_points = cv2.perspectiveTransform(pts, updated_homographies[img2-1])

        l_center = (l_points[1][0][0]+l_points[2][0][0])/2
        r_center = (r_points[1][0][0]+r_points[2][0][0])/2

        if(l_center > r_center):
            i = 0
            while i < len(image_order):
                temp = image_order.pop(0)
                image_order.append(temp)
                i+=1
        print(image_order, self.base_image)




                # Transform the corners of all the images to get the global panorama size
        # x_min = 0
        # x_max = 0
        # y_min = 0
        # y_max = 0
        # for i,homo in enumerate(updated_homographies):
        #     dst = cv2.perspectiveTransform(pts, homo)
        #     if(dst[0][0][0] < x_min):
        #         x_min = dst[0][0][0]
        #     elif(dst[0][0][0] > x_max):
        #         x_max = dst[0][0][0]
        #     if(dst[0][0][1] < y_min):
        #         y_min = dst[0][0][1]
        #     elif(dst[0][0][1] > y_max):
        #         y_max = dst[0][0][1]       
            
        #     if(dst[1][0][0] < x_min):
        #         x_min = dst[1][0][0]
        #     elif(dst[1][0][0] > x_max):
        #         x_max = dst[1][0][0]
        #     if(dst[1][0][1] < y_min):
        #         y_min = dst[1][0][1]
        #     elif(dst[1][0][1] > y_max):
        #         y_max = dst[1][0][1]    

        #     if(dst[2][0][0] < x_min):
        #         x_min = dst[2][0][0]
        #     elif(dst[2][0][0] > x_max):
        #         x_max = dst[2][0][0]
        #     if(dst[2][0][1] < y_min):
        #         y_min = dst[2][0][1]
        #     elif(dst[2][0][1] > y_max):
        #         y_max = dst[2][0][1]    

        #     if(dst[3][0][0] < x_min):
        #         x_min = dst[3][0][0]
        #     elif(dst[3][0][0] > x_max):
        #         x_max = dst[3][0][0]
        #     if(dst[3][0][1] < y_min):
        #         y_min = dst[3][0][1]
        #     elif(dst[3][0][1] > y_max):
        #         y_max = dst[3][0][1]    
            
        # print('[INFO] The overall panorama coordinated x_min:', x_min, 'x_max:', x_max, 'y_min:',y_min, 'y_max:',y_max)
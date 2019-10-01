import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class blender:

    def Laplacian_blending(self, img1, img2, mask, levels=4):

        G1 = img1.copy()
        G2 = img2.copy()
        GM = mask.copy()
        gp1 = [G1]
        gp2 = [G2]
        gpM = [GM]
    
        for i in range(levels-1):
            G1 = cv2.pyrDown(G1)
            G2 = cv2.pyrDown(G2)
            GM = cv2.pyrDown(GM)
            gp1.append(np.float32(G1))
            gp2.append(np.float32(G2))
            gpM.append(np.float32(GM))


        lp1 = [gp1[levels-1]]
        lp2 = [gp2[levels-1]]
        gpMr = [gpM[levels-1]]
    
        for i in range(levels-1, 0, -1):
            L1 = np.subtract(gp1[i-1], cv2.pyrUp(gp1[i]))
            L2 = np.subtract(gp2[i-1], cv2.pyrUp(gp2[i]))
            lp1.append(L1)
            lp2.append(L2)
            gpMr.append(gpM[i-1])

        LS = []
        for i, (l1, l2, gm) in enumerate(zip(lp1, lp2, gpMr)):
            ls = l1*(gm) + l2*(1-gm)
            LS.append(ls)

        ls_ = LS[0]
        for i in range(1, levels):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[i])
        return ls_ 

        
    
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#Prepocessing image to make it ready
def preprocess(image_path1, image_path2):

    #Loading images
    image1=cv2.imread(image_path1)
    image2=cv2.imread(image_path2)
    if(image1 is None or image2 is None):
        raise FileNotFoundError("Image nor found or improper format")

    #Convert from BGR to RGB for display
    original_rgb1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
    original_rgb2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)

    #Convert to grayscale and invert
    gray1=255-(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))
    gray2=255-(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))

    return gray1, gray2, original_rgb1, original_rgb2

#Actual processing for Homograph
def Homograph(gray1, gray2, original_rgb1, original_rgb2):

    #ORB detector
    orb=cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    
    h, w = gray1.shape
    mask1=np.zeros_like(gray1)
    mask1[0:int(0.45*h), :]=255

    #keypoints and descriptors
    kp1, des1=orb.detectAndCompute(gray1,mask1)
    kp2, des2=orb.detectAndCompute(gray2,mask1)
    if(des1 is None or des2 is None):
        raise ValueError("Descriptors not found")

    #Using BFMatching to find close matches using k-th nearest neighbour
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.knnMatch(des1,des2,k=2)

    peak_matches=[]
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            peak_matches.append(m)

    if len(peak_matches) < 4:
        raise ValueError("Not enough matches to compute homography")
    
    # Extract location of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in peak_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in peak_matches]).reshape(-1, 1, 2)

    # Find the homography matrix using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        raise ValueError("Homography computation failed")
    inliers=[]
    for i in range(len(peak_matches)):
        if mask[i]:
            inliers.append(peak_matches[i])

    return H, kp1, kp2, inliers

#Drawing Matches
def draw(img1,img2,kp1,kp2,matches):
    
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = img1
    out[:h2, w1:w1 + w2] = img2

    for m in matches[:30]:

        pt1 = tuple(np.int32(kp1[m.queryIdx].pt))
        pt2 = tuple(np.int32(kp2[m.trainIdx].pt) + np.array([w1, 0]))

        # Draw thick line
        cv2.line(out, pt1, pt2, (0, 255, 0), thickness=2)

        # Draw thicker circles
        cv2.circle(out, pt1, 3, (255, 0, 0), -1)
        cv2.circle(out, pt2, 3, (255, 0, 0), -1)

    plt.figure(figsize=(15,8))
    plt.imshow(out)
    plt.axis("off")
    plt.show()

def Warp(img1,img2,H):
    h,w,_ = img1.shape
    warped=cv2.warpPerspective(img2,H, (w,h))
    plt.figure(figsize=(12,8))
    plt.imshow(warped)
    plt.axis("off")
    plt.show()
    return warped

if __name__=="__main__":
    path1="image1.jpg"
    path2="image2.jpg"

    gray1,gray2,rgb1,rgb2=preprocess(path1,path2)
    H,kp1,kp2,peak_matches=Homograph(gray1,gray2,rgb1,rgb2)

    draw(rgb1,rgb2,kp1,kp2,peak_matches)
    align=Warp(rgb1,rgb2,H)

    
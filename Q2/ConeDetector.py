import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def preprocess(image_path):
    image1=cv2.imread(image_path)
    if(image1 is None):
        raise ValueError("Image not found")
    #Resize image
    image1=cv2.resize(image1,(640,480))
    original = image1.copy()
    blur1=cv2.GaussianBlur(image1,(5,5),1.2)
    hsv1=cv2.cvtColor(blur1,cv2.COLOR_BGR2HSV)

    #Create masks
    blue=cv2.inRange(hsv1,(100,150,0),(140,255,255))
    yellow=cv2.inRange(hsv1,(22,100,100),(30,255,255))
    orange=cv2.inRange(hsv1,(5,150,150),(20,255,255))

    #Filtered images
    bluef=cv2.bitwise_and(image1,image1,mask=blue)
    yellowf=cv2.bitwise_and(image1,image1,mask=yellow)
    orangef=cv2.bitwise_and(image1,image1,mask=orange)

    #Combined masks
    combmask=cv2.bitwise_or(blue,yellow)
    combmask=cv2.bitwise_or(combmask,orange)

    #Edge Detection
    edges=cv2.Canny(combmask,50,150)
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height=[]
    box=[]
    for i in contours:
        area=cv2.contourArea(i)
        if(area<30):
            continue
        x,y,w,h=cv2.boundingRect(i)

         # Aspect ratio filter (cones are taller than wide)
        aspect_ratio = h / float(w)
        if aspect_ratio < 0.8 or aspect_ratio > 6:
            continue
        cx=int(x+w/2)
        cy=int(y+h/2)

        height.append(h)
        box.append((x,y,w,h,cx,cy))

    if(len(height)>0):
        depth=np.array([1.0/h for h in height])
        geometricMean=np.prod(depth)**(1.0/len(depth))
        r=depth/geometricMean
    
    else:
        r=[]

    for i in range(len(box)):
        x,y,w,h,cx,cy = box[i]
        rela = r[i]

        cv2.circle(original, (cx,cy), 4, (0,0,255), -1)

        text = f"({cx},{cy})  d={rela:.2f}"
        cv2.putText(original, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,0,0), 2)

    fig, axs = plt.subplots(1,4, figsize=(18,5))

    axs[0].imshow(cv2.cvtColor(bluef, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Blue Filtered")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(yellowf, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Yellow Filtered")
    axs[1].axis("off")

    axs[2].imshow(cv2.cvtColor(orangef, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Orange Filtered")
    axs[2].axis("off")

    axs[3].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[3].set_title("Cones")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    image_name = input("Enter image name: ").strip()

    folder = "Input images"
    image_path = os.path.join(folder, image_name)

    if not os.path.exists(image_path):
        print("Image not found in Input images folder.")
        return

    preprocess(image_path)

main()




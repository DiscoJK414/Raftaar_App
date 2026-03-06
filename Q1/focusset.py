import cv2
import numpy as np

img = cv2.imread("Outputen.jpg")
if img is None:
    raise ValueError("Image not found!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Simulated focus levels
img1 = cv2.GaussianBlur(gray, (25,25), 0)
img2 = cv2.GaussianBlur(gray, (15,15), 0)
img3 = cv2.GaussianBlur(gray, (7,7), 0)
img4 = gray

# Stack properly
stack = np.array([img1, img2, img3, img4])

def compute_sharpness_stack(stack):
    sharpness_stack = []

    for img in stack:
        img = cv2.GaussianBlur(img, (5,5), 0)
        laplacian = cv2.Laplacian(img, cv2.CV_32F)
        sharpness = np.abs(laplacian)
        sharpness_stack.append(sharpness)

    return np.array(sharpness_stack)

def compute_depth_indices(sharpness_stack):
    return np.argmax(sharpness_stack, axis=0)



def normalize_depth_map(depth_indices):
    depth_map = depth_indices.astype(np.float32)

    depth_map = cv2.normalize(depth_map,None,0,255,cv2.NORM_MINMAX)

    return depth_map.astype(np.uint8)


# Run pipeline
sharpness_stack = compute_sharpness_stack(stack)
depth_indices = compute_depth_indices(sharpness_stack)
depth_map = normalize_depth_map(depth_indices)

cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("depth_map.png", depth_map)
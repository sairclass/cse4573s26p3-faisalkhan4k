'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####

    # Convert torch.Tensor (H x W x 3) to a uint8 numpy-compatible format
    # face_recognition expects a numpy array, but we convert via bytes
    # img is RGB torch.Tensor with values likely in [0, 255] or [0.0, 1.0]
    

    # img is shape (3, H, W), uint8, RGB — from torchvision.io.read_image()
    # face_recognition needs (H, W, 3) uint8 numpy array
    img_hwc = img.permute(1, 2, 0)         # (3, H, W) → (H, W, 3)
    img_np = img_hwc.numpy()               # convert to numpy for face_recognition

    # Detect face locations → list of (top, right, bottom, left)
    face_locations = face_recognition.face_locations(img_np, model="hog")

    # Convert to [x, y, width, height] as floats
    for (top, right, bottom, left) in face_locations:
        x      = float(left)
        y      = float(top)
        width  = float(right - left)
        height = float(bottom - top)
        detection_results.append([x, y, width, height])

    return detection_results

#--------------------------------------------------------------------------------

def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:

    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)]

    ##### YOUR IMPLEMENTATION STARTS HERE #####

    # Step 1: Get face encodings for each image
    img_names = []
    encodings = []

    for img_name, img in imgs.items():
        # img is (3, H, W) RGB uint8
        img_hwc = img.permute(1, 2, 0)   # → (H, W, 3)
        img_np  = img_hwc.numpy()

        # Get face location first
        face_locations = face_recognition.face_locations(img_np, model="hog")

        if len(face_locations) == 0:
            # If no face detected, use full image as the box
            h, w = img_np.shape[:2]
            face_locations = [(0, w, h, 0)]

        # Convert [x, y, w, h] → (top, right, bottom, left) for face_encodings
        boxes = [(top, right, bottom, left) for (top, right, bottom, left) in face_locations]

        # Get 128-d encoding
        enc = face_recognition.face_encodings(img_np, boxes)

        if len(enc) > 0:
            # Convert to torch tensor
            enc_tensor = torch.tensor(enc[0], dtype=torch.float32)
        else:
            # If encoding fails, use zeros
            enc_tensor = torch.zeros(128, dtype=torch.float32)

        img_names.append(img_name)
        encodings.append(enc_tensor)

    # Step 2: Stack all encodings into a matrix (N x 128)
    enc_matrix = torch.stack(encodings)   # shape: (N, 128)
    N = enc_matrix.shape[0]

    # Step 3: K-Means implementation (from scratch, no library)
    torch.manual_seed(42)

    # Initialize centroids by randomly picking K encodings
    perm = torch.randperm(N)
    centroids = enc_matrix[perm[:K]].clone()  # (K, 128)

    assignments = torch.zeros(N, dtype=torch.long)

    for iteration in range(100):  # max 100 iterations
        # Compute distances from each point to each centroid
        # enc_matrix: (N, 128), centroids: (K, 128)
        dists = torch.cdist(enc_matrix, centroids)   # (N, K)

        # Assign each point to nearest centroid
        new_assignments = torch.argmin(dists, dim=1)  # (N,)

        # Check for convergence
        if torch.equal(new_assignments, assignments):
            break

        assignments = new_assignments

        # Update centroids
        for k in range(K):
            mask = (assignments == k)
            if mask.sum() > 0:
                centroids[k] = enc_matrix[mask].mean(dim=0)

    # Step 4: Build output — group filenames by cluster
    for i, img_name in enumerate(img_names):
        cluster_idx = assignments[i].item()
        cluster_results[cluster_idx].append(img_name)

    return cluster_results





'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)

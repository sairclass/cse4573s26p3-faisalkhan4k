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

    
    img_hwc = img.permute(1, 2, 0)         
    img_np = img_hwc.numpy()              

    face_locations = face_recognition.face_locations(img_np, model="hog")

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

    #face encodings for each image
    img_names = []
    encodings = []

    for img_name, img in imgs.items():
        
        img_hwc = img.permute(1, 2, 0) 
        img_np  = img_hwc.numpy()

        # face location 
        face_locations = face_recognition.face_locations(img_np, model="hog")

        if len(face_locations) == 0:
            # If no face detected use full image instead
            h, w = img_np.shape[:2]
            face_locations = [(0, w, h, 0)]

        boxes = [(top, right, bottom, left) for (top, right, bottom, left) in face_locations]

        enc = face_recognition.face_encodings(img_np, boxes)

        if len(enc) > 0:
            enc_tensor = torch.tensor(enc[0], dtype=torch.float32)
        else:
            enc_tensor = torch.zeros(128, dtype=torch.float32)

        img_names.append(img_name)
        encodings.append(enc_tensor)

    enc_matrix = torch.stack(encodings)  
    N = enc_matrix.shape[0]

    torch.manual_seed(42)
    perm = torch.randperm(N)
    centroids = enc_matrix[perm[:K]].clone() 
    assignments = torch.zeros(N, dtype=torch.long)

    for iteration in range(100):  

        # Compute distances from each point to each centroid
        dists = torch.cdist(enc_matrix, centroids)  

        # Assign each point to nearest centroid
        new_assignments = torch.argmin(dists, dim=1)  


        if torch.equal(new_assignments, assignments):
            break

        assignments = new_assignments

        for k in range(K):
            mask = (assignments == k)
            if mask.sum() > 0:
                centroids[k] = enc_matrix[mask].mean(dim=0)

    for i, img_name in enumerate(img_names):
        cluster_idx = assignments[i].item()
        cluster_results[cluster_idx].append(img_name)

    return cluster_results





'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)

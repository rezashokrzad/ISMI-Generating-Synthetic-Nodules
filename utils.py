import os
from os import walk
import SimpleITK
import SimpleITK as sitk

import numpy as np
import pandas as pd
from pandas import DataFrame
from medpy.io import load, save
import random
import PIL
from PIL import Image
import scipy.ndimage
import json
import math
import matplotlib.pyplot as plt
from skimage import feature

#preprocessing utility functions
def resample_nodule(ct_data, xray_spacing):
    resize_factor = [1,1,1]/xray_spacing
    new_real_shape = ct_data.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / ct_data.shape
    new_spacing = [1,1,1] / real_resize_factor
    resampled_nodule = scipy.ndimage.interpolation.zoom(ct_data, real_resize_factor, mode='nearest')
    return resampled_nodule

def resample_nodule_2d(ct, spacing):
    resize_factor = [1,1] / np.array([spacing[0], spacing[1]])
    new_real_spacing = ct.shape * resize_factor
    new_shape = np.round(new_real_spacing)
    real_resize_factor = new_shape / ct.shape
    resampled_nodule = scipy.ndimage.interpolation.zoom(ct, real_resize_factor, mode='nearest')
    return resampled_nodule

def segment_nodule(ct, mask):
    background = np.full(ct.shape, np.min(ct))
    segmented_nodule = np.where(mask == 0, background, ct)
    return segmented_nodule

#Code from the baseline shared by Ecem
def project_ct(X_ct, p_lambda = 0.85, axis=1):
    '''
    Generate 2D digitally reconstructed radiographs from CT scan. (DRR, fake CXR, simulated CXR)
    X_ct: CT scan
    p-lambda:  β controls the boosting of X-ray absorption as the tissue density increases.
    We have chosen β=0.85 for our experiments after performing a visual comparison with real chest X-rays.
    author: Ecem Sogancioglu
    '''
    X_ct[X_ct > 400] = 400
    X_ct[X_ct < -500] = -500
    X_ct[X_ct < -1024] = -1024
    X_ct += 1024
    # 1424 524 698.748232
    X_ct = X_ct/1000.0
    X_ct *= p_lambda
    X_ct[X_ct > 1] = 1
    #1.0 0.4454 0.5866707652
    X_ct_2d = np.mean(np.exp(X_ct), axis=axis)
    return X_ct_2d

def crop_nodule(ct_2d, dimensions=2):
    outlines = np.where(ct_2d != np.min(ct_2d))
    x_min = np.min(outlines[0])
    x_max = np.max(outlines[0])
    y_min = np.min(outlines[1])
    y_max = np.max(outlines[1])
    if dimensions == 2:
        cropped_patch = ct_2d[x_min:x_max+1, y_min:y_max+1]
    elif dimensions == 3:
        z_min = np.min(outlines[2])
        z_max = np.max(outlines[2])
        cropped_patch = ct_2d[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    else:
        raise("Wrong number of dimensions")
    cropped_patch = np.pad(cropped_patch, 2)
    return cropped_patch
      
def preprocess_nodule_old(xray_header, ct, mask):
    #Resample the ct and the mask based on Xray spacing
    xray_spacing = xray_header.get_voxel_spacing()
    resampled_ct = resample_nodule(ct, np.array([xray_spacing[0],
                                                 xray_spacing[1],
                                                 1]
                                                )
                                   )
    resampled_mask = resample_nodule(mask, np.array([xray_spacing[0],
                                                     xray_spacing[1],
                                                     1]
                                                    )
                                     )

    #Segment the nodule using the mask
    segmented_ct = segment_nodule(resampled_ct, resampled_mask)
    
    #Project the segmented CT data from 3d to 2d
    ct_2d = project_ct(segmented_ct, axis=2)
    
    #crop the patch around the nodule
    cropped_ct_2d = crop_nodule(ct_2d)

    return cropped_ct_2d


def preprocess_nodule(ct, ct_mask, xray, spacing, verbose=False):
    ct = scale(ct, 0, 1) # make positive
    segmented_ct = segment_nodule(ct, ct_mask) # segment nodule from ct
    cropped_ct = crop_nodule(segmented_ct, dimensions = 3) #crop nodule from ct
    ct_2d = project_ct(cropped_ct) #project cropped ct
    resampled_ct = resample_nodule_2d(ct_2d, spacing) # resampled 2d nodule to xray spacing
    normalized_ct = scale(resampled_ct, xray.min(), xray.max()) # called 'normalize' in report
    if not verbose:
        return normalized_ct
    else:
        return segmented_ct, cropped_ct, ct_2d, resampled_ct, normalized_ct

def normalize_nodule_old(xray: np.ndarray, ct: np.ndarray, x, y):
    width = ct.shape[0]
    height = ct.shape[1]
    
    #normalize projected ct
    xray_patch = xray[ x - math.floor(width/2): x + math.ceil(width/2), y - math.floor(height/2): y + math.ceil(height/2) ]
    xray_max = np.max(xray_patch)
    xray_min = np.min(xray_patch)
   
    #changing the min value temporarily
    # temp = ct.copy()
    # temp[temp == temp.min()] = np.inf
    # second_min = temp.min()
    
    ct_min = np.min(ct[ct != ct.min()]) # this step has to be taken because the lowest min is black and shouldn't be taken into account
    xray_min = (ct_min / ct.max()) * xray_max
    
    #normalization
    # normalized_ct = ((ct - ct_min)*((xray_max - xray_min)/(ct.max() - ct_min)) + xray_min).astype('uint8')
    # normalized_ct = ((ct - ct_min) * (1/(xray_max - xray_min) * xray_max)).astype('uint8')
    normalized_ct = scale(ct, xray_min, xray_max)

    #for keepin the border as zero
    normalized_ct = np.multiply(normalized_ct, ct > np.min(ct))
    return normalized_ct
    
def normalize_nodule(xray: np.ndarray, ct: np.ndarray):
    return scale(ct, xray.min(), xray.max())

def normalize_x_ray(cxr, max_val = 255):
    return ((cxr - cxr.min()) * (1/(cxr.max() - cxr.min()) * max_val)).astype('uint8')

#Superimposition section
def mean_superimpose(location, nodule, xray):
    x = location[1]
    y = location[0]
    width = nodule.shape[0]
    height = nodule.shape[1]
    
    xray_patch = np.copy(xray[x - math.floor(width/2): x + math.ceil(width/2), y - math.floor(height/2): y + math.ceil(height/2)])
    valued_indices = (nodule > 1.1 * nodule.min())
    
    # blend cxr patch and nodule patch
    nodule[valued_indices] = np.mean(np.array([xray_patch[valued_indices],nodule[valued_indices]]), axis=0)
    nodule[~valued_indices] = xray_patch[~valued_indices]

    xray[x - math.floor(width/2): x + math.ceil(width/2), y - math.floor(height/2): y + math.ceil(height/2)] = nodule
    return xray, xray_patch, nodule

def mean_superimpose_new(nodule, xray, x, y):
    width = nodule.shape[0]
    height = nodule.shape[1]
    
    xray_patch = xray[x - math.floor(width/2): x + math.ceil(width/2), y - math.floor(height/2): y + math.ceil(height/2)]
    #create a clean copy for testing purposes
    clean_patch = np.copy(xray_patch)
    
    #changing the min value temporarily
    temp = nodule.copy()
    temp[temp == temp.min()] = temp.mean()
    second_min = temp.min()
    C = 0.5
    
    #Diameter of nodule
    D = np.sqrt(width**2 + height**2)
    
    for i in range(width):
        for j in range(height):
            #ecludian distance of pixels at i,j location to the center of nodule
            r = np.sqrt((width//2-i)**2 + (height//2-j)**2)
            c = C*( 4*r**4/D**4 - 4.2*r**2/D**2 + 1)
            #add nodule to the x_ray
            xray[x - width//2+i, y-height//2+j] += c*nodule[i,j]
    return xray, clean_patch, xray_patch

def preprocess_lung_mask(lung_segmentation, border_size = 10, visualize = False):
    edges = feature.canny(lung_segmentation, sigma=3).astype(float)
    lungseg = lung_segmentation.astype(float)
    edges_idx = np.where(edges != np.amin(edges))
    thick = border_size

    for x, y in np.dstack(edges_idx).squeeze():
        edges[x-thick:x+thick, y-thick:y+thick] = edges.max()

    lungseg = lungseg.astype(float)

    lungseg *= 1.0 / lungseg.max()   

    if visualize: 
        new_lungseg = lungseg - 0.5 * edges
    else:
        new_lungseg = lungseg - edges
    new_lungseg[new_lungseg <= 0] = 0
    return new_lungseg
    
def get_random_location(lung_mask):
    locations = np.where(lung_mask == lung_mask.max())
    r_idx = np.random.randint(0, len(locations[0]))
    return (locations[0][r_idx], locations[1][r_idx])

def spherical_superimpose(location: tuple, nodule: np.array, xray: np.array, C:float = 0.5) -> np.array:
    x = location[0]
    y = location[1]       

    #Diameter of nodule
    D = np.sqrt(nodule.shape[0]**2 + nodule.shape[1]**2)
    
    #spherical formula
    for i in range(nodule.shape[0]):
        for j in range(nodule.shape[1]):
            #ecludian distance of pixels at i,j location to the center of nodule
            r = np.sqrt((nodule.shape[0]//2-i)**2 + (nodule.shape[1]//2-j)**2)
            c = C*( 4*r**4/D**4 - 4.2*r**2/D**2 + 1)
            #add nodule to the x_ray
            xray[x - nodule.shape[0]//2+i, y-nodule.shape[1]//2+j] += c*nodule[i,j]
    return xray

def augment_nodule(nodule):
    idx = np.random.randint(0, 6)
    if idx == 0:
        #rotation 90 degree
        augmented_nodule = nodule.T
        augment_type = "rotation"
    elif idx == 1:
        #vertical flip 
        augmented_nodule = np.flip(nodule, axis=0)
        augment_type = "flip vertically"
    elif idx == 2:
        #horizontal flip 
        augmented_nodule = np.flip(nodule, axis=1)
        augment_type = "flip horizontally"    
    elif idx == 3:
        #magnifying
        augmented_nodule = np.kron(nodule, np.ones((2,2)))
        augment_type = "zooming in"
    else:
        return nodule, "original(no_aug)"
    
    return augmented_nodule, augment_type

def scale(x, min, max):
    input = x.copy().astype(float)
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input
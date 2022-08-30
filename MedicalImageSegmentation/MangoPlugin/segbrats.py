import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
import nibabel as nib
from tensorflow import keras
import json

import argparse

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--folder")
parser.add_argument("--model")
parser.add_argument("--output")

args = parser.parse_args()

f_value = args.folder
m_value = args.model
out_value = args.output


def load_img(img_files, seg_file, with_seg):
    
    if with_seg:
        N = len(img_files)
        y = nib.load(seg_file).get_fdata(dtype='float32', caching='unchanged')
        y = y[40:200,34:226,8:136]
        y[y==4]=3
        
        X_norm = np.empty((240, 240, 155, 4))
        for channel in range(N):
            X = nib.load(img_files[channel]).get_fdata(dtype='float32', caching='unchanged')
            XX = nib.load(img_files[channel])
            img_affine=XX.affine
            brain = X[X!=0] 
            brain_norm = np.zeros_like(X)
            norm = (brain - np.mean(brain))/np.std(brain)
            brain_norm[X!=0] = norm
            X_norm[:,:,:,channel] = brain_norm        
            
        X_norm = X_norm[40:200,34:226,8:136,:]    
        del(X, brain, brain_norm)
        
    else: 
        N = len(img_files)
       
        X_norm = np.empty((240, 240, 155, 4))
        for channel in range(N):
            X = nib.load(img_files[channel]).get_fdata(dtype='float32', caching='unchanged')
            XX = nib.load(img_files[channel])
            img_affine=XX.affine
            brain = X[X!=0] 
            brain_norm = np.zeros_like(X)
            norm = (brain - np.mean(brain))/np.std(brain)
            brain_norm[X!=0] = norm
            X_norm[:,:,:,channel] = brain_norm        
            
        X_norm = X_norm[40:200,34:226,8:136,:]
        
        y = ''
        
        del(X, brain, brain_norm)
    
    return X_norm.astype('float32'),y,img_affine
    

from scipy import ndimage

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 1e-5

    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def binary_dice3d(s,g):
    #dice score of two 3D volumes
    smooth = 1e-5
    num=np.sum(np.multiply(s, g))
    denom=s.sum() + g.sum() + smooth
    
    return  2.0*num/denom

def sensitivity (seg,ground): 
    #computs false negative rate
    smooth = 1e-5
    num=np.sum(np.multiply(ground, seg))
    denom=np.sum(ground)+smooth
    
    return  num/denom

def specificity (seg,ground): 
    #computes false positive rate
    smooth = 1e-5
    num=np.sum(np.multiply(ground==0, seg ==0))
    denom=np.sum(ground==0)+smooth
    
    return  num/denom

def border_map(binary_img,neigh):
    """
    Creates the border for a 3D image
    """
    binary_map = np.asarray(binary_img, dtype=np.uint8)
    neigh = neigh
    west = ndimage.shift(binary_map, [-1, 0,0], order=0)
    east = ndimage.shift(binary_map, [1, 0,0], order=0)
    north = ndimage.shift(binary_map, [0, 1,0], order=0)
    south = ndimage.shift(binary_map, [0, -1,0], order=0)
    top = ndimage.shift(binary_map, [0, 0, 1], order=0)
    bottom = ndimage.shift(binary_map, [0, 0, -1], order=0)
    cumulative = west + east + north + south + top + bottom
    border = ((cumulative < 6) * binary_map) == 1
    return border

def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    neigh=8
    border_ref = border_map(ref,neigh)
    border_seg = border_map(seg,neigh)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg#, border_ref, border_seg

def Hausdorff_distance(ref,seg):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a segmentation and a reference image
    :return: hausdorff distance and average symmetric distance
    """
    ref_border_dist, seg_border_dist = border_distance(ref,seg)
    hausdorff_distance = np.max([np.max(ref_border_dist), np.max(seg_border_dist)])
    return hausdorff_distance

def DSC_whole(pred, orig_label):
    #computes dice for the whole tumor
    return dice_coef(pred>0,orig_label>0)

def DSC_en(pred, orig_label):
    #computes dice for enhancing region
    return dice_coef(pred==3,orig_label==3)

def DSC_core(pred, orig_label):
    #computes dice for core region
    seg_=np.copy(pred)
    ground_=np.copy(orig_label)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return dice_coef(seg_>0,ground_>0)

def sensitivity_whole (seg,ground):
    return sensitivity(seg>0,ground>0)

def sensitivity_en (seg,ground):
    return sensitivity(seg==3,ground==3)

def sensitivity_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return sensitivity(seg_>0,ground_>0)

def specificity_whole (seg,ground):
    return specificity(seg>0,ground>0)

def specificity_en (seg,ground):
    return specificity(seg==3,ground==3)

def specificity_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return specificity(seg_>0,ground_>0)
    
def hausdorff_whole (seg,ground):
    return Hausdorff_distance(seg==0,ground==0)

def hausdorff_en (seg,ground):
    return Hausdorff_distance(seg!=3,ground!=3)

def hausdorff_core (seg,ground):
    seg_=np.copy(seg)
    ground_=np.copy(ground)
    seg_[seg_==2]=0
    ground_[ground_==2]=0
    return Hausdorff_distance(seg_==0,ground_==0)

stats={'dice':[],'spec':[],'sen':[],'hau95':[]}
folder = f_value

t1 = ""
t2 = ""
t1ce = ""
flair = ""
seg = ""

for file in os.listdir(folder):
    if(file.endswith(".nii") or file.endswith(".nii.gz")):
        if(file.endswith("_t1.nii") or file.endswith("_t1.nii.gz")):
            t1 = os.path.join(folder,file)
        elif(file.endswith("_t1ce.nii") or file.endswith("_t1ce.nii.gz")):
            t1ce = os.path.join(folder,file)
        elif(file.endswith("_t2.nii") or file.endswith("_t2.nii.gz")):
            t2 = os.path.join(folder,file)
        elif(file.endswith("_flair.nii") or file.endswith("_flair.nii.gz")):
            flair = os.path.join(folder,file)
        elif(file.endswith("_seg.nii") or file.endswith("_seg.nii.gz")):
            seg = os.path.join(folder,file)
            
        
get_seg = False
if seg != "":
    get_seg = True

G = keras.models.load_model(m_value)

Xb = np.empty((4, *(160,192,128), 4))

Xb,yb,AFFb = load_img([t1,t2,t1ce,flair],seg,get_seg)

tempy=np.empty((160,192,128,4))
tempy=np.zeros_like(tempy)

part1=Xb[0:128,0:128,:,:]
part2=Xb[0:128,64:192,:,:]
part3=Xb[32:160,0:128,:,:]
part4=Xb[32:160,64:192,:,:]

Batch=np.empty((4,128,128,128,4))

Batch[0,:]=part1
Batch[1,:]=part2
Batch[2,:]=part3
Batch[3,:]=part4

del Xb

segBatch = G.predict(Batch)

segG1 = segBatch[0,:]
segG2 = segBatch[1,:]
segG3 = segBatch[2,:]
segG4 = segBatch[3,:]

tempy[0:128,0:128,:,:]=segG1
tempy[0:128,64:192,:,:]=segG2
tempy[128:160,0:128,:,:]=segG3[96:128,0:128,:,:]
tempy[128:160,128:192,:,:]=segG4[96:128,64:128,:,:]

del segBatch

segG = np.empty((240,240,155,4))
segG = np.zeros_like(segG)
segTr = np.empty((240,240,155,4))
segTr = np.zeros_like(segTr)

segG[40:200,34:226,8:136,:] = tempy

del tempy
if (yb!=''):

	yb = to_categorical(yb, 4)
	segTr[40:200,34:226,8:136,:] = yb

	segG_all = np.argmax(segG, axis=-1)

	case_path = folder

	segTr_all = np.argmax(segTr, axis=-1)

	
	dscET=DSC_en(segG_all,segTr_all)
	dscWT=DSC_whole(segG_all,segTr_all)
	dscTC=DSC_core(segG_all,segTr_all)

	specET= specificity_en(segG_all,segTr_all)
	specWT= specificity_whole(segG_all,segTr_all)
	specTC= specificity_core(segG_all,segTr_all)

	senET= sensitivity_en(segG_all,segTr_all)
	senWT= sensitivity_whole(segG_all,segTr_all)
	senTC= sensitivity_core(segG_all,segTr_all)

	hauET= hausdorff_en(segG_all,segTr_all)
	hauWT= hausdorff_whole(segG_all,segTr_all)
	hauTC= hausdorff_core(segG_all,segTr_all)

	stats['dice'].append({'caseID':case_path,'ET':dscET,'WT':dscWT,'TC':dscTC})
	stats['spec'].append({'caseID':case_path,'ET':specET,'WT':specWT,'TC':specTC})
	stats['sen'].append({'caseID':case_path,'ET':senET,'WT':senWT,'TC':senTC})
	stats['hau95'].append({'caseID':case_path,'ET':hauET,'WT':hauWT,'TC':hauTC})

	segG_all[segG_all==3]=4
else:
	segG_all = np.argmax(segG, axis=-1)
	segG_all[segG_all==3]=4
    
final_img = nib.Nifti1Image(segG_all, AFFb, dtype = 'int32')

nib.save(final_img, os.path.join(out_value, 'segbrats.nii.gz'))
if(yb!=''):
	with open(out_value + '/stats.json', 'w') as f:
    		json.dump(stats, f, indent=2)
    		print("stats saved.")
    
print(segG_all)

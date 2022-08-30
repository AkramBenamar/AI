#coding: UTF-8
import sys
# import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
import nibabel as nib

from tensorflow import keras

import json

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import tarfile
import nibabel as nib
import glob
import time
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sys import stdout

import json
import cv2 as cv
from tensorflow.keras.models import Model

import argparse

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--folder")
parser.add_argument("--model")
parser.add_argument("--output")

args = parser.parse_args()

f_value = args.folder
m_value = args.model
out_value = args.output

#LOAD
def load_img(img_files):
    ''' Load one image and its target form file
    '''
    N = len(img_files)
    # target
    y = nib.load(img_files[N-1]).get_fdata(dtype='float32', caching='unchanged')
    y[y==2]=1
    y[y==3]=1
    
    # y = y[0:630,130:590,0:35] #(630,460,35)
    y_les = nib.load(img_files[1]).get_fdata(dtype='float32', caching='unchanged')
    
    
    X_norm = np.empty(y.shape)

    X = nib.load(img_files[0]).get_fdata(dtype='float32', caching='unchanged')
    # X = X[0:630,130:590,0:35] 
    X = X*y
    lung = X[X!=0] 
    lung_norm = np.zeros_like(X) # background at -100

    
    xmax, xmin = np.max(lung), np.min(lung)
    lung_norm[X!=0]  = (lung - xmin)/(xmax - xmin)
    
    X_norm[:,:,:] = lung_norm        
            
    del(X, lung, lung_norm)
    
    X_norm = np.rot90(np.array(X_norm))
    y = np.rot90(np.array(y))
    y_les = np.rot90(np.array(y_les))
    
    
    X_new_shape = np.empty(( *(X_norm.shape), 1))
    image_data = X_norm[:,:,:]
    neww1 = np.zeros(image_data.shape)

    for i in range(len(image_data[0,0,:])):
        slicee = image_data[:,:,i]
        slicee = np.uint8(slicee*255) 
        clahe1 = cv.createCLAHE(clipLimit=3.0)
        clahe_img1 = clahe1.apply(slicee)
        neww1[:,:,i]= clahe_img1
   
    X_new_shape[:,:,:,0] = neww1
    del image_data,neww1,slicee,clahe_img1

    return X_new_shape.astype('float32'), to_categorical(y_les).astype('float32')


def predict_cov2(inp):
    X,y=load_img(inp)
    
    rows,columns,slices,_=X.shape
    
    
        
    PatchesShape=(columns,columns,32)
#re shape incoherent data
    if rows < 512:
            
            
            Xnv=np.empty((columns,columns,slices))
            Xnv=X[:,:,:,0]
            ynv=np.empty((columns,columns,slices))

            ynv=np.argmax(y,axis=-1)
            if (rows%2)!=0:
                rows = rows-1
            
            new_frame_X = np.empty((PatchesShape[0],columns,slices))
            new_frame_y = np.empty((PatchesShape[0],columns,slices))
            
            for slc in range(slices):
                slicee_X = Xnv[:,:,slc]
                slicee_y = ynv[:,:,slc]
                
                for c in range(columns):
                    column_X = slicee_X[:,c]
                    column_y = slicee_y[:,c]
                    pad_arr_X = np.pad(column_X[:-1], (((PatchesShape[0]-rows)//2),), 'constant', constant_values=(0, 0))
                    pad_arr_y = np.pad(column_y[:-1], (((PatchesShape[0]-rows)//2),), 'constant', constant_values=(0, 0))

                    new_frame_X[:,c,slc] = pad_arr_X
                    new_frame_y[:,c,slc] = pad_arr_y
            
            Xnv = new_frame_X
            ynv = new_frame_y
            y=np.empty((ynv.shape[0],ynv.shape[1],ynv.shape[2],2))
            y=to_categorical(ynv).astype('float32')
            X=np.empty((Xnv.shape[0],Xnv.shape[1],Xnv.shape[2],1))
            X[:,:,:,0]=Xnv
            del(Xnv,ynv,new_frame_X,new_frame_y,slicee_X,slicee_y,pad_arr_X,pad_arr_y)

    
    
    
    
    
    
    
    
    
    Xnew=np.empty((32,512,512,1))
    
    Xnew=np.zeros_like(Xnew)
    ynew=np.empty((y.shape))
    ynew=np.zeros_like(ynew)
    
    
    x1=int((X.shape[0])//2 -512/2)
#     print(x1)
    zmi=int(X.shape[2]//2)
#     print(x1,X.shape,y.shape)
    nslice=0
    nslicenew=0
    for kh in range(32):
        
#         print(Xnew.shape,zmi)
#         print('X',X.shape,x1)
        Xnew[kh,:,:,:]=X[x1:x1+512,x1:x1+512,zmi-16+kh,:]
        
    
    segGtemp=G.predict(Xnew)
    del(Xnew,X)
    for s in range(32):
                slicetemp=segGtemp[s,:]
                # print(ynew.shape)
                ynew[x1:x1+512,x1:x1+512,zmi-16+s,:]=slicetemp
                
    
    
    y=np.argmax(y,axis=-1)
    ynew=np.argmax(ynew,axis=-1)
    return y,ynew


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
    

stats={'dice':[],'spec':[],'sen':[],'hau95':[]}
folder = f_value
# filelist = []

ct = folder
seg_les = folder
seg = folder
if ("_org_" in folder or "_org_covid-19-pneumonia-" in folder) :
    ct=folder
    seg_les=seg_les.replace("ct_scans","infection_mask")
    seg=seg.replace("ct_scans","lung_and_infection_mask")
    if("_org_covid-19-pneumonia-" in folder):
        seg_les=seg_les.replace("org_covid-19-pneumonia-","")
        seg_les=seg_les.replace("-dcm","")
        seg=seg.replace("org_covid-19-pneumonia-","")
        seg=seg.replace("-dcm","")

    elif("_org_" in folder):
        seg_les=seg_les.replace("org_","")
        
        seg=seg.replace("org_","")

    
else :
    print('please enter a ct scan from specified dataset')


G = keras.models.load_model(m_value)


X,y=load_img([ct,seg_les,seg])
XX = nib.load(ct)
img_affine=XX.affine
del(XX)
print('image loaded')
yout,ynew=predict_cov2([ct,seg_les,seg])



dice=dice_coef(ynew,yout)
sens=sensitivity(ynew,yout)
stats['dice'].append(dice)
stats['sen'].append(sens)

ynew=np.rot90(np.array(ynew))
ynew=np.rot90(np.array(ynew))
ynew=np.rot90(np.array(ynew))

final_img = nib.Nifti1Image(ynew,img_affine,dtype='int32')
del(ynew,yout)


nib.save(final_img, os.path.join(out_value, 'segct.nii.gz'))
with open(out_value + '/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
    print("stats saved.")
 
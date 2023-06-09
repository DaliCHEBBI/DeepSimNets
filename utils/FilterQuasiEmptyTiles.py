import numpy as np
import torch
import glob
import os
import timeit 
import tifffile as tff 
from PIL import Image
import re


def read_file_from_list(f_):
    with open(f_,
            'r') as ff_:
        allL=ff_.read().splitlines()
        allL=[el for el in allL]
    return allL

def LoadImages(folder):
    LeftTileNames=[]
    RightTileNames=[]
    DispTileNames=[]
    files = [] 
    if not isinstance(folder,list):
        folder=[folder]
    for (dirpath, dirnames, filenames) in os.walk(folder.pop()):
        folder.extend(dirnames)
        files.extend(map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames)))
    # classify file names into 3 classes Left Rigth and disparities
    LeftTileNames=[el for el in files if re.search("colored_0",el)]
    RightTileNames=[el for el in files if re.search("colored_1",el)]
    DispTileNames=[el for el in files if re.search("disp_occ",el)]
    return LeftTileNames,RightTileNames,DispTileNames

def LoadANDFilterImages(folder):
    imnl=open('{}/l_1_left_train.txt'.format(folder),'a')
    imnr=open('{}/l_1_right_train.txt'.format(folder),'a')
    imnd=open('{}/l_1_disp_train.txt'.format(folder),'a')
    imnm=open('{}/l_1_masq_train.txt'.format(folder),'a')
    SUFFIX_IMAGE="DMTrain"
    SUFFIX_DISP='DensifyPx'
    SUFFIX_MASQ="Nocc_clean"
    IMAGES=os.listdir(folder)
    IMLR=[el_ for el_ in IMAGES if el_.startswith(SUFFIX_IMAGE)]
    IML=[el_ for el_ in IMLR if el_.endswith('Im1.tif')]
    IMR=[el_ for el_ in IMLR if el_.endswith('Im2.tif')]
    IMD=[el_ for el_ in IMAGES if el_.startswith(SUFFIX_DISP)]
    IMM=[el_ for el_ in IMAGES if el_.startswith(SUFFIX_MASQ)]
    assert(len(IML)==len(IMR))
    assert(len(IMD)==len(IMM))
    for im_ in IML:
        # GET THE LEFT IMAGE 
        iml_fname=os.path.join(folder,im_)
        imr_fname=os.path.join(folder,im_[:-5]+"2.tif")
        print(imr_fname)
        # Check if file 2 exists 
        if os.path.exists(imr_fname) and os.path.exists(iml_fname):
            # READ images 
            LL=tff.imread(iml_fname)
            RR=tff.imread(imr_fname)
            thrsh=int(0.97*1024*1024)
            
            print("IMAGE CONTENT =>", np.count_nonzero(LL),np.count_nonzero(RR), thrsh)
            
            if (np.count_nonzero(LL)>thrsh and np.count_nonzero(RR)>thrsh) :
                # save in files 
                imnl.write(iml_fname+'\n')
                imnr.write(imr_fname+'\n')
                imnd.write(os.path.join(folder,"DensifyPx_"+im_)+'\n')
                imnm.write(os.path.join(folder,"Nocc_clean_DensifyPx_"+im_)+'\n')
                
                
def MoveLR():
    FolderimagesLR='/home/mohamedali/Documents/Code/Datasets/DublinCity-stereo_echo_new/testing_all'
    FolderMasqs='/media/mohamedali/GEOMAKER/Clean_TEST_IMAGES'
    MASQsAndDisps=os.listdir(FolderMasqs)
    PATT="Nocc_clean_DensifyPx_"
    MASQS=[el_ for el_ in MASQsAndDisps if el_.startswith(PATT)]
    print(len(MASQS))
    for amsq in MASQS:
        # image left and right 
        iml=os.path.join(FolderimagesLR,amsq[len(PATT):])
        imr=os.path.join(iml.replace('Im1','Im2'))
        #print(iml)
        #print(imr)
        if os.path.exists(iml) and os.path.exists(imr):
            print("Moving left and right images ")
            cmd='mv {} {}'.format(iml,FolderMasqs)
            os.system(cmd)
            cmd='mv {} {}'.format(imr,FolderMasqs)
            os.system(cmd)      
        else:
            print("Image L or R  not found !")
       
def MoveFolders(NamedFileList, CLEAN_FOLDER):
    if not os.path.exists(CLEAN_FOLDER):
        os.system('mkdir {}'.format(CLEAN_FOLDER))
    for el in NamedFileList:
        cmd='mv {} {}'.format(el,CLEAN_FOLDER)
        os.system(cmd)

        

def MovePX():
    FolderimagesLR='/home/mohamedali/Documents/Code/Datasets/DublinCity-stereo_echo_new/testing_all'
    FolderMasqs='/media/mohamedali/GEOMAKER/Clean_TEST_IMAGES'
    MASQsAndDisps=os.listdir(FolderMasqs)
    PATT="Nocc_refine_DensifyPx_"
    MASQS=[el_ for el_ in MASQsAndDisps if el_.startswith(PATT)]
    print(len(MASQS))
    for amsq in MASQS:
        # image left and right 
        imd=os.path.join(FolderimagesLR,amsq[12:])
        if os.path.exists(imd):
            print("Moving left and right images ")
            cmd='mv {} {}'.format(imd,FolderMasqs)
            os.system(cmd) 
        else:
            print("Image L or R  not found !")

def CheckIfInTrainData(folderTest, FoldersTRAIN):
    # list all test masqs
    MASQS=os.listdir(folderTest)
    MASQS=[el for el in MASQS if el.startswith('Nocc_refine')]
    allmasqs=[]
    RetainedForTest=[]
    for folder in FoldersTRAIN:
        # list all masqs
        MSQs=os.listdir(folder)
        MSQs=[el for el in MSQs if el.startswith('Nocc_clean')]
        allmasqs.extend(MSQs)
    for eachMasq in MASQS:
        if not eachMasq.replace('Nocc_refine','Nocc_clean') in  allmasqs:
            RetainedForTest.append(eachMasq)
    # Save the retained masqs to a file 
    """with open("./RETAINED_FILES.txt", "w") as F_:
        F_.writelines([line +"\n" for line in RetainedForTest])"""
    os.system('mkdir Clean_TEST_IMAGES')
    for mask in RetainedForTest:
        print(mask)
        cmd= "mv {} ./Clean_TEST_IMAGES".format(os.path.join(folderTest,mask))
        os.system(cmd)
        
     
            
"""
if __name__=="__main__":
    path_train= 'DublinCity-stereo_echo_new/training'
    path_val= 'DublinCity-stereo_echo_new/testing'
    lefts, rights, disp=LoadImages(path_val)
    imnl=open('left_test.txt','a')
    imnr=open('right_test.txt','a')
    for l,r in zip(lefts,rights):
        iml=np.array(Image.open(l))
        imr=np.array(Image.open(r))
        #
        thrsh=int(0.9*1024*1024)
        if (np.count_nonzero(iml)>thrsh and np.count_nonzero(imr)>thrsh) :
            # save in files 
            imnl.write(l+'\n')
            imnr.write(r+'\n')
    imnl.close()
    imnr.close()
""" 
if __name__=="__main__":
    """Folder="/media/mohamedali/GEOMAKER/TRAINING_3_2"
    LoadANDFilterImages(Folder)
    F1="/media/mohamedali/GEOMAKER/TRAINING_1"
    F2="/media/mohamedali/GEOMAKER/TRAINING_2"
    F31="/media/mohamedali/GEOMAKER/TRAINING_3_1"
    F32="/media/mohamedali/GEOMAKER/TRAINING_3_2"
    FT='/media/mohamedali/GEOMAKER/DUBLIN_DENSE/MASQ_TEST'
    CheckIfInTrainData(FT,[F1,F2,F31,F32])"""
    #MoveLR()
    # SET 1 TRAINING 
    """str1L='/media/mohamedali/GEOMAKER/TRAINING_1/1_left_train.txt'
    tr1R='/media/mohamedali/GEOMAKER/TRAINING_1/1_right_train.txt'
    tr1D='/media/mohamedali/GEOMAKER/TRAINING_1/1_disp_train.txt'
    tr1M='/media/mohamedali/GEOMAKER/TRAINING_1/1_masq_train.txt'
    
    tr1L=read_file_from_list(tr1L)
    tr1R=read_file_from_list(tr1R)
    tr1D=read_file_from_list(tr1D)
    tr1M=read_file_from_list(tr1M)    
    # 
    MoveFolders(tr1L,"/media/mohamedali/GEOMAKER/training_1")
    MoveFolders(tr1R,"/media/mohamedali/GEOMAKER/training_1")
    MoveFolders(tr1D,"/media/mohamedali/GEOMAKER/training_1")
    MoveFolders(tr1M,"/media/mohamedali/GEOMAKER/training_1")"""
    
    """tr2L='/media/mohamedali/GEOMAKER/TRAINING_2/2_left_train.txt'
    tr2R='/media/mohamedali/GEOMAKER/TRAINING_2/2_right_train.txt'
    tr2D='/media/mohamedali/GEOMAKER/TRAINING_2/2_disp_train.txt'
    tr2M='/media/mohamedali/GEOMAKER/TRAINING_2/2_masq_train.txt'
    tr2L=read_file_from_list(tr2L)
    tr2R=read_file_from_list(tr2R)
    tr2D=read_file_from_list(tr2D)
    tr2M=read_file_from_list(tr2M)    
    # 
    MoveFolders(tr2L,"/media/mohamedali/GEOMAKER/training_2")
    MoveFolders(tr2R,"/media/mohamedali/GEOMAKER/training_2")
    MoveFolders(tr2D,"/media/mohamedali/GEOMAKER/training_2")
    MoveFolders(tr2M,"/media/mohamedali/GEOMAKER/training_2")"""
    
    """tr2L='/media/mohamedali/GEOMAKER/TRAINING_3_1/3_left_train.txt'
    tr2R='/media/mohamedali/GEOMAKER/TRAINING_3_1/3_right_train.txt'
    tr2D='/media/mohamedali/GEOMAKER/TRAINING_3_1/3_disp_train.txt'
    tr2M='/media/mohamedali/GEOMAKER/TRAINING_3_1/3_masq_train.txt'
    tr2L=read_file_from_list(tr2L)
    tr2R=read_file_from_list(tr2R)
    tr2D=read_file_from_list(tr2D)
    tr2M=read_file_from_list(tr2M)    
    # 
    MoveFolders(tr2L,"/media/mohamedali/GEOMAKER/training_3_1")
    MoveFolders(tr2R,"/media/mohamedali/GEOMAKER/training_3_1")
    MoveFolders(tr2D,"/media/mohamedali/GEOMAKER/training_3_1")
    MoveFolders(tr2M,"/media/mohamedali/GEOMAKER/training_3_1")"""
    
    Folder="/mnt/c/Image_MVE_DATASET/DUBLIN_DENSE/training_1"
    LoadANDFilterImages(Folder)
    
    
    
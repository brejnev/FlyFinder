#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:57:30 2020

@author: brejnev.muhire
"""

## This are the main function used in the FlyFinder pipeline
'''
FlyFinder is a computer program used for measuring pixel distances between drosophila using either a single or four-quadrant triangle. FlyFinder streamlines the segmentation of drosophila, post-processing, analysis, and reporting of nearest neighbor distances.
It trains a random forest classifier using labeled images and then this classifier used to prosses newly produced images. It generates PDF reports and data tables are during each processing step, along with the generation of a final distance data table and final figures pdf report. First, second, and third nearest neighbor distances are calculated, along with the average neighbor distance of each input image. FlyFinder is mainly written in python and uses R for distance calculation and final figures.

'''



###################
## Module imports #
###################

import os, sys, time, datetime, img2pdf, pickle, cv2, argparse, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage as nd
from skimage.filters import roberts, sobel,  scharr, scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h #sobel_h, sobel_v,
from skimage.draw import polygon, line
from skimage import data, img_as_float, io
from skimage.morphology import square, erosion, remove_small_objects
from skimage.segmentation import (morphological_chan_vese,morphological_geodesic_active_contour,inverse_gaussian_gradient,checkerboard_level_set,watershed)
from skimage.feature import peak_local_max 

#from skimage import  segmentation, feature, future

from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from functools import partial

#########################
## Function definitions #
#########################

pen=5; 


## Gets the center of mass coord, aras size and boxlocation of the flies
def get_prop(labImg):
   regions = regionprops(labImg)
   coors=[r.centroid for r in regions]
   areas=[r.area for r in regions]
   bbox=[r.bbox for r in regions]
   bbox=[r.bbox for r in regions]
   fimgs=[r.filled_image for r in regions]
   return(coors, areas,bbox,fimgs)
   
## Finds fly area in pxl  from a labeled objects in the image 
def get_areas(labImg):
   regions = regionprops(labImg)
   areas=[r.area for r in regions]
   return(areas)

## Applies a MASK to the image before segmentation
def maskFig(inpImg_,bottomMar=60,topGap=0, col=255):
    inpImg=inpImg_.copy()
    
    #topGap=100

    # fill polygon1
    poly = np.array(( (0, 0), (0, int(inpImg.shape[1]/2)- int(topGap/2)), (inpImg.shape[0], 0)))
    rr, cc = polygon(poly[:,0], poly[:,1], inpImg.shape)
    inpImg[rr,cc] = col
    
    
    # fill polygon2
    poly = np.array(( (0,int(inpImg.shape[1]/2)+int(topGap/2)), (0,inpImg.shape[1]), (inpImg.shape[0],inpImg.shape[1])))
    rr, cc = polygon(poly[:,0], poly[:,1], inpImg.shape)
    inpImg[rr,cc] = col


    # fill polygon3
    poly = np.array((( inpImg.shape[0]-bottomMar,0), (inpImg.shape[0],0),(inpImg.shape[0],inpImg.shape[1]), (inpImg.shape[0]-bottomMar,inpImg.shape[1]) ))
    rr, cc = polygon(poly[:,0], poly[:,1], inpImg.shape)
    inpImg[rr,cc] = col
    
    return(inpImg)

## Draws points given pixel xy coordinates
def draw_points(image,cords,color,pwd=1):
    out=image.copy()
    for cor in cords:
        r= int(round(cor[0],0)); c = int(round(cor[1],0))
        P=out[r:r + pwd, c:c + pwd]
        if (len(P.shape)==3):
            P[:,:,: ]=color
        if (len(P.shape)==2):
            P[:,:]=color
    return(out)
    
    
## Demarcates where the Mask border are
def maskLines3C(inpImg_, bottomMar=60,topGap=0, col=(255,0,255),lThick=1):
    inpImg=inpImg_.copy()
    
    inpImg = cv2.line(img=inpImg, pt1=( int((inpImg.shape[0]/2)-int(topGap/2)),0), pt2=(0,inpImg.shape[0]-1), color=col, thickness=lThick)
    inpImg = cv2.line(img=inpImg, pt1=( int((inpImg.shape[0]/2)+int(topGap/2)),0) ,pt2=(inpImg.shape[1]-1, inpImg.shape[0]-1), color=col, thickness=lThick)
    inpImg = cv2.line(img=inpImg, pt1=( 0,inpImg.shape[0]-bottomMar), pt2=(inpImg.shape[1]-1,inpImg.shape[0]-bottomMar), color=col, thickness=lThick)

    return(inpImg)




## Converts a grayscale image to binary
def binFig(inpImg,thr=250,aboveVal=255,belowVal=0):
    binary=inpImg.copy()  #showFig(img_labeled)
    binary[inpImg>thr]=aboveVal
    binary[inpImg<=thr]=belowVal
    return(binary)


## Takes a directory and display images (1Panel) showing the bottomMarging and topGap
def viewImageMargins1Panel(imagPath,bottomMargin=60, topGap=100, cropLeft=0, cropRight=0, cropTop=0,cropBottom=0,marginCol=(255,255,255),figSize=5,pen_=1): 
    if os.path.isdir(imagPath)==False:
        print("The directory '"+imagPath+"' does not exist!")
        return()
    
    ## Checking the validity of croping, bottom margin and top gap parameters
    if min(cropLeft, cropRight,cropTop,cropBottom,topGap,bottomMargin) < 0 :
        print("\nError: None of cropping, bottomMargin or topGap parameters can be negative.\n\n")
        return() 
    
    
    imgs_=[im for im in os.listdir(imagPath) if (im.lower().endswith(".jpg")==True and im.startswith(".")==False)] ## list all jpg files except the file starting with "." character
    if len(imgs_)==0:
        print("No images found within the specified directory: "+imagPath)
        return()
    else:
        
        if len(imgs_)>20:
            imgs_=imgs_[0:20]
            
        imgs_.sort()
        for imgN in imgs_:
            img_=cv2.imread(imagPath+imgN)
            
            ##Cropping the image
            img_=img_[:, cropLeft:(img_.shape[1]-cropRight),:]
            img_=img_[cropTop:(img_.shape[0]-cropBottom), :, :]
            
            
            img=maskLines3C(img_, bottomMar=bottomMargin,topGap=topGap, col=marginCol, lThick=pen_)
            showFig(img,size=figSize,title=imgN+"\nbottomMargin:"+str(bottomMargin)+" topGap: "+str(topGap) )

            
## Takes a directory of images and displays images (4Panel) showing the bottomMarging and topGap   
def viewImageMargins4Panel(imagPath,bottomMargin=60, topGap=100, cropLeft=0, cropRight=0, cropTop=0,cropBottom=0, marginCol=(255,255,255),figSize=5,pen_=1):
    
    if os.path.isdir(imagPath)==False:
        print("The directory '"+imagPath+"' does not exist!")
        return()
    
    ## Checking the validity of croping, bottom margin and top gap parameters
    if min(cropLeft, cropRight,cropTop,cropBottom,topGap,bottomMargin) < 0 :
        print("\nError: None of cropping, bottomMargin or topGap parameters can be negative.\n\n")
        return() 
  
    imgs_=[im for im in os.listdir(imagPath) if (im.lower().endswith(".jpg")==True and im.startswith(".")==False)] ## list all jpg files except the file starting with "." character
    if len(imgs_)==0:
        print("Sorry, the specified directory containes not images!")
        return()
    else:
        if len(imgs_)>20:
            imgs_=imgs_[0:20]
        
        imgs_.sort()
        for imgN in imgs_:
            img_1=cv2.imread(imagPath+imgN)
            
            ##Cropping the image
            img_1=img_1[:, cropLeft:(img_1.shape[1]-cropRight),:]
            img_1=img_1[cropTop:(img_1.shape[0]-cropBottom), :, :]
            
            corr={}
            corr["a"]=[0,int(img_1.shape[0]/2),0,int(img_1.shape[1]/2)]
            corr["b"]=[0,int(img_1.shape[0]/2),int(img_1.shape[1]/2)+1,img_1.shape[1]]
            corr["c"]=[int(img_1.shape[0]/2)+1,img_1.shape[0],0,int(img_1.shape[1]/2)]
            corr["d"]=[int(img_1.shape[0]/2)+1,img_1.shape[0],int(img_1.shape[1]/2)+1,img_1.shape[1]]
                
            for panel in ["a","b","c","d"]:
                #img=maskLines3C(img_, bottomMar=bottomMargin,topGap=topGap, col=marginCol)
                #showFig(img,size=figSize,title=imgN+"\nbottomMargin:"+str(bottomMargin)+" topGap: "+str(topGap) )
                
                pos=corr[panel]
                img_=img_1[pos[0]:pos[1],pos[2]:pos[3],:]
                img=maskLines3C(img_, bottomMar=bottomMargin,topGap=topGap, col=marginCol, lThick=pen_)
                showFig(img,size=figSize,title=imgN+"_"+panel+"\nbottomMargin:"+str(bottomMargin)+" topGap: "+str(topGap) )
                
                
## Displays the image
def showFig(img_,size=10,title=""):
	out=plt.figure(figsize = (size,size))
	plt.title(title)
	plt.imshow(img_,cmap="gray")

## Tests the model reporting several metrics 
def test_model(X_, Y_, Model_):

    pred = Model_.predict(X_)
    precision = metrics.precision_score(Y_, pred, average='weighted', labels=np.unique(pred))
    recall = metrics.recall_score(Y_, pred, average='weighted', labels=np.unique(pred))
    f1 = metrics.f1_score(Y_, pred, average='weighted', labels=np.unique(pred))
    accuracy = metrics.accuracy_score(Y_, pred)

    print ('--------------------------------')
    print ('[RESULTS] Accuracy: %.2f' %accuracy)
    print ('[RESULTS] Precision: %.2f' %precision)
    print ('[RESULTS] Recall: %.2f' %recall)
    print ('[RESULTS] F1: %.2f' %f1)
    print ('--------------------------------')

## get a range of values by intervals    
def getSigmaz(gaussNumber=30):
    initSigma=1
    currSigma=0
    ret=[]
    for x in range(gaussNumber):
        currSigma=currSigma+initSigma
        ret.append(currSigma)
    return(ret)
    
   
    
## Extracts features from the original  training images
## Generates features using   Origianl images GABOR, GAUSSIAN kernel
def ftGenerator(img_,gaussianF_, medianF): 
    
	df=pd.DataFrame()
	img2_=img_.reshape(-1)
	df["Original_image"]=img2_

	#Gaussian filter sigm  3 7
	#print("  Gaussian filter")

	GaussianFilter="ON"
	if GaussianFilter=="ON":
		sigmaz=getSigmaz(gaussianF_)
		for sigma_ix in range(gaussianF_):
			sigma_=sigmaz[sigma_ix]
        
			gaussian_label=" Gaussian_"+str(sigma_) 
			gaussian_img=nd.gaussian_filter(img_,sigma=sigma_)
			gaussian_img1=gaussian_img.reshape(-1)
			df[gaussian_label]=gaussian_img1

	GaborFilter="OFF"
	if GaborFilter=="ON":
		#Gabor filters	

		# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
		# ksize - size of gabor filter (n, n)
		# sigma - standard deviation of the gaussian function
		# theta - orientation of the normal to the parallel stripes
		# lambda - wavelength of the sunusoidal factor
		# gamma - spatial aspect ratio
		# psi - phase offset
		# ktype - type and range of values that each pixel in the gabor kernel can hold

		phi=0
		num=1
		kernels=[]
		for n in range(1,4):
			#theta=theta /4*np.pi      
			theta= 2*np.pi/n      
                                                  # angle
			for sigma in (3,3) :                                        #  standard deviation 
				for lamda in  np.arange(1.5,np.pi, np.pi /4):             # range of wave lengths
					for gamma in (0.05, 0.5):                           #
						gabor_label="Gabor_"+str(num)
						
						sys.stdout.write("\r"+gabor_label)
						sys.stdout.flush()
        
						ksize=5
						kernel=cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,phi,ktype=cv2.CV_32F)
						kernels.append(kernel)
						fimag=cv2.filter2D(img_,cv2.CV_8UC3, kernel)
                    
						filtered_img=fimag.reshape(-1)
						num +=1
						#strng=str(num)+" "+gabor_label+" theta:"+str(theta)+" sigma:"+str(sigma)+" lambda:"+str(lamda)+" gamma:"+str(gamma)
						#print(strng+"\n")
						#showFig(fimag.reshape((img_.shape)),5,strng)
                
						#if num in [15,24,6,5,31,32,29,23,30,21,4,16,7]:   #select only parameter increasin accuracy ( 0.01>=)
						df[gabor_label] = filtered_img
						#print(str(num)+"\t"+gabor_label+"\ttheta\t"+str(theta)+"\tsigma\t"+str(sigma)+"\tlambda\t"+str(lamda)+"\tgamma\t"+str(gamma)+"\n") 
    
	EdgeDetect="OFF"
	if EdgeDetect=="ON":
		print("\nEdge Roberts, Sobel, Scharrr filters")
		# Edge Roberts
		edge_roberts=roberts(img_)		; print("Roberts")
		edge_roberts1=edge_roberts.reshape(-1)
		df["Roberts"]=edge_roberts1		
		# Edge Sobel
		edge_sobel=sobel(img_)			; print("Sobel")
		edge_sobel1=edge_sobel.reshape(-1)
		df["Sobel1"]=edge_sobel1	
		
		# Edge Scharr
		edge_scharr=scharr(img_)		; print("Scharr")
		edge_scharr1=edge_scharr.reshape(-1)
		df["Scharr"]=edge_scharr1
		
		# Edge Scharr v
		edge_scharr_v=scharr_v(img_)		#; print("Scharr_v")
		edge_scharr_v1=edge_scharr_v.reshape(-1)
		df["Scharr_v"]=edge_scharr_v1
	
		# Edge Scharr h
		edge_scharr_h=scharr_h(img_)		#; print("Scharr_h")
		edge_scharr_h1=edge_scharr_h.reshape(-1)
		df["Scharr_h"]=edge_scharr_h1
		
		# Edge Prewitt
		edge_prewitt = prewitt(img_)           #; print("Prewitt")
		edge_prewitt1=edge_prewitt.reshape(-1)
		df["Prewitt"]=edge_prewitt1	
		
		# Edge Prewitt v
		edge_prewitt_v = prewitt_v(img_)           #; print("Prewitt_v")
		edge_prewitt_v1=edge_prewitt_v.reshape(-1)
		df["Prewitt_v"]=edge_prewitt_v1	

		# Edge Prewitt h
		edge_prewitt_h = prewitt_h(img_)           #; print("Prewitt_h")
		edge_prewitt_h1=edge_prewitt_h.reshape(-1)
		df["Prewitt_h"]=edge_prewitt_h1	

	
	MedianFilter="ON"
	if MedianFilter=="ON":
        
		if medianF>2:
			#print("\nMedian filter")
        #Median with sigma 
			for sigma_ in range(1,medianF,1):
				median_label="Median_"+str(sigma_)		#;print(median_label)

				#sys.stdout.write("\r"+median_label)
				#sys.stdout.flush()

				median_img=nd.median_filter(img_,size=sigma_)
                #showFig(median_img)
				median_img1=median_img.reshape(-1)
				df[median_label]=median_img1    
	return(df)



## Shows fly image with coordiates highlighted - used for debuging purposes
def show_corrs():           
    newPlt=draw_points(imgO_,coors,color=(255,255,255),pwd=10)
    showFig(newPlt,10)

## Writes on disk fly image and append coordinates to the csv  [1 panel image]  
def writeFlies(pnameW,fnameW,imgOW,coorsW,areasW,clmpW,margW,topG,pen_) :        
    
    # Final Figures [capeted in original size, to use for manual editing]
    newPlt=draw_points(imgOW,coorsW,color=(255,255,255),pwd=10)
    outname=fnameW.lower().split(".jpg")[0]+"_flies_"+str(len(coorsW))+".jpg"
    io.imsave(pnameW+outname,newPlt)  
    
    ## Images to included in pdf and delete [add the title of ]
    newPlt= maskLines3C(newPlt,margW,topG,col=(255,0,255),lThick=pen_)
    
    outname_=fnameW.lower().split(".jpg")[0]+"_flies_"+str(len(coorsW))+"_.jpg"
    out=plt.figure(figsize=(13, 13))
    plt.imshow(newPlt, cmap='gray')
    plt.title(outname,loc="center",fontsize=13)
    plt.savefig(pnameW+outname_)
    plt.close()
    
    csvName=fnameW.lower().split(".jpg")[0]+"_flies_"+str(len(coorsW))+".csv"
    
    h=open(pnameW+csvName,"w")
    f_counter=1
    for j in range(len(coorsW)): 
        r=coorsW[j] ; a=areasW[j]; c=clmpW[j]
        h.write(outname+"\tfly_"+str(f_counter)+"\t"+str(round(r[1],2))+"\t"+str(round(imgOW.shape[1]-r[0],2))+"\t"+str(a)+"\t"+str(c)+"\n")
        f_counter+=1
    h.close()
    return(outname)
 
## Writes on disk fly image and append coordinates to the csv  [4 panel image]
def writeFlies4(pnameW,fnameW,panelW,imgOW,coorsW,areasW,clmpW,margW,topG,pen_) :        
    newPlt=draw_points(imgOW,coorsW,color=(255,255,255),pwd=10)
   
    outname=fnameW.lower().split(".jpg")[0]+"_panel_"+panelW+"_flies_"+str(len(coorsW))+".jpg"
    io.imsave(pnameW+outname,newPlt)  
    
    newPlt=maskLines3C(newPlt,margW,topG,(255,0,255),lThick=pen_)
    
    outname_=fnameW.lower().split(".jpg")[0]+"_panel_"+panelW+"_flies_"+str(len(coorsW))+"_.jpg"
    out=plt.figure(figsize=(13, 13))
    plt.imshow(newPlt, cmap='gray')
    plt.title(outname,loc="center",fontsize=13)
    plt.savefig(pnameW+outname_)
    plt.close()
    
    csvName=fnameW.lower().split(".jpg")[0]+"_panel_"+panelW+"_flies_"+str(len(coorsW))+".csv"
    
    h=open(pnameW+csvName,"w")
    f_counter=1
    for j in range(len(coorsW)): 
        r=coorsW[j] ; a=areasW[j]; c=clmpW[j]
        h.write(outname+"\tfly_"+str(f_counter)+"\t"+str(round(r[1],2))+"\t"+str(round(imgOW.shape[1]-r[0],2))+"\t"+str(a)+"\t"+str(c)+"\n")
        f_counter+=1
    h.close()
    return(outname)

## Segmants a clamped region
def segm_clamp(fimgI,bboxI,areaI,nFlies):
   
   #fimgI=fimgs_[x];bboxI=bbox_[x];areaI=areas_[x];medianFlyAI=medianFlyA
   #showFig(erosion(fimgI, square(2)));
   distance = nd.distance_transform_edt(fimgI)
   distance=nd.gaussian_filter(distance,3)
   
   # Old fashiion skimage 0.18
   '''
   local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=fimgI)
   markers = nd.label(local_maxi)[0]
   labels = watershed(-distance, markers, mask=fimgI)
   '''

   #  New way for skimage 0.20 
   coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=fimgI)
   mask = np.zeros(distance.shape, dtype=bool)
   mask[tuple(coords.T)] = True
   markers, _ = nd.label(mask)
   labels = watershed(-distance, markers, mask=fimgI)
    

   cr,ar,bb,fi= get_prop(labels)
   
   ## Get indices of top largest labeled areas
   ix=sorted(range(len(ar)), key=lambda i: ar[i])[-nFlies:]
   corrOut=[]; corrOut1=[]; areaOut=[]
   for v in ix: 
       corrOut.append((int(round(cr[v][0]+bboxI[0],0)),int(round(cr[v][1]+bboxI[1]))))
       corrOut1.append((int(round(cr[v][0],0)),int(round(cr[v][1]))))
       areaOut.append(ar[v])
    
   return(corrOut,areaOut)   
   
## Performs segmantation and used local maximum, watershed algorithm, small objects and dynamic erosion algorithms
## to segment regions of clamped flies
## Works for 1 or 4 panel images    
def dyn_Segm_DW14(pname_, pname_F, fname_,panel_,imgO_, segmented_, maxFlyA_,minFlyA_, expFlies_,bottomMarg_,topG_,plateType_, pen, prank_):
    
    #test line
    #pname_=pathIMG_out;pname_F=pathIMG_out_fail; fname_=imgFile; imgO_=imgO; segmented_=segmented
    msg_=""
    
    binary=segmented_.copy()
    binary[segmented_ > 250]=1
    binary[segmented_ <= 250]=0
    
    #binary=erosion(binary, square(2))  ## reduced the sides slight to separate the objects
    binaryL=label(binary,background=0)
    
    ## Denoising the labeled images
    if prank_==0:
        print(" ->Denoising the obtained image")
    
    msg_=msg_+" ->Denoising the obtained image"
    
    smallObj=minFlyA_ #-25##+- some pixels
    
    binaryL=remove_small_objects(binaryL,smallObj, connectivity=1,in_place=True)
    coors_,areas_,bbox_,fimgs_=get_prop(binaryL)     
    
    ## When no flye detected
    if len(coors_)==0:
        if plateType_.lower()=="1panel":   
                if prank_==0:
                    print(" ** Warning: no fly was ditected in "+ fname_+" - this image is skipped") 
                msg_=msg_+" \n** Warning: no fly was ditected in "+ fname_+" - this image is skipped"
        elif plateType_.lower()=="4panel": 
                if prank_==0:
                    print(" ** Warning: no fly was ditected in "+ fname_+" , panel "+panel_+" - this panel is skipped")   
                msg_=msg_+" ** Warning: no fly was ditected in "+ fname_+" , panel "+panel_+" - this panel is skipped"
        return()
    
    ## Quality control of coordinates
    maxFlyA=maxFlyA_ #+- some pixels
   
    coors=[]; areas=[]; clmp=[]; outNam=""
    
    ## Identify area of regions that represent clamped up flies
    for x in range(len(areas_)):
        if (areas_[x] > smallObj) and (areas_[x]<= maxFlyA):
            coors.append(coors_[x])
            areas.append(areas_[x])
            clmp.append(0)
    
    ## Save data if converged - getting  expected number of fly (csv + figures)
    if len(coors)==expFlies_:
        if plateType_.lower()=="1panel":            
                outNam=writeFlies(pname_,fname_,imgO_,coors,areas,clmp,bottomMarg_,topG_,pen_=pen)
        elif plateType_.lower()=="4panel": 
                outNam=writeFlies4(pname_,fname_,panel_,imgO_,coors,areas,clmp,bottomMarg_,topG_,pen_=pen)
        if prank_==0:
            print(" * Converged, noise cutoff (min fly Area) "+str(smallObj)+" --> number of flies "+str(len(coors))+";  minimum area "+str(np.min(areas))+" pxl;  miximum area "+str(np.max(areas))+" pxl")   
        msg_=msg_+"\n * Converged, noise cutoff (min fly Area) "+str(smallObj)+" --> number of flies "+str(len(coors))+";  minimum area "+str(np.min(areas))+" pxl;  miximum area "+str(np.max(areas))+" pxl"
        return(msg_)
    
    ## More than expected number of flies
    if len(coors)<expFlies_:
        nFlies=expFlies_-len(coors)
        
        # get all clumped region > threshold & the number of flies expected in each region
        iz=[i for i in range(len(areas_)) if areas_[i]>maxFlyA]
        areas__=[];  bbox__=[];  nFlies__=[]; fimgs__=[]
        for z in iz: 
            areas__.append(areas_[z])
            bbox__.append(bbox_[z])
            fimgs__.append(fimgs_[z])
        
        
        ## Get expected number of flies in each clamped areas
        for y in range(len(areas__)): 
            aFly=np.sum(areas__)/nFlies
            zFlies=int(round((areas__[y]/aFly),0))
            nFlies__.append(zFlies)
        
        ### segmenting all clamped objects/regions
        for x in range(len(areas__)):
            if prank_==0:
                print(" -> segmenting "+str(areas__[x])+ " pixels for "+str(nFlies__[x])+" flies")
            msg_=msg_+ "\n -> segmenting "+str(areas__[x])+ " pixels for "+str(nFlies__[x])+" flies"
            
            new_coors,new_areas= segm_clamp(fimgs__[x],bbox__[x],areas__[x],nFlies__[x])
            coors=coors+new_coors
            areas=areas+new_areas
            clmp=clmp+(len(new_areas)*[1])
        
        ### Save data if converged or not
        if len(coors)==expFlies_:
            if plateType_.lower()=="1panel":            
                 outNam=writeFlies(pname_, fname_, imgO_, coors, areas, clmp, bottomMarg_, topG_, pen_=pen)
            elif plateType_.lower()=="4panel": 
                 outNam=writeFlies4(pname_, fname_, panel_, imgO_, coors, areas, clmp,bottomMarg_, topG_,pen_=pen)
            if prank_==0:
                print(" * Converged, noise cutoff (min fly Area) "+str(smallObj)+" --> number of flies "+str(len(coors))+";  minimum area "+str(np.min(areas))+";  miximum area "+str(np.max(areas)))
            msg_=msg_+"\n * Converged, noise cutoff (min fly Area) "+str(smallObj)+" --> number of flies "+str(len(coors))+";  minimum area "+str(np.min(areas))+";  miximum area "+str(np.max(areas))
            return(msg_)
            
        else:
            if plateType_.lower()=="1panel":            
                 outNam=writeFlies(pname_F, fname_, imgO_, coors, areas, clmp, bottomMarg_, topG_, pen_=pen)
            elif plateType_.lower()=="4panel": 
                 outNam=writeFlies4(pname_F, fname_, panel_, imgO_, coors, areas, clmp, bottomMarg_, topG_ ,pen_=pen)
            if prank_==0:
                print(" ** Failed to detect the correct number of flies! Noise cutoff (min fly Area) "+str(smallObj)+" --> number of flies "+str(len(coors))+";  minimum area "+str(np.min(areas))+" pxl;  miximum area "+str(np.max(areas))+" pxl")   
            msg_=msg_+"\n ** Failed to detect the correct number of flies! Noise cutoff (min fly Area) "+str(smallObj)+" --> number of flies "+str(len(coors))+";  minimum area "+str(np.min(areas))+" pxl;  miximum area "+str(np.max(areas))+" pxl"
            return(msg_)
    
    ## More than expected number of flies       
    if len(coors)>expFlies_:
        if plateType_.lower()=="1panel":            
             outNam=writeFlies(pname_F,fname_,imgO_,coors,areas,clmp,bottomMarg_,topG_, pen_=pen)
        elif plateType_.lower()=="4panel": 
             outNam=writeFlies4(pname_F, fname_, panel_, imgO_, coors, areas, clmp, bottomMarg_, topG_, pen_=pen)
        if prank_==0:
            print(" ** Many flies or noise are detect! Noise cutoff (min fly Area) "+str(smallObj)+" --> number of flies "+str(len(coors))+";  minimum area "+str(np.min(areas))+" pxl;  miximum area "+str(np.max(areas))+" pxl")
        msg_=msg_+"\n ** Many flies or noise are detect! Noise cutoff (min fly Area) "+str(smallObj)+" --> number of flies "+str(len(coors))+";  minimum area "+str(np.min(areas))+" pxl;  miximum area "+str(np.max(areas))+" pxl"
        return(msg_)


#####################
## Post-processing ##
#####################

## Writes on disk fly image and append coordinates to the csv   
def writeFlies_ps(pnameW,fnameW,imgOW,coorsW,areasW) :   
     
    #This line is for testing
    #pnameW=pname_; fnameW=fname_; imgOW=imgO_; coorsW=coors; areasW=areas; clmpW=bbox
    newPlt=draw_points(imgOW,coorsW,color=(255,0,255),pwd=10)
    outname=fnameW.lower().split(".jpg")[0]+"_flies_"+str(len(coorsW))+"_ps.jpg"
    io.imsave(pnameW+outname,newPlt)  
    
    outname_=fnameW.lower().split(".jpg")[0]+"_flies_"+str(len(coorsW))+"_ps_.jpg"
    
    out=plt.figure(figsize=(13, 13))
    plt.imshow(newPlt, cmap='gray')
    plt.title(outname,loc="center",fontsize=13)
    plt.savefig(pnameW+outname_)
    plt.close()
    
    csvName=fnameW.lower().split(".jpg")[0]+"_flies_"+str(len(coorsW))+"_ps.csv"
    
    h=open(pnameW+csvName,"w")
    f_counter=1
    for j in range(len(coorsW)): 
        r=coorsW[j] ; a=areasW[j]; #c=clmpW[j]
        #h.write(fnameW+"\tfly_"+str(f_counter)+"\t"+str(round(r[0],2))+"\t"+str(round(r[1],2))+"\t"+str(a)+"\t"+str(c)+"\n")
        h.write(outname+"\tfly_"+str(f_counter)+"\t"+str(round(r[1],2))+"\t"+str(round(imgOW.shape[1]-r[0],2))+"\t"+str(a)+"\t0\n")
        f_counter+=1
    h.close()
    return(outname)

## Used to peak the flies colored in "red"  
def dyn_Segm_posproc(pname_,fname_,imgO_,imgGray_):
    
    #Line for testing
    #pname_=pathIMG_out; fname_=imgFile;imgO_=imgO;imgGray_=imgGray 
    
    binary=binFig(imgGray_,250,255,0)
    
    binaryL=label(binary,background=0)
    
    ## Denoising the labeled images
    smallObj=20 
    binaryL=remove_small_objects(binaryL,smallObj, connectivity=1,in_place=True)
    coors,areas,bbox,img =get_prop(binaryL)       
    
    outNam=writeFlies_ps(pname_,fname_,imgO_,coors,areas)
    
    return(outNam)
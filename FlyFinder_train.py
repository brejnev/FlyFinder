#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:57:30 2020

@author: brejnev.muhire
"""

# This script trains a random forest classifer that is used for fly image thresholding and segmentation
# It takes input the original and labeled images, it generates feature using gaussian and median kernels
# And trains a random forest classifier
# It save the model within the program directory "RFC_model" subdirectory. It report the model accuracy and
# the distribution of fly area in (pxl) which can be used during segmentation step

# Usage
#------

'''
python FlyFinder_predict.py --modelName M4Panel_GF40_MF5_Est30_mar100_topG10_RFC \
                                          --plateType 4Panel \
                                          --minArea 100 \
                                          --maxArea 400 \
                                          --nGaussianFilters 40 \
                                          --nMedianFilters 5 \
                                          --bottomMargin 200 \
                                          --topGap 50 \
                                          --cropLeft 0 \
                                          --cropRight 0 \
                                          --cropTop 0 \
                                          --cropBottom 0 \
                                          --numFlies 20 \
                                          --inPath input_directory
'''


from FF_utils import *

print("\n-* FlyFinder_v1.0 *-\n")


pathApp=os.path.dirname(os.path.abspath(__file__))+"/"
pathRFC=pathApp+"RFC_model/"
if os.path.isdir(pathRFC)==False:    
    os.mkdir(pathRFC)
    
if sys.argv[1].lower()=="models":
    models=[m for m in os.listdir(pathRFC) if m.endswith("pkl")==True]
    if len(models)>0:
          print("Available trained random forest classifiers:\n")
          for m in models: print(m.split(".pkl")[0])
          print("\n")
          sys.exit() 
    else:
          print("No random forest classifier trained yet.\n")
          sys.exit()
          
## Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modelName', action='store', help='the name you want to give to your random forest classifier.',type=str,required=True)
parser.add_argument('--plateType', action='store', help='the 1 panel plate \'1panel\' or 4 panel plate \'4panel\'.',type=str,required=True)
parser.add_argument('--nGaussianFilters', action='store', help='the number of gaussian filters. Default=40.',type=int,required=True)
parser.add_argument('--nMedianFilters', action='store', help='the number of median filters. Default=40.',type=int,required=True)
parser.add_argument('--nEstimators', action='store', help='the number of estimators (trees). Default=30.',type=int,required=True)
parser.add_argument('--bottomMargin', action='store',help='an integer for bottomMargin (pixels) between the triangle corners and the edge of the photo.', type=int, required=True)
parser.add_argument('--topGap', action='store',help='an integer for a gap (pixels) added to the mask at the top of triangle.', type=int, required=True)
parser.add_argument('--cropLeft' , action='store', help='crop the left side of the input image with a given amount of pxl. Default=0.',type=int,required=True)
parser.add_argument('--cropRight' , action='store', help='crop the right side of the input image with a given amount of pxl. Default=0.',type=int,required=True) 
parser.add_argument('--cropTop'  , action='store', help='crop the top side of the input image with a given amount of pxl. Default=0.',type=int,required=True)
parser.add_argument('--cropBottom', action='store', help='crop the bottom side of the input image with a given amount of pxl. Default=0.',type=int,required=True)
parser.add_argument('--inPath', action='store', help='the path of directory containing labeled and unlebeld photos.',type=str,required=True)

args = parser.parse_args()

print("\nTraining a random forest classifier for image segmentation (",args.plateType+" images) \n----------------------------------------------------------------------------------")
 
 
## Create the RFC directory if it does not exitst  # if the RFC already exist, prompt whether to overwrite it or not      
modelFile=pathRFC+args.modelName+".pkl"

if  os.path.isfile(modelFile)==True:
     print("The RFC '"+args.modelName+"' already exists.\n Please choose a different name or delete this model using 'rm -v ./RFC_model/"+args.modelName+ ".pkl'")
     sys.exit()
    
## Checking the validity of croping, bottom margin and top gap parameters
if min(args.cropLeft, args.cropRight,args.cropTop,args.cropBottom,args.topGap,args.bottomMargin) < 0 :
    sys.stderr.write("\nError: None of cropping, bottomMargin or topGap parameters can be negative.\n\n")
    sys.exit()
        
    
pathTrain=os.path.abspath(args.inPath)
if os.path.isdir(pathTrain) ==False:
      sys.stderr.write("\nError: the input directory given does not exit: %s\n\n"%(pathTrain))
      sys.exit()     
if pathTrain.endswith("/")==False:
      pathTrain=pathTrain+"/"
    

labeled=[r for r in os.listdir(pathTrain) if (r.lower().endswith("_lbl.jpg")==True and r.lower().startswith("._")==False)]
originals=[]
for r in labeled: originals.append(r.lower().split("_lbl.jpg")[0]+".jpg")   
print("\nOriginal images:"+str(originals))
print("Labeled images: "+str(labeled))
print("Training set size: "+str(len(labeled)) +" files\n")
if len(labeled)==0:
    print("Error: no labeled images detected! They should be named as 'NAME.jpg' and 'NAME_lbl.jpg'\nfor the original and the labeled image respectively.\n")
    sys.exit()

DF=pd.DataFrame()
flyAreas=[]

## Training for a 1Panel image classifier
if args.plateType=="1Panel":
    for i in range(len(labeled)):
          print(originals[i]+"\t"+labeled[i])
            
          ## Original image
          imgO=cv2.imread(pathTrain+originals[i])
            
          ## Cropping the image
          imgO=imgO[:, args.cropLeft:(imgO.shape[1]-args.cropRight),:]
          imgO=imgO[args.cropTop:(imgO.shape[0]-args.cropBottom), :, :]
        
          imgO=cv2.cvtColor(imgO, cv2.COLOR_BGR2GRAY)
          imgO=maskFig(imgO, args.bottomMargin, args.topGap, 255)

          imgO=imgO.reshape(-1)
            
          ## Indices for processing only pixels whithin the triangle/mask
          indx=np.argwhere(imgO<255)
            
            
            
            
          ## Labeled images
          imgL=cv2.imread(pathTrain+labeled[i])
            
          ## Cropping the image
          imgL=imgL[:, args.cropLeft:(imgL.shape[1]-args.cropRight),:]
          imgL=imgL[args.cropTop:(imgL.shape[0]-args.cropBottom), :, :]
        
          imgL=cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
          imgL=maskFig(imgL,args.bottomMargin,args.topGap,0)
          imgLCopy=imgL.copy()    ## For measuring the fly areas
          imgL=imgL.reshape(-1)
          
          #imgL=binFig(imgL,220,255,0)
          imgL=binFig(imgL,250,255,0)
            
          ## Generate features
          tmpO=imgO.copy()
            
            
          print(" Generating features -- please wait, it might take some minutes ...")  
          tDF=ftGenerator(tmpO[indx,],args.nGaussianFilters, args.nMedianFilters)
          tDF["Labels"]= imgL[indx,].reshape(-1)
           
            
          '''  
          print(" Generating features -- please wait, it might take some minutes ...")
          
          sigma_min = 1
          sigma_max = 16
          features_func = partial(feature.multiscale_basic_features, intensity=True, edges=False, texture=True, sigma_min=sigma_min, sigma_max=sigma_max,channel_axis=-1)
          tDF= features_func(tmpO[indx,])
          '''
            
            
          if DF.shape[0]==0 and DF.shape[1]==0:
                    DF=tDF
          else:
                    DF=DF.append(tDF, ignore_index=True)
          print(DF.shape)
          print("\n")
        
          ## Get fly areas from the labeled image
          binary=imgLCopy.copy()
          binary[imgLCopy > 250]=1
          binary[imgLCopy <= 250]=0
          binaryL=label(binary,background=0)
          binaryL=remove_small_objects(binaryL,10, connectivity=1,in_place=True) ## small object considered as noise in the labeled imageduring training = 10pxl
          flyAreas=flyAreas+get_areas(binaryL)   

## Training for a 4Panel image classifier

if args.plateType=="4Panel":
    for i in range(len(labeled)):
           print(originals[i]+"\t"+labeled[i])
            
           ## Original image and labeled images
           imgO_=cv2.imread(pathTrain+originals[i])
           imgL_=cv2.imread(pathTrain+labeled[i])
               
           ## Cropping the image
           imgO_=imgO_[:, args.cropLeft:(imgO_.shape[1]-args.cropRight),:]
           imgO_=imgO_[args.cropTop:(imgO_.shape[0]-args.cropBottom), :, :]
           imgL_=imgL_[:, args.cropLeft:(imgL_.shape[1]-args.cropRight),:]
           imgL_=imgL_[args.cropTop:(imgL_.shape[0]-args.cropBottom), :, :]
            
            
           ## convert from rgb to gray scale 
           imgO_=cv2.cvtColor(imgO_, cv2.COLOR_BGR2GRAY) 
           imgL_=cv2.cvtColor(imgL_, cv2.COLOR_BGR2GRAY)
            
            ## work throuth the panels
           if imgO_.shape!=imgL_.shape:
                  print("Training images are not equal size: original ="+str(imgO_.shape)+ " and labeled ="+str(imgL_.shape))
                  sys.exit()
            
            ## Cordinates of each panel
           corr={}
           corr["a"]=[0,int(imgO_.shape[0]/2),0,int(imgO_.shape[1]/2)]
           corr["b"]=[0,int(imgO_.shape[0]/2),int(imgO_.shape[1]/2)+1,imgO_.shape[1]]
           corr["c"]=[int(imgO_.shape[0]/2)+1,imgO_.shape[0],0,int(imgO_.shape[1]/2)]
           corr["d"]=[int(imgO_.shape[0]/2)+1,imgO_.shape[0],int(imgO_.shape[1]/2)+1,imgO_.shape[1]]
            
           for panel in ["a","b","c","d"]:
                   print("\n  -- Processing panel "+panel+" --")
                   pos=corr[panel]
                   imgO=imgO_[pos[0]:pos[1],pos[2]:pos[3]]
                   imgO=maskFig(imgO, args.bottomMargin, args.topGap, 255)
                   
                   io.imsave(pathRFC+args.modelName+"_Masked.jpg",imgO)  
                   
                   imgO=imgO.reshape(-1)
                    
                   ## Indices for processing only pixels whithin the triangle/mask
                   indx=np.argwhere(imgO<255)
                    
                   ## Labeled images
                   imgL=imgL_[pos[0]:pos[1],pos[2]:pos[3]]
                   imgL=maskFig(imgL,args.bottomMargin,args.topGap,0)
                   imgLCopy=imgL.copy()    ## For measuring the fly areas
                   imgL=imgL.reshape(-1)
                   #imgL=binFig(imgL,220,255,0)
                   imgL=binFig(imgL,250,255,0)
                    
                   ## Generate features
                   tmpO=imgO.copy()
                   tDF=ftGenerator(tmpO[indx,],args.nGaussianFilters,args.nMedianFilters)
                   tDF["Labels"]= imgL[indx,].reshape(-1)
                   
                   '''  
                   print(" Generating features -- please wait, it might take some minutes ...")
                   sigma_min = 1
                   sigma_max = 16
                   features_func = partial(feature.multiscale_basic_features, intensity=True, edges=False, texture=True, sigma_min=sigma_min, sigma_max=sigma_max,channel_axis=-1)
                   tDF= features_func(tmpO[indx,])
                   ''' 
                
                   if DF.shape[0]==0 and DF.shape[1]==0:
                         DF=tDF
                   else:
                         DF=DF.append(tDF, ignore_index=True)
                   print("  "+str(DF.shape))
                 
                    ## Get fly areas from the labeled image
                   binary=imgLCopy.copy()
                   binary[imgLCopy > 250]=1
                   binary[imgLCopy <= 250]=0
                   binaryL=label(binary,background=0)
                   binaryL=remove_small_objects(binaryL,10, connectivity=1,in_place=True) ## small object considered as noise in the labeled imageduring training = 10pxl
                   flyAreas=flyAreas+get_areas(binaryL)   
        
        
DF=DF.drop_duplicates()
print(" \nDimensions after removing dupliate rows:")
print(" "+str(DF.shape))        

start = time.time()

## Definition of the dependent variable
Y=DF["Labels"].values
    
## Definition of the independend variables 
X=DF.drop(labels=["Labels"],axis=1)

## Split data into test and train
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.1,random_state=42) 

## Train the Random Forest Algorithm
print("Trainning the RandomForestClassifier, number of estimators = ",args.nEstimators,"\n")
model=RandomForestClassifier(n_estimators=args.nEstimators, random_state=42)
model.fit(X_train, Y_train)

## Predidition 
prediction_test=model.predict(X_test)
#prediction_test=model.predict_proba(X_test)

h=open(pathRFC+args.modelName+".txt" ,"w")

confMat=metrics.confusion_matrix(y_true=Y_test, y_pred=prediction_test)
print("\n Confusion matrix:")
print(confMat)
h.write("\n Confusion matrix:")
h.write(str(confMat))


print("\nAccuracy =", metrics.accuracy_score(Y_test, prediction_test))
h.write("\nAccuracy ="+ str(metrics.accuracy_score(Y_test, prediction_test))+"\n")
features_list=list(X.columns)
feature_imp=pd.Series(model.feature_importances_,index=features_list).sort_values(ascending=False)
print("\nFeature_importance:\n--------------------")
print(type(feature_imp))
np.set_printoptions(threshold=np.inf)
print(feature_imp)
    
h.write("\nFeature_importance:\n--------------------\n")
h.write(str(feature_imp))

    
flyAreas.sort()
print("\n---Fly area in pxl---")
print(str(flyAreas))
print("Max fly area :"+str(np.max(flyAreas)))
print("Min fly area :"+str(np.min(flyAreas)))
print("Number of flies:"+str(len(flyAreas)))
h.write("\n\n---Fly area in pxl---\n")
h.write(str(flyAreas)+"\n")
h.write("\nMax fly area :"+str(np.max(flyAreas))+"\n")
h.write("Min fly area :"+str(np.min(flyAreas))+"\n")
h.write("Number of flies:"+str(len(flyAreas))+"\n")

## Save histogram of areas
num_bins = 50
n, bins, patches = plt.hist(flyAreas, num_bins, facecolor='blue', alpha=0.8,edgecolor='gray', linewidth=1.2)
plt.title("Fly Area - "+args.modelName)
plt.ylabel("Counts")
plt.xlabel("pixels")
plt.savefig(pathRFC+args.modelName+"_flyArea.png")

## Model Saving the model   
pickle.dump(model, open(modelFile, 'wb'))
print("\nModel saved in '"+modelFile+"'")
end = time.time()
print ("\nDuration: "+ str(round((end-start)/60,2)) + " min / "  + str(round((end-start),2))+" seconds\n\n") 
print("\n*---  FlyFinder RFC Trainer completed successfully ---*\n\n" )

h.write("\nDuration: "+ str(round((end-start)/60,2)) + " min / "  + str(round((end-start),2))+" seconds\n\n") 
h.write("\n*---  FlyFinder RFC Trainer completed successfully ---*\n\n" )

h.close()
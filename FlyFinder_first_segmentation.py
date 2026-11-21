#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:57:30 2020

@author: brejnev.muhire
"""
 
# This script reads the input images and uses the trained random classifiter to perform the thresholding. 
# The classifier helpss to predict whether a given pixel is associated with a fly or background giving a 1 or 0 binary response
# After thresholding, it perfoms image segmentation using scikit-image functions and for clamped images it uses functions for finding local maximum
# and watershed algorithm. Finally  it reports each fly's center of mass x-y coordinates.
#
# It can run in parralell or in serial.
#
# Usage:
#-------
#
# parallel:
'''
mpiexec -n 10  python FlyFinder_first_segmentation.py --modelName M4Panel_GF40_MF5_Est30_mar100_topG10_RFC \
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

# serial:
'''
python FlyFinder_first_segmentation.py --modelName M4Panel_GF40_MF5_Est30_mar100_topG10_RFC \
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


if __name__ == '__main__':
    
   
    from mpi4py import MPI
    
    #getting the ranks and the total number of processesors
    myrank=MPI.COMM_WORLD.Get_rank()
    nprocs=MPI.COMM_WORLD.Get_size()
    ## Timing
    starting=MPI.Wtime()
    
    
    if myrank==0:
         print("\n-* FlyFinder_v1.0 *-\n")
  
    pathApp=os.path.dirname(os.path.abspath(__file__))+"/"
    pathRFC=pathApp+"RFC_model/"
    
    
    if sys.argv[1].lower()=="models":
        models=[m for m in os.listdir(pathRFC) if m.endswith("pkl")==True]
        if len(models)>0:
              print("Avalable random forest classifiers:\n")
              for m in models: print(m.split(".pkl")[0])
              print("\n")
              sys.exit(); MPI.Finalize()   
        else:
              print("\nError: no random forest classifier found! You need first to train the model with 'FlyFinder_train.py' script.\n")
              sys.exit(); MPI.Finalize()   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', action='store', help='the name of the random forest classifier to use.',type=str,required=True)
    parser.add_argument('--plateType', action='store', help='the 1 panel plate \'1panel\' or 4 panel plate \'4panel\'.',type=str,required=True)
    parser.add_argument('--nGaussianFilters', action='store', help='the number of gaussian filters. Default=40.',type=int,required=True)
    parser.add_argument('--nMedianFilters', action='store', help='the number of median filters. Default=40.',type=int,required=True)
    parser.add_argument('--bottomMargin', action='store',help='an integer for bottomMargin (pixels) between the triangle corners and the bottom edge of the photo.', type=int, required=True)
    parser.add_argument('--topGap', action='store',help='an integer for a gap (pixels) added to the mask at the top of triangle.', type=int, required=True)
    parser.add_argument('--cropLeft' , action='store', help='crop the left side of the input image with a given amount of pxl. Default=0.',type=int,required=True)
    parser.add_argument('--cropRight' , action='store', help='crop the right side of the input image with a given amount of pxl. Default=0.',type=int,required=True) 
    parser.add_argument('--cropTop'  , action='store', help='crop the top side of the input image with a given amount of pxl. Default=0.',type=int,required=True)
    parser.add_argument('--cropBottom', action='store', help='crop the bottom side of the input image with a given amount of pxl. Default=0.',type=int,required=True)    
    parser.add_argument('--maxArea', action='store',help='an integer for expected maximum fly area (in pixels). This was estimated during training.', type=int,required=True)
    parser.add_argument('--minArea', action='store',help='an integer for expected minimum fly area (in pixels). This was estimated during training.', type=int,required=True)
    parser.add_argument('--numFlies', action='store',help='an integer for expected number of flies.', type=int,required=True)
    parser.add_argument('--inPath', action='store', help='the path of input directory containing photos to be segmented.',type=str,required=True)

    args = parser.parse_args()

    ## Input arguments
    pathIMG=os.path.abspath(args.inPath)
    topGap=args.topGap

    ## checks the existance of input directoy
    if os.path.isdir(pathIMG) ==False:
        if myrank==0:
            sys.stderr.write("\nError: the input directory given does not exit: %s\n\n"%(pathIMG))
        sys.exit(); MPI.Finalize() 

    ## Checking the validity of croping, bottom margin and top gap parameters
    if min(args.cropLeft, args.cropRight,args.cropTop,args.cropBottom,args.topGap,args.bottomMargin) < 0 :
        if myrank==0:
            sys.stderr.write("\nError: None of cropping, bottomMargin or topGap parameters can be negative.\n\n")
        sys.exit(); MPI.Finalize() 

    ## Adding a slash on full path if there is none    
    if pathIMG.endswith("/")==False:
        pathIMG=pathIMG+"/"

    ## Make the output directories
    pathIMG_out= pathIMG+"out/"
    pathIMG_out_fail=pathIMG+"outf/"    
    pathIMG_log=pathIMG+"log/"  
    
    
    if myrank==0: 
        
        ## Remove output directory if  they exists
        if os.path.isdir(pathIMG_out) ==False:    
             os.mkdir(pathIMG_out)
        else:
            shutil.rmtree(pathIMG_out)
            os.mkdir(pathIMG_out)
            
        if os.path.isdir(pathIMG_out_fail) ==False:    
             os.mkdir(pathIMG_out_fail)
        else:
            shutil.rmtree(pathIMG_out_fail)
            os.mkdir(pathIMG_out_fail)
    
        if os.path.isdir(pathIMG_log) ==False:    
             os.mkdir(pathIMG_log)
        else:
            shutil.rmtree(pathIMG_log)
            os.mkdir(pathIMG_log)
    
    MPI.COMM_WORLD.Barrier()
    

    h=open(pathIMG_log+"core"+str(myrank).zfill(2)+".out","a")
   
    
    if myrank==0:
        print("\n\n\n*--- Running FlyFinder Random Forest-based segmantation : core n. "+str(myrank).zfill(2)+ " ---*\n")
        print("-----------------------------------------------------------------------------------------------\n")
        print ("Started time: "+(str(datetime.datetime.now()))+"\n\n\n") 
        print("Active path : "+pathIMG+"\n")
        print("Settings :")
        print(args)
        ### Loading the RFC models
        print ("\nLoading the RFC model. Please wait ... \n")
    
    h.write("\n\n\n*--- Running FlyFinder Random Forest-based segmantation : core n. "+str(myrank).zfill(2)+ "---*\n")
    h.write("-----------------------------------------------------------------------------------------------\n")
    h.write("\nStarted time: "+(str(datetime.datetime.now()))+"\n\n\n")
    h.write("\nActive path : "+pathIMG+"\n")
    h.write("\nSettings :")
    h.write("\n"+str(args))
    h.write("\nLoading the RFC model. Please wait\n")
    h.write("\n---------------------------------------------\n")

    

    modelFile=pathRFC+args.modelName+".pkl"
 
    if os.path.isfile (modelFile)==False:
        if myrank==0:
            print("\nError: the specified model '"+args.modelName+"' is not available. Use the command 'FlyFinder_predict.py models' to see available model names.\n")
        h.write("\nError: the specified model '"+args.modelName+"' is not available. Use the command 'FlyFinder_predict.py models' to see available model names.\n")
        sys.exit(); MPI.Finalize()   
    
    
    RFC_model=pickle.load(open(modelFile,"rb"))
    
    if myrank==0:
        print ("  Model loaded successfully: "+modelFile)
     
    h.write("\n  Model loaded successfully : "+modelFile+"\n")
    
    imFiles=[r for r in os.listdir(pathIMG) if (r.lower().endswith(".jpg")==True and r.lower().startswith("._")==False)]
    
    noFiles=len(imFiles)
    
    if myrank==0:
        print("\n\n  Number of images : "+str(noFiles)+"\n")
        print(imFiles)
    
    h.write("\n\nTotal  Number of images : "+str(noFiles)+"\n")
    
    ### distribution of work load to processor
    if noFiles<nprocs:
        if myrank==0:
            sys.stderr.write("Error: the number of processors (%d) chosen is bigger than the number of images (%d) to segment!\nPlease use processor less or equal to %d\n\n"%(nprocs,noFiles,noFiles))
        h.write("\nError: the number of processors (%d) chosen is more than the images (%d) to segment!\nPlease use processor less or equal to %d\n\n"%(nprocs,noFiles,noFiles))
        h.close()
        MPI.Finalize(); sys.exit()
    else:
        chunk_size=0
       	if noFiles % nprocs== 0:
            chunk_size =int(noFiles/nprocs)
            start_in = myrank*chunk_size
            ending_in= start_in + chunk_size			
    
        elif noFiles % nprocs!=0:
            remainder=noFiles % nprocs
            chunk_size =int(noFiles/nprocs)
            if myrank<remainder:
                start_in=myrank*(chunk_size+1)               #getting average load + 1
                ending_in= start_in + (chunk_size+1)
            else:	
                start_in= int((myrank*chunk_size)+remainder) #getting average load
                ending_in= start_in + (chunk_size)	


    ## Parallel run  for 1Panel image segmanation
    if args.plateType.lower()=="1panel":
         for i in range(start_in,ending_in):   
            
              imgFile=imFiles[i]
              
              if myrank==0:
                  print("\n"+imgFile)
              
              if os.path.isfile (pathIMG_out+imgFile.lower().split(".jpg")[0]+"_out.jpg") or os.path.isfile (pathIMG_out_fail+imgFile.lower().split(".jpg")[0]+"_out.jpg"):
                  if myrank==0:  
                      print("\t"+imgFile+" is already processed, thus skipped!")
                  h.write("\n\t"+imgFile+" is already processed, thus skipped!\n" )
                  continue
              else:
                  imgO=cv2.imread(pathIMG+imgFile)
                  
                  ## Cropping the image
                  imgO=imgO[:, args.cropLeft:(imgO.shape[1]-args.cropRight),:]
                  imgO=imgO[args.cropTop:(imgO.shape[0]-args.cropBottom), :, :]

                  imgGray=cv2.cvtColor(imgO, cv2.COLOR_BGR2GRAY)
                  imgGray=maskFig(imgGray,args.bottomMargin,args.topGap,255);#showFig(imgGray)
                  
                  tmpGray=imgGray.reshape(-1)     # tmp image is used to only run the RFC on relavant pixels (within the MASK)
                  indx=np.argwhere(tmpGray<255)
                  
                  if myrank==0:  
                      sys.stdout.write(" Proc_" + str(myrank).zfill(2))
                      sys.stdout.write("\n Generating feature matrix using different kernel functions (Gaussian, and Median)\nPlease wait ...")
                  h.write("\n Proc_" + str(myrank).zfill(2))  
                  h.write("\n Generating feature matrix using different kernel functions (Gaussian, and Median)\nPlease wait ...")
                
                  X_=ftGenerator(tmpGray[indx,], args.nGaussianFilters, args.nMedianFilters)   #generate features only for the pixels wthin the mask
                  
                  if myrank==0:
                      print("proc_"+str(myrank).zfill(2)+" Running the random forest classifier")
                  
                  h.write("\nproc_"+str(myrank).zfill(2)+" Running the random forest classifier")
                        
                  results=RFC_model.predict(X_)
                  #results=RFC_model.predict_proba(X_); 
             
                  results=results.reshape(tmpGray[indx,].shape)
                  tmpGray[tmpGray>0]=0
                  tmpGray[indx,]=results 
                  
                  tmpGray=tmpGray.reshape(imgGray.shape)
                  
                  panel=None
          
                  msg=dyn_Segm_DW14(pathIMG_out, pathIMG_out_fail, imgFile, panel, imgO, tmpGray, args.maxArea, args.minArea, args.numFlies, args.bottomMargin, args.topGap, args.plateType.lower(), pen, prank_=myrank)
                  h.write(msg)
                    

    ## Parallel run  for 4Panel image segmanation
    if args.plateType.lower()=="4panel":
    
        for i in range(start_in,ending_in):   
            
            imgFile=imFiles[i]
            if myrank==0:
                print("\n"+imgFile)
            
            if os.path.isfile (pathIMG_out+imgFile.lower().split(".jpg")[0]+"_out.jpg") or os.path.isfile (pathIMG_out_fail+imgFile.lower().split(".jpg")[0]+"_out.jpg"):
                if myrank==0:
                    print("\t"+imgFile+" is already processed, thus skipped!")
                continue
            else:
                imgO_1=cv2.imread(pathIMG+imgFile)
                
                ## Cropping the image
                imgO_1=imgO_1[:, args.cropLeft:(imgO_1.shape[1]-args.cropRight),:]
                imgO_1=imgO_1[args.cropTop:(imgO_1.shape[0]-args.cropBottom), :, :]
                
                imgGray_=cv2.cvtColor(imgO_1, cv2.COLOR_BGR2GRAY)
                
                
                ## work throuhg the panals
                corr={}
                corr["a"]=[0,int(imgGray_.shape[0]/2),0,int(imgGray_.shape[1]/2)]
                corr["b"]=[0,int(imgGray_.shape[0]/2),int(imgGray_.shape[1]/2)+1,imgGray_.shape[1]]
                corr["c"]=[int(imgGray_.shape[0]/2)+1,imgGray_.shape[0],0,int(imgGray_.shape[1]/2)]
                corr["d"]=[int(imgGray_.shape[0]/2)+1,imgGray_.shape[0],int(imgGray_.shape[1]/2)+1,imgGray_.shape[1]]
                
                for panel in ["a","b","c","d"]:  
                    
                      if myrank==0:
                           print("\n -- Segmenting panel "+panel+" --")
                      h.write("\n -- Segmenting panel "+panel+" --\n")
                      pos=corr[panel]
                      
                      imgO=imgO_1[pos[0]:pos[1],pos[2]:pos[3],:]
                      imgGray=imgGray_[pos[0]:pos[1],pos[2]:pos[3]]
    
                      imgGray=maskFig(imgGray,args.bottomMargin,args.topGap,255); 
                      
                      tmpGray=imgGray.reshape(-1)     # tmp image is used to only run the RFC on relavant pixels (within the MASK)
                      indx=np.argwhere(tmpGray<255)
                      
                      
                      if myrank==0:
                           print(" Generating features, please wait")
                      h.write("\n Generating features, please wait\n")
                      X_=ftGenerator(tmpGray[indx,], args.nGaussianFilters, args.nMedianFilters)   #generate features only for the pixels wthin the mask
                      
                      if myrank==0:
                           print(" Running the randomforest flassifier")
                      h.write("\n Running the randomforest flassifier\n")
                      results=RFC_model.predict(X_)
                      #results=RFC_model.predict_proba(X_); 
                 
                      results=results.reshape(tmpGray[indx,].shape)
                      tmpGray[tmpGray>0]=0
                      tmpGray[indx,]=results 
                      
                      tmpGray=tmpGray.reshape(imgGray.shape)
              
                      msg=dyn_Segm_DW14(pathIMG_out,pathIMG_out_fail, imgFile, panel, imgO, tmpGray, args.maxArea, args.minArea, args.numFlies, args.bottomMargin, args.topGap, args.plateType.lower(), pen, prank_=myrank)
                      h.write(msg)
            
     
    ## Closing the files
    h.close()
    MPI.COMM_WORLD.Barrier()
    
    ## Combine all the images into on pdf
    if myrank==0:

        imagelist=[f for f in os.listdir(pathIMG_out) if (f.endswith("_.jpg")==True and f.startswith("._")==False)]
        if len(imagelist)>0:
            imagelist.sort()
            os.chdir(pathIMG_out)
            with open("out.pdf", "wb") as f:
                 f.write(img2pdf.convert(imagelist))
            for f in imagelist:
                os.remove(f)    

        imagelist=[f for f in os.listdir(pathIMG_out_fail) if (f.endswith("_.jpg")==True and f.startswith("._")==False)]
        if len(imagelist)>0:
            imagelist.sort()
            os.chdir(pathIMG_out_fail)
            with open("outf.pdf", "wb") as f:
                 f.write(img2pdf.convert(imagelist))
            print("done")
            for f in imagelist:
                os.remove(f)   
    
    
        ## Get the success rates
        out_=[f for f in os.listdir(pathIMG_out) if f.endswith("csv")]
        out_f_=[f for f in os.listdir(pathIMG_out_fail) if f.endswith("csv")]
        print("Passed ="+str(len(out_))+",   Failed ="+str(len(out_f_))+ ",   Success rate = "+str(round((len(out_)/(len(out_)+len(out_f_)))*100,1))+"%.\n")
    
        ending=MPI.Wtime()
      
        print ("Duration: "+ str(round((ending-starting)/60,2)) + " min / "  + str(round((ending-starting),2))+" seconds\n\n") 
        print("\n*---  FlyFinder segmentation completed successfully! "+ "---*\n\n" )
        print ("End time: "+(str(datetime.datetime.now()))+"\n\n")
    
    
    MPI.Finalize(); sys.exit()
    
    
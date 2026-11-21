#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:57:30 2020

@author: brejnev.muhire
"""
# This script reads the input images where flies are marked with white 10x10 pxl points and report each point is x-y coordinate.
# It can run in parralell or in serial.

# Usage:
#-------

# parallel:
'''
mpiexec -n 4  python FlyFinder_second_segmentation.py  --inPath input_directory
'''

# serial:
'''
python FlyFinder_second_segmentation.py  --inPath input_directory
'''



from FF_utils import *

if __name__ == '__main__':
    
   
    from mpi4py import MPI

    #getting the ranks and the total number of processesors
    myrank=MPI.COMM_WORLD.Get_rank()
    nprocs=MPI.COMM_WORLD.Get_size()
    ## Timing
    starting=MPI.Wtime()
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--inPath', action='store', help='the path of input directory containing photos to be segmented.',type=str,required=True)

    args = parser.parse_args()

    if myrank==0:
        
        print("\n-* FlyFinder_v1.0 *-\n")
        print("\nPerforming segmantation for image where flies are marked with 10x10 pxl white points (squares)\n----------------------------------------------------------------------------------------")
     

    ## Input arguments
    pathIMG=args.inPath 
    
    if os.path.isdir(pathIMG) ==False:
        sys.stderr.write("Error: input directory does not exit: %s"%(pathIMG))
        sys.exit()#MPI.Finalize()  
        
    if pathIMG.endswith("/")==False:
        pathIMG=pathIMG+"/"
    
    ## Make the input directory
    pathIMG_out= pathIMG+"out_ps/"

    if myrank==0:  

        if os.path.isdir(pathIMG_out) ==False:    
             os.mkdir(pathIMG_out)
        else:
            shutil.rmtree(pathIMG_out)
            os.mkdir(pathIMG_out)
    
    if myrank==0:
        print ("Started time: "+(str(datetime.datetime.now()))+"\n\n\n") 
        print("Active path : "+pathIMG +"\n\n")
    
    
    ###getting all files in the input directory
    imFiles=[r for r in os.listdir(pathIMG) if (r.lower().endswith(".jpg")==True and r.lower().startswith("._")==False)]
    
    noFiles=len(imFiles)
    
    if myrank==0:
        print("Number of images "+str(noFiles)+"\n\n")

    ### distribution of work load to processor
    
    if noFiles<nprocs:
        if myrank==0:
            sys.stderr.write("Error: the number of processors chosen is more than the images to segment!\nPlease use processor less or equal to than %d\n"%(noFiles))
        MPI.Finalize()
        sys.exit()
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

           
    #Parallel run
    for i in range(start_in,ending_in):   
    #This line is for testing
    #for i in range(13):
        imgFile=imFiles[i]
        imgFileOut= imgFile.split(".jpg")[0]+"_ps.jpg"
        
        if myrank==0:
            print("\n"+imgFile)
        imgO=cv2.imread(pathIMG+imgFile)
        imgGray=cv2.cvtColor(imgO, cv2.COLOR_BGR2GRAY) 
        dyn_Segm_posproc(pathIMG_out,imgFile,imgO,imgGray)    

    
    if myrank==0:    
        ending = MPI.Wtime()
        print ("Time elabsed: "+ str(round((ending-starting)/60,2)) + " min / "  + str(round((ending-starting),2))+" seconds \n\n") 

    MPI.COMM_WORLD.Barrier()
    
    ## Combine all the images into on pdf
    if myrank==0:

        imagelist=[f for f in os.listdir(pathIMG_out) if (f.endswith("_.jpg")==True and f.startswith("._")==False)]
        if len(imagelist)>0:
            imagelist.sort()
            os.chdir(pathIMG_out)
            with open("out_ps.pdf", "wb") as f:
                 f.write(img2pdf.convert(imagelist))
            print("done")
            for f in imagelist:
                os.remove(f)
            
        ending=MPI.Wtime()
        
        print ("Duration: "+ str(round((ending-starting)/60,2)) + " min / "  + str(round((ending-starting),2))+" seconds\n\n") 
        print("\n*---  FlyFinder post-processing segmentation completed successfully! "+ "---*\n\n" )
        print ("End time: "+(str(datetime.datetime.now()))+"\n\n")
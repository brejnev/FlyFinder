#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 04:29:16 2020

@author: brejnev.muhire
"""
# This script combines the csv tables containing fly x and y coordinates on the image.
# Combined tables that are from the fist run based on ML-RFC segmantation and the second 
# run where clamped/undetetected flies are manually edited using a white color, and the image is edited to remove errors. 

# Usage
#------

'''
python FlyFinder_maketable.py --inPath  input_directory

'''


import os, sys, argparse


parser = argparse.ArgumentParser()
parser.add_argument('--inPath', action='store', help='the path of input directory containing .csv file containng fly coordinates.',type=str,required=True)
args = parser.parse_args()

pathIMG=args.inPath

      
if os.path.isdir(pathIMG) ==False:
   sys.stderr.write("Error: input directory does not exit: %s\n\ne"%(pathIMG))
   sys.exit()
    
if pathIMG.endswith("/")==False:
   pathIMG=pathIMG+"/"
    
## Make the input directory
pathCSV_in= pathIMG+"out/"
pathCSV_in_ps=pathIMG+"outf/out_ps/"
   
## RFC based tables   
csv_in=[r for r in os.listdir(pathCSV_in) if r.endswith(".csv")==True and r.startswith("._")==False]
csv_in.sort()
noFiles=len(csv_in)

hOut=open(pathIMG+"FinalTable.csv","w")
hOut.write ("no\tname\tfly_no\tx_coor\ty_coor\tareas\tclamped\n" )
counter=1

# out
if noFiles>0:
  
    print("\n* RFC segmantation : Number of tables = %d\n"%(noFiles))
    
    for f in csv_in:
        lines=open(pathCSV_in+f).readlines()    
        for l in lines:
            hOut.write(str(counter)+"\t"+l)
            counter+=1
        print (f)
  
else:
    print("Error: not csv tables found in %s \n please check your input."%(pathCSV_in))

## outf : Post processing based tables
csv_in_ps=[r for r in os.listdir(pathCSV_in_ps) if r.endswith(".csv")==True and r.startswith("._")==False]
csv_in_ps.sort()
noFiles_ps=len(csv_in_ps)


if noFiles_ps>0:
     
   print("\n* Post-processsing : Number of tables = %d\n"%(noFiles_ps))
   for f in csv_in_ps:
        lines=open(pathCSV_in_ps+f).readlines()    
        for l in lines:
            hOut.write(str(counter)+"\t"+l)
            counter+=1
        print (f)
   
else:   
   print("\nNo csv tables found in %s \n please check your input."%(pathCSV_in_ps))
  
hOut.close()


print("\nThe combined table path \n%s "%(pathIMG+"FinalTable.csv"))

print("\n*---  FlyFinder final table generated successfully ✓---*\n\n")
    

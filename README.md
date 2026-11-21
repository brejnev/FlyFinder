# FlyFinder_v1.0
Drosophila Melanogaster has been a popular model organism to study autism. In some experiments mutants and wild-type flies are analyzed to mean change in their social distancing habits. Usually, flies are placed on a plate and pictures are taken and processed by a software to determine the fly location and distances from each other. Current imaging software used in biology such CellProfiler lacks learning capability to consistently and accurately segment newly generated images, i.e the thresholding and fragmentation of clamped flies working on for an image fails when new images are analyzed. The Van Andel Institute Lempradl Laboratory developed an image segmentation computer program, FlyFinder, that uses machine learning and computer vision techniques to locate flies (Drosophila Melanogaster) on a plate and compute their distances. FlyFinder streamlines the segmentation starting by training a random forest classifier on labeled fly images and then the classifier is used for thresholding. Next, FlyFinder applies image processing algorithms from computer vision and scikit image packages to segment flies including those that are clamped  A final PDF report generated shows each fly and its x-y a coordinate marked with a point and a the final table lists each fly from each picture with its three nearest neighbors’ distances as well as the median/average distance of that fly with the rest of flies on the same plate. FlyFinder is written in python and R and is equipped with Jupyter Notebooks for easy use.

# Jupyter notebooks for Google Colab
* Download all files from the github repo and upload them on google drive
* Configure the data location, output folders and other parameters
    * Users should adjust the following configuration "fly_finder_dir", "train_dir", "test_dir" and "modelName"
* Execute the jupyter notebooks depending on the data (1 panel or 4 panels)on Google colab environment (the Google colab environment makes it easy for users to install the necessary library installation - click "Install dependencies")
* Inspired by https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb


## For 4 panels jupyter notebook

* Please use this jupyter notebook for 4 panels

https://github.com/VAI-Metabolism/VAI-Metabolism-FlyFinder-JupyterNotebook-colab/blob/main/FlyFinder_4p_colab.ipynb  

* Example - how to use it on Google colab

https://colab.research.google.com/drive/17giUcNKqX7vcTsIr1A2t_yjhTkEqylXI

## For 1 panel on github repo

* Please use this jupyter notebook for 1 panel

https://github.com/VAI-Metabolism/VAI-Metabolism-FlyFinder-JupyterNotebook-colab/blob/main/FlyFinder_1p_colab.ipynb

* Example - how to use it on Google colab

https://colab.research.google.com/drive/1VFuBCOjPrKc_VDnfit-Sm3p5gY2iTZ-c#scrollTo=-71mL3byBmQN
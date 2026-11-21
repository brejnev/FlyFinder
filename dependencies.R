
## FlyFinder_v1.0
## Installing required R packages
## use the command, rscript dependencies.R  or run this script in RStudio


list.of.packages=c("ggplot2", "gridExtra","RColorBrewer","argparse")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]


if(length(new.packages)){
    cat("installing new packages: please wait ...")
    cat(new.packages)
    install.packages(new.packages)
}


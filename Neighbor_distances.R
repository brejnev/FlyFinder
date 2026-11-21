#!/Library/Frameworks/R.framework/Resources/Rscript

suppressPackageStartupMessages({
library(ggplot2)
library(gridExtra)
library(RColorBrewer)
library(argparse)
})


distCol<-function(n=20){
        qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
        col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
        col_vector[1]="green"
        col_vector[4]="red"
        col_vector[6]="black"
        col_vector[3]="blue"
        return(col_vector)
}

euclidDist<-function(x1,y1,x2,y2){
        d=sqrt((x1-x2)**2+(y1-y2)**2)
        return(round(d,2))}


parser <- ArgumentParser(description='Calculation of nearest neigbor distances')

parser$add_argument('--inPath',  help=' the input path of table of x-y coordinates of flies')
parser$add_argument('--imgWidth',  type="integer",help=' the width of the image in pixels')
parser$add_argument('--imgHeight',  type="integer",help=' the height of the image in pixels')

args <- parser$parse_args()

dirPath = args$inPath
if (!dir.exists(dirPath)){
        stop(paste0("'",dirPath,"' ,  the directory does not exit!"))
}


cat("\n\n * ----  Computing nearest neigbor distance  ---- *\n")


lenX = as.integer(args$imgWidth)
lenY = as.integer(args$imgHeight)

cat(paste0("\n Path of x-y coordiate table :  ",dirPath))
cat(paste0("\n Images width  (pxl) :  ",lenX))
cat(paste0("\n Images hieght (pxl) :  ",lenY,"\n"))

if  (Sys.info()["sysname"]=="Windows"){
        tb1<- read.table( file=paste0(dirPath,"//FinalTable.csv"),header=T,sep="\t",stringsAsFactors = F)
}else{
        tb1<- read.table( file=paste0(dirPath,"FinalTable.csv"),header=T,sep="\t",stringsAsFactors = F)
}

distDF=NULL

ggP<-list()
for (iname in unique(tb1$name)){

        Xs=tb1$x_coor[which(tb1$name==iname)]
        Ys=tb1$y_coor[which(tb1$name==iname)]
        flyN=tb1$fly_no[which(tb1$name==iname)]

        #testing images
        m=numeric()
        for (i in 1:length(flyN)){  if (i%%2!=0){m=c(m,-250)}else{m=c(m,150)}} ## where to place the labels

        DF=data.frame(Xs,Ys,flyN=gsub("ly.","",flyN),lbl.X=Xs+m)

        colorz=c("black","red","blue","green","darkcyan","chocolate4","brown3","chartreuse4","darkmagenta","darkslateblue","dodgerblue4",
               "gray30","deeppink","goldenrod4","darkblue","brown4","gray55","darkorange","deepskyblue","gold3","gray35","gray65","gray65","gray80")[1:dim(DF)[1]]

        g<-ggplot(DF, aes(x=Xs, y=Ys)) +
                geom_point(aes(col=colorz),size=1.4, shape=20)+
                scale_color_manual(values=colorz)+
                ggtitle( iname)+
                xlab('x-coorditate (px)')+
                ylab("y-coorditate (px)")+
                xlim(0,lenX)+
                ylim(0,lenY)+
                geom_text(aes(x=lbl.X, y=Ys-10,label=flyN,col=colorz),hjust=0, vjust=0,size=2.5)+
                theme(plot.title=element_text(face="bold",hjust=0.5,size=10),panel.border = element_blank(),panel.background =element_blank(),panel.grid.major = element_line(colour="gray57",size=.1),panel.grid.minor = element_line(colour="gray57",size=.1),axis.line=element_line(colour="black"),plot.margin = unit(c(.8,.9,.8,.9), "cm"),legend.position = "none")
        #g
        ggP[[iname]]<- ggplotGrob(g)

        distM=matrix(NA,nrow=length(flyN),ncol=7)

        for (i in 1:length(Xs)){
                diST=c()
                for (j in 1:length(Xs)){
                        diST=c(diST,euclidDist(Xs[i],Ys[i],Xs[j],Ys[j]) )
                }
                distM[i,]=c( paste0(flyN[i],"-",flyN[order(diST)[2]]),diST[order(diST)[2]],
                             paste0(flyN[i],"-",flyN[order(diST)[3]]), diST[order(diST)[3]],
                             paste0(flyN[i],"-",flyN[order(diST)[4]]), diST[order(diST)[4]],
                             round(mean(diST[2:length(diST)]),2)
                            )
        }
        distDF<-rbind(distDF,distM)
        cat(iname,"\n")
}

## Write coordinate within the CSV
colnames(distDF)=c("ngbr1","dist1","ngbr2","dist2","ngbr3","dist3","av_dist")
tbFinal=cbind(tb1,distDF)
write.table(tbFinal,file=paste0(dirPath,"FinalTable.dist.csv"),row.names=F,sep=",",quote=FALSE)

## Write the combinted figures into pdf file

cat("Generating final pdf of figures")
ggsave(paste0(dirPath,"FinalFigures.dist.pdf"), marrangeGrob(grobs = ggP , nrow=3, ncol=2), height=11, width=9)
ggsave(paste0(dirPath,"FinalFigures.dist.png"), marrangeGrob(grobs = ggP , nrow=3, ncol=2), height=11, width=9)

cat("\n\n * ----  Successfuly completed!  ---- *\n\n")


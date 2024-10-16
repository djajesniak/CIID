library(MVN)
library(pbapply)
library(expm)
library(scales)
library(ggplot2)
library(dplyr)

library(mvtnorm)
#require(expm)


setwd("D:/Praca/Artykuly/FID_plus")


euk_dist_sq=function(x,y) sum((x-y)^2)


rmixed=function(n,p=1/2,m1=0,s1=1,m2=0,s2=1){
  x=rnorm(n,m1,s1)
  y=rnorm(n,m2,s2)
  z=sample(1:0,n,prob=c(p,1-p),replace=TRUE)
  return(z*x+(1-z)*y)
}

#hist(rmixed(1000,m1=-1,m2=10))


W_1=function(X,Y,p=1){
  K=unique(sort(c(X,Y)))
  k=length(K)
  mid_K=K[-k]+diff(K)/2
  F_X=ecdf(X)
  F_Y=ecdf(Y)
#  plot(F_X,xlim=c(min(min(X),min(Y)),max(max(X),max(Y))))
#  lines(F_Y,col="red")  
  
return( sum( diff(K)*( abs(F_X(mid_K)-F_Y(mid_K))^p ) ) )
}

W_1(rmixed(100),rmixed(100))

#################################################

COV=function(M,k=1,l=1){
  c=apply(M,2,mean)
  M_c=t(apply(M,1,function(x) x-c))
  return(t(M_c^k)%*%(M_c^l)/(dim(M)[1]-1))
}

##



FID=function(M,N){
  
  C_M=apply(M,2,mean)
  C_N=apply(N,2,mean)
  
  cov_M=COV(M)
  cov_N=COV(N)
  
  return(sum((C_M-C_N)^2) + 
           sum(diag((cov_M+cov_N-2*sqrtm(cov_M%*%cov_N)))) 
  ) 
}




#########################################
#####################################


########

d=2
n=1000
m=0.95
s=sqrt(1-m^2)

X=matrix(rnorm(d*n),nrow=n)
XX=matrix(rnorm(d*n),nrow=n)
Y=cbind(rmixed(n,m1=m,s1=s,m2=-m,s2=s),rmixed(n,m1=m,s1=s,m2=-m,s2=s))
YY=cbind(rmixed(n,m1=m,s1=s,m2=-m,s2=s),rmixed(n,m1=m,s1=s,m2=-m,s2=s))

#m=0.95
#s=sqrt(1-m^2)
#m^4+6*m^2*s^2+3*s^4


COV(X)
COV(XX)
COV(Y)

COV(X,2,2)
COV(Y,2,2)

plot(X[,1],X[,2],col=alpha(rgb(219/255,242/255,228/255),1),xlab="",ylab="",axes=FALSE,pch=8,cex=1)
plot(X[,1],X[,2],col=rgb(48/255,182/255,101/255),xlab="",ylab="",axes=FALSE,pch=8,cex=1)
points(Y[,1],Y[,2],col=alpha(rgb(178/255,59/255,51/255), 0.3),bg="red",pch=21,cex=1.2)

Figure2=as.data.frame(X)
names(Figure2)=c("X1","X2")
Figure2=data.frame(Figure2,distribution=rep("N",1000))
Figure2_y=as.data.frame(Y)
names(Figure2_y)=c("X1","X2")
Figure2_y=data.frame(Figure2_y,distribution=rep("Mixed",1000))

Figure_2=rbind(Figure2,Figure2_y)

Figure_2 %>% 
  ggplot(aes(X1,X2)) + geom_point(aes(col=distribution))

write.csv(Figure_2,"Figure_2.csv",sep=",",dec=".",row.names = FALSE)


FID(X,XX)

FID(X,Y)

#############################################################################
###############################################################################



######################


rmi=function(n=100,d=2,m1=1,s1=1,m2=-1,s2=1,p=1/2){
  
Y=matrix(0,n,d)
for (k in 1:d) Y[,k]=rmixed(n,m1=m1,s1=s1,m2=m2,s2=s2,p=1/2)

return(Y)
}



#########################
IID=function(X1,X2,Y1,Y2,p=1){
  
  n=nrow(X1)
  
  DIST=matrix(0,n,3)
  for (i in 1:n){
    DIST[i,1]=euk_dist_sq(X1[i,],X2[i,])
    DIST[i,2]=euk_dist_sq(Y1[i,],Y2[i,])
    DIST[i,3]=euk_dist_sq(X1[i,],Y1[i,])
  }
  
  IID=W_1(DIST[,1],DIST[,2],p=p)+W_1(DIST[,1],DIST[,3],p=p)+W_1(DIST[,2],DIST[,3],p=p) 
 return(IID) 
}



#####################################################
ODLEGLOSCI=function(n=100,d=2,m=0.95){


X1=matrix(rnorm(d*n),nrow=n)
X2=matrix(rnorm(d*n),nrow=n)

X3=matrix(rnorm(d*n),nrow=n)
X4=matrix(rnorm(d*n),nrow=n)

s=sqrt(1-m^2)

Y1=rmi(n,d=d,m1=m,s1=s,m2=-m,s2=s)
Y2=rmi(n,d=d,m1=m,s1=s,m2=-m,s2=s)

IID=IID(X1,X2,Y1,Y2)
IID2=IID(X1,X2,Y1,Y2,2)

FID=FID(X1,Y1)

#KID=KID(X1,Y1)

ans1=c(FID,IID,IID2)
names(ans1)=c("FID","IID","IID2")

IID=IID(X1,X2,X3,X4)
IID2=IID(X1,X2,X3,X4,2)

FID=FID(X1,X2)

#KID=KID(X1,X2)

ans2=c(FID,IID,IID2)
names(ans2)=c("FID","IID","IID2")


return(list(ans_rozne=ans1,ans_te_same=ans2))
}

ODLEGLOSCI(10000,d=2,m=0.95)


N=seq(1000,10000,by=50)

nn=length(N)
M_rozne=matrix(0,nn,4)
M_te_same=matrix(0,nn,4)

M_rozne[,1]=N
M_te_same[,1]=N 

colnames(M_rozne)=c("n","FID","IID","IID2")
colnames(M_te_same)=c("n","FID","IID","IID2")

w=1
for( i in N){
  print(paste("n= ",i)) 
  L=ODLEGLOSCI(n=i)
  print(L)
  M_rozne[w,2:4]=L$ans_rozne
  M_te_same[w,2:4]=L$ans_te_same
  w=w+1
  print('###################')
}


par(mfrow=c(1,1))

write.csv(M_rozne,"Figure_3_top.csv",row.names = FALSE)

matplot(M_rozne[,1],M_rozne[,2:4],
        xlab="", 
        ylab="",
        type="l",lty=1,lwd=1,
        col=c("blue","orange","green"),axes=FALSE)
box()
axis(2)

 legend(6900,1,legend=c("FID","CIID-1","CIID-2"),
       horiz=TRUE, lty=1,lwd=2,col=c("blue","orange","green"))

 write.csv(M_te_same,"Figure_3_bottom.csv",row.names = FALSE) 
 
 
matplot(M_te_same[,1],M_te_same[,2:4],axes=FALSE,
        xlab="Sample size", ylab="",
 #       main="Estimated distances between p and p",
        type="l",lty=1,lwd=1,col=c("blue","orange","green"))

box()
axis(2)
axis(1)

legend("topright",legend=c("FID","CIID-1","CIID-2"),
       horiz=TRUE, lty=1,lwd=2,col=c("blue","orange","green"))

legend(7000,-3,legend=c("FID","CIID-1","CIID-2"),
       horiz=TRUE, lty=1,lwd=2,col=c("blue","orange","green"))


#as.expression(bquote(CIID^1))


###################################################

SYMULACJA_p=function(n=1000,p=1/2,d=2){
  
  X1=matrix(rnorm(d*n),nrow=n)
  X2=matrix(rnorm(d*n),nrow=n)
  
  X3=matrix(rnorm(d*n),nrow=n)
  X4=matrix(rnorm(d*n),nrow=n)
  
  
  Y1=rmi(n,d=d,m1=1-p,s1=sqrt(1-(1-p)^2),m2=-p,s2=sqrt(1-p^2),p=p)
  Y2=rmi(n,d=d,m1=1-p,s1=sqrt(1-(1-p)^2),m2=-p,s2=sqrt(1-p^2),p=p)
  
  IID=IID(X1,X2,Y1,Y2)
  IID2=IID(X1,X2,Y1,Y2,2)
  
  FID=FID(X1,Y1)
  
  #KID=KID(X1,Y1)
  
  ans1=c(FID,IID,IID2)
  names(ans1)=c("FID","IID","IID2")
  
  return(ans1)
} 
  
  
SYMULACJA_p(n=10000,p=0.5)
SYMULACJA_p(n=10000,p=0.99)

P=seq(0,1,by=0.01)

MMM=matrix(0,length(P),4)
MMM[,1]=P
w=1
for (p in P){
  MMM[w,2:4]=SYMULACJA_p(n=10000,p=p)
  w=w+1
} 

par(mfrow=c(1,1))
matplot(MMM[,1],MMM[,2:4],
        xlab="p", 
        ylab="",
        #       main="Estimated distances between p and q", 
        type="l",lty=1,lwd=1,col=c("blue","orange","green"),ylim=c(0,2.5))
legend("topleft",legend=c("FID","CIID-1","CIID-2"),
       horiz=TRUE, lty=1,lwd=2,col=c("blue","orange","green"))



###################################################

SYMULACJA_m=function(n=1000,p=1/2,d=2,m=0.95){
  
  X1=matrix(rnorm(d*n),nrow=n)
  X2=matrix(rnorm(d*n),nrow=n)
  
  X3=matrix(rnorm(d*n),nrow=n)
  X4=matrix(rnorm(d*n),nrow=n)
  
  
  Y1=rmi(n,d=d,m1=m,s1=sqrt(1-m^2),m2=-m,s2=sqrt(1-m^2),p=p)
  Y2=rmi(n,d=d,m1=m,s1=sqrt(1-m^2),m2=-m,s2=sqrt(1-m^2),p=p)
  
  IID=IID(X1,X2,Y1,Y2)
  IID2=IID(X1,X2,Y1,Y2,2)
  
  FID=FID(X1,Y1)
  
  #KID=KID(X1,Y1)
  
  ans1=c(FID,IID,IID2)
  names(ans1)=c("FID","IID","IID2")
  
  return(ans1)
} 


SYMULACJA_m(n=10000,m=0.95)
SYMULACJA_m(n=10000,m=0.02)

M=seq(0,0.99,by=0.01)

MMMM=matrix(0,length(M),4)
MMMM[,1]=M
w=1
for (m in M){
  MMMM[w,2:4]=SYMULACJA_m(n=10000,m=m)
  w=w+1
} 

par(mfrow=c(1,1))

write.csv(MMMM,"Figure_4.csv",row.names = FALSE) 

matplot(MMMM[,1],log(MMMM[,2:4]),
        xlab="Parameter m", 
        ylab="",
        axes=FALSE, 
        type="l",lty=1,lwd=1,col=c("blue","orange","green"))
box()
e.y <- (-7):0 ; at.y <- (e.y)
axis(2, at = at.y, col.axis = "black", las = 1,
     labels = as.expression(lapply(e.y, function(E) round(exp(E),3))))
axis(1)

legend(-0.03,1,legend=c("FID","CIID-1","CIID-2"),
       horiz=TRUE, lty=1,lwd=2,col=c("blue","orange","green"))


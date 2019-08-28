
###################
###   Library   ###
###################

library("dplyr")
library(ggplot2)
## library("stringr")
## library(jsonlite)
## library("RJSONIO")



###################
###     ETL     ###ss
###################

list.files()
source("Rcode_ETL_and_Cleaning__20190825.R")



###################
###     EDA     ###
###################


train_transaction_result <- train_transaction[,c("TransactionID","isFraud")]
train <- train_transaction%>% left_join(train_identity)
test <- test_transaction%>% left_join(test_identity)
train$idenTag <- "no_identity";  train$idenTag[train$id_01 < 1] <- "identity"
test$idenTag <- "no_identity";  test$idenTag[test$id_01 < 1] <- "identity"
cbddata <- rbind(cbind(tag="train",train[,!is.element(colnames(train),"isFraud")]),cbind(tag="test",test))
cbddata$isTrain <- cbddata$tag=="train"






## train$isFraud <- NULL



## write.csv(sample,"sample.csv", row.names=FALSE)
dim(sample_submission)
dim(train_transaction_result)
dim(train_identity)
dim(train_transaction)
dim(test_identity)
dim(test_transaction)
dim(train)
dim(test)
dim(cbddata)


sapply(sample_submission, class)
sapply(train_identity, class)
sapply(train_transaction, class)
sapply(test_identity, class)
sapply(test_transaction, class)
sapply(train, class)
sapply(test, class)
sapply(cbddata, class)


head(sample_submission)
head(train_identity)
head(train_transaction)
head(test_identity)
head(test_transaction)
head(train)
head(test)
head(cbddata)
table(train$isFraud)


####  Fraud rate in identity group is almost 4 times that in NO identity group !!!
train_identity_with_isFraud <- subset(train, idenTag == "identity")[, c("isFraud",colnames(train_identity))]
train_no_identity_with_isFraud <- subset(train, idenTag == "no_identity")[, c("isFraud",colnames(train_identity))]
paste("Fraud rate in identity data is ", round(sum(train_identity_with_isFraud$isFraud)/nrow(train_identity_with_isFraud)*100,2),"%",sep="")
paste("Fraud rate in NO identity data is ", round(sum(train_no_identity_with_isFraud$isFraud)/nrow(train_no_identity_with_isFraud)*100,2),"%",sep="")

isFraud <- subset(train, isFraud == 1);  dim(isFraud)
round(table(isFraud$idenTag)/nrow(isFraud)*100,2)

nrow(train_identity_with_isFraud)/nrow(train)
sum(train_identity_with_isFraud$isFraud)
table(train_identity_with_isFraud$isFraud)/nrow(train_identity_with_isFraud)






####  In the subset of data with identity, what is the difference between Fraud/Non-Fraud groups ???
two_group_hist <- function(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield, N0=20, M0=0.3, print_text=c(T,F,"Max","Rate")[1], truncR=0.000){
## cfield <- "id_01"
## cgroup <- "isFraud"
## cdata <- train_identity_with_isFraud
  par(mfrow=c(3,1))
  data0 <- cdata[,c(cgroup,cfield)];  dim(data0)
  ## To truncate a % of possible outliers (assigned by truncR)
  data0 <- subset(data0, data0[[cfield]] <= quantile(data0[[cfield]], 1-truncR, na.rm=T))
  data1 <- subset(data0, data0[[cgroup]]==1);  dim(data1)

  h0 <- hist(data0[[cfield]], plot=F, n=N0)
  h1 <- hist(data1[[cfield]], plot=F, breaks=h0$breaks)
  plot(h1, xlab=cfield, main=paste(cgroup," Group  (sample size=",nrow(data1),", ",round(sum(is.na(data1[[cfield]]))/nrow(data1)*100,2),"% NA)",sep=""))
  plot(h0, xlab=cfield, main=paste("All Data  (sample size=",nrow(data0),", ",round(sum(is.na(data0[[cfield]]))/nrow(data0)*100,2),"% NA)",sep=""))
  num <- h1$counts
  den <- h0$counts;  den[den==0] <- 1
  fraudrate <- round(num/den*100,2)
  cols <- (fraudrate>quantile(fraudrate,(1-M0-0.001)))+1
  cols[fraudrate==max(fraudrate)] <- 6
  xlimit <- range(h0$mids)+c(-1,1)*(max(h0$mids)-min(h0$mids))/(10+length(h0$counts))
  ylimit <- c(0,max(fraudrate*1.2))
  plot(h1$mids, fraudrate, type='l', xlab=cfield, ylab=paste(cgroup,"rate"), xlim=xlimit, ylim=ylimit)
  mtext(paste("Range : (",paste(range(na.omit(data0[[cfield]])), collapse=", "),")",sep=""))
  points(h1$mids, fraudrate, xlab=cfield, ylab="Rate", xlim=xlimit, ylim=ylimit, col=cols, pch=16)
  points(h1$mids[cols>1], fraudrate[cols>1], xlab=cfield, ylab="Fraud rate", xlim=xlimit, ylim=ylimit, col=cols[cols>1], cex=3)
  texts <- round(h0$counts/sum(h0$counts)*100,3)
  cols[texts>10] <- 4

  if ( print_text != F ) {
      if ( print_text == T ) {
          text(h1$mids, fraudrate+max(fraudrate)/6, paste(texts,"%",sep=""), col=cols)
          idx0 <- is.element(cols,c(2,4,6))
      } else if ( print_text == "Rate" ) {
          idx0 <- is.element(cols,c(2,6))
      } else if ( print_text == "Max" ) {
          idx0 <- is.element(cols,c(6))
      }
      text(h1$mids[idx0], (fraudrate-max(fraudrate)/8*ifelse(print_text=="Rate",-1,1))[idx0], paste("rate = ",fraudrate[idx0],"%",sep=""), col=cols[idx0])
  }
  title(main=paste(cgroup," rate in each bin (% of data in each bin & top ",round(M0*100),"% ",cgroup," rates in bins colored red)",sep=""))
  mtext(paste(cfield," column with around ",N0," bins and top ",truncR*100,"% of data are removed as outlier",sep=""), side = 3, line = -1.5, outer = TRUE)
}


pdf("test.pdf", width=16, height=9)
two_group_hist(cdata=cbddata, cgroup="isTrain", cfield="id_01")
two_group_hist(cdata=cbddata, cgroup="isTrain", cfield="TransactionDT")
two_group_hist(cdata=cbddata, cgroup="isTrain", cfield="TransactionAmt")
dev.off()

two_group_hist(cdata=train, cgroup="isFraud", cfield="TransactionAmt", N0=500, M0=0.10, truncR=0.0005, print_text="Max")
two_group_hist(cdata=subset(train,!is.na(DeviceType)), cgroup="isFraud", cfield="TransactionAmt", N0=200, M0=0.10, truncR=0.0005, print_text="Max")



two_group_hist(cdata=train, cgroup="isFraud", cfield="TransactionAmt", N0=20, M0=0.30, print_text=T, truncR=0.00)
two_group_hist(cdata=train, cgroup="isFraud", cfield="TransactionAmt", N0=100, M0=0.30, print_text=T, truncR=0.00005)
two_group_hist(cdata=train, cgroup="isFraud", cfield="TransactionAmt", N0=100, M0=0.15, print_text="Max", truncR=0.00005)
two_group_hist(cdata=train, cgroup="isFraud", cfield="TransactionAmt", N0=1000, M0=0.05, print_text="Max", truncR=0.00005)
two_group_hist(cdata=train, cgroup="isFraud", cfield="TransactionAmt", N0=100, M0=0.15, print_text="Max", truncR=0.05)

two_group_hist(cdata=train, cgroup="isFraud", cfield="TransactionDT", N0=2000, M0=0.01, print_text="Max")
two_group_hist(cdata=train, cgroup="isFraud", cfield="card1", N0=50, M0=0.25)
two_group_hist(cdata=train, cgroup="isFraud", cfield="card2")



two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", N0=100, M0=0.30, cfield="id_04", print_text="Rate")



pdf("train_identity_with_isFraud.pdf", width=9, height=6)
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_01")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_02")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_03")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_04")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_05")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_06")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_07")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_08")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_09")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_10")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_11")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_13")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_14")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_17")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_18")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_19")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_20")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_21")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_22")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_24")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_25")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_26")
two_group_hist(cdata=train_identity_with_isFraud, cgroup="isFraud", cfield="id_32")
dev.off()





























###################
###   Library   ###
###################

library("dplyr")
## library("stringr")
## library(jsonlite)
## library("RJSONIO")




###################
###     ETL     ###
###################



## setwd("C:/Users/momo_hsieh/OneDrive - Default Directory/Documents/Data Science/kaggle/IEEE-Fraud-Detection/R code v1")
cwd <- getwd()
cwd

setwd('../Data'); datawd <- getwd()
files <- list.files()

datapath <- list(  sample_submission="sample_submission.csv"
                 , train_identity= "train_identity.csv"
                 , train_transaction= "train_transaction.csv"
                 , test_identity=  "test_identity.csv"
                 , test_transaction=  "test_transaction.csv")

if( is.element("sample_submission.RData", files) ) {
    load("sample_submission.RData")
    print("sample_submission.RData is loaded")
} else {
    sample_submission <- read.csv(datapath$sample_submission); dim(sample_submission);
##  sample_submission <- read.csv(datapath$sample_submission, colClasses=c('character')); dim(sample_submission);
    save(sample_submission, file="sample_submission.RData")
    print("sample_submission.RData is saved")
}

if( is.element("train_identity.RData", files) ) {
    load("train_identity.RData")
    print("train_identity.RData is loaded")
} else {
    time_train_identity <- system.time({
        train_identity <- read.csv(datapath$train_identity); dim(train_identity);
    ##  train_identity <- read.csv(datapath$train_identity, colClasses=c('character')); dim(train_identity);
    })
    print(time_train_identity)
    save(train_identity, file="train_identity.RData")
    print("train_identity.RData is saved")
}

if( is.element("train_transaction.RData", files) ) {
    load("train_transaction.RData")
    print("train_transaction.RData is loaded")
} else {
    time_train_transaction <- system.time({
        train_transaction <- read.csv(datapath$train_transaction); dim(train_transaction);
    ##  train_transaction <- read.csv(datapath$train_transaction, colClasses=c('character')); dim(train_transaction);
    })
    print(time_train_transaction)
    save(train_transaction, file="train_transaction.RData")
    print("train_transaction.RData is saved")
}

if( is.element("test_identity.RData", files) ) {
    load("test_identity.RData")
    print("test_identity.RData is loaded")
} else {
    time_test_identity <- system.time({
        test_identity <- read.csv(datapath$test_identity); dim(test_identity); 
    ##  test_identity <- read.csv(datapath$test_identity, colClasses=c('character')); dim(test_identity); 
    })
    print(time_test_identity)
    save(test_identity, file="test_identity.RData")
    print("test_identity.RData is saved")
}

if( is.element("test_transaction.RData", files) ) {
    load("test_transaction.RData")
    print("test_transaction.RData is loaded")
} else {
    time_test_transaction <- system.time({
        test_transaction <- read.csv(datapath$test_transaction); dim(test_transaction); 
    ##  test_transaction <- read.csv(datapath$test_transaction, colClasses=c('character')); dim(test_transaction); 
    })
    print(time_test_transaction)
    save(test_transaction, file="test_transaction.RData")
    print("test_transaction.RData is saved")
}

setwd(cwd)






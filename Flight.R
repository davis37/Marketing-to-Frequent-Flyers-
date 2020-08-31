library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(lattice)
library(ggplot2)
library(dummies)

flightdelay.df <- read.csv("FlightDelays.csv")


flightdelay.df<-flightdelay.df[,-c(3, 6, 7, 11, 12)]
View(flightdelay.df)
flightdelay.df$DAY_WEEK <- factor(flightdelay.df$DAY_WEEK )
View(flightdelay.df)

breaks <- c(300,600,900,1200,1300,1500,1800,2100)
tags <- c("300-600","600-900", "900-1200", "1200-1500","1500-1800", "1800-2100","2100-2400")
flightdelay.df$CRS_DEP_TIME <-  cut(flightdelay.df$CRS_DEP_TIME, 
                  breaks = breaks, 
                  include.lowest=TRUE, 
                  right=TRUE, 
                  labels=tags)




flightdelay.df <- dummy.data.frame(flightdelay.df, names= "CRS_DEP_TIME", omit.constants =  FALSE)
View(flightdelay.df)



#Training and validation Partition

numberOfRows <- nrow(flightdelay.df)
set.seed(1)
train.index <- sample(numberOfRows, numberOfRows*0.6)
train.df <- flightdelay.df[train.index, ]
valid.df <- flightdelay.df[-train.index, ]
View(train.df)
View(valid.df)

# create a classification tree
.ct <- rpart(Flight.Status ~., data = train.df, method = "class", maxdepth = 8)

# plot tree
prp(.ct, type = 1, extra = 1, under = FALSE, split.font = 1, varlen = -10)

# classify records in the validation data using the classification tree.
# set argument type = "class" in predict() to generate predicted class membership.
ct.pred <- predict(.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(ct.pred, as.factor(valid.df$Flight.Status))

# build a deeper classification tree
max.ct <- rpart(Flight.Status ~ ., data = train.df, method = "class", cp = 0, minsplit = 50,maxdepth=8)


# count number of leaves
length(max.ct$frame$var[max.ct$frame$var == "<leaf>"])

# plot tree
prp(max.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(max.ct$frame$var == "<leaf>", 'gray', 'white'))  

# classify records in the validation data.
# set argument type = "class" in predict() to generate predicted class membership.
max.pred <- predict(max.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(max.pred, as.factor(valid.df$Flight.Status))
### repeat the code for the validation set, and the deeper tree


# Create code to prune the tree
# xval refers to the number of partitions to use in rpart's built-in cross-validation
# procedure argument.  With xval = 5, bank.df is split into 5 partitions of 1000
# observations each.  A partition is selected at random to hold back for validation 
# while the remaining 4000 observations are used to build each split in the model. 
# Process is repeated for each parition and xerror is calculated as the average error across all partitions.
# complexity paramater (cp) sets the minimum reduction in complexity required for the model to continue.
# minsplit is the minimum number of observations in a node for a split to be attempted.
cv.ct <- rpart(Flight.Status ~ ., data = flightdelay.df, method = "class", 
               control = rpart.control(cp = 0.001, minsplit = 30, xval=5))

# use printcp() to print the table. 
printcp(cv.ct)
prp(cv.ct, type = 1, extra = 1, split.font = 1, varlen = -10)

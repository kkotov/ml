require(ranger)
require(MASS)

covar <- t( matrix(c(1.00,0.99,0.98,
                     0.99,1.00,0.99,
                     0.98,0.99,1.00),ncol=3) )

covar <- t( matrix(c(1.00,0.00,0.00,
                     0.00,1.00,0.00,
                     0.00,0.00,1.00),ncol=3) )

covar <- t( matrix(c(1.00,0.90,0.70,
                     0.90,1.00,0.90,
                     0.70,0.90,1.00),ncol=3) )


df <- data.frame( mvrnorm( 10000, c(0,0,0), covar ) )
colnames(df) <- c("V1","V2","V3")

df$V3 <- factor( ifelse( df$V3>=0, rep(1,length(df$V3)), rep(-1,length(df$V3)) ) )

plot(df[sample(nrow(df),10000),c("V1","V3")], xlab="x", ylab="y", pch=1)
write.csv(file="two.csv",x=df) #[sample(df(xyz),10000),])

trainSet <- df[seq(1,nrow(df),2),]
testSet  <- df[seq(2,nrow(df),2),]

predictors <- c("V1", "V2")

f <- as.formula(paste("V3 ~ ", paste(predictors, collapse= "+")))

#df <- read.csv(file="two.csv", header=T, sep=",")

modelFit <- ranger(f, data=trainSet, importance="impurity")

#mean(testSet$V1 - predict(modelFit,testSet)$prediction)

table(testSet$V3,  predict(modelFit,testSet)$prediction)

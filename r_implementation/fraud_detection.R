# Note: Install packages if not already installed
# install.packages(c("caret", "randomForest", "smotefamily", "data.table"))

library(caret)
library(smotefamily)
library(randomForest)
library(data.table)

cat("Loading data...\n")
df <- fread("../data/creditcard.csv")

# Scale Time and Amount
df$Scaled_Amount <- scale(df$Amount)
df$Scaled_Time <- scale(df$Time)
df[, c("Time", "Amount") := NULL]

# Convert Class to factor
df$Class <- as.factor(df$Class)

cat("Splitting data...\n")
set.seed(42)
trainIndex <- createDataPartition(df$Class, p = .8, list = FALSE, times = 1)
dfTrain <- df[ trainIndex,]
dfTest  <- df[-trainIndex,]

cat("Applying SMOTE...\n")
# SMOTE requires numeric inputs, and factor target
smote_result <- smotefamily::SMOTE(dfTrain[, -which(names(dfTrain) == "Class"), with=FALSE], 
                                   as.numeric(as.character(dfTrain$Class)))
dfTrain_smote <- smote_result$data
names(dfTrain_smote)[ncol(dfTrain_smote)] <- "Class"
dfTrain_smote$Class <- as.factor(dfTrain_smote$Class)

cat("Training Random Forest model...\n")
# Using a small ntree and subset for demonstration to run quickly
rf_model <- randomForest(Class ~ ., data=dfTrain_smote, ntree=50, sampsize=5000)

cat("Evaluating Model...\n")
predictions <- predict(rf_model, dfTest)
cm <- confusionMatrix(predictions, dfTest$Class)
print(cm)

# Save model
dir.create("../models", showWarnings = FALSE)
saveRDS(rf_model, "../models/fraud_model_rf.rds")
cat("Model saved to models/fraud_model_rf.rds\n")

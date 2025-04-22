library(tidyverse)
library(caret)
library(randomForest)
library(class)        # KNN
library(cluster)      # Clustering
library(dendextend)   # Hierarchical Clustering

library(corrplot)





# Load data
data <- read.csv(file.choose())  # Replace with your actual path
View(data)

# 1. Check class distribution
cat("Class distribution:\n")
print(table(data$num))
print(prop.table(table(data$num)))

# 2. Check for duplicate rows
cat("\nDuplicate rows:\n")
print(sum(duplicated(data)))

# 3. Correlation check (exclude target and non-numeric columns)
cat("\nCorrelation matrix:\n")
numeric_data <- data %>% select(where(is.numeric)) %>% select(-num)
cor_matrix <- cor(numeric_data, use = "complete.obs")
corrplot(cor_matrix, method = "color", tl.cex = 0.8, number.cex = 0.7)

# 4. Check target correlation with other variables
cat("\nCorrelation with target:\n")
cor_target <- cor(data %>% select(where(is.numeric)), use = "complete.obs")
print(cor_target["num", ])

# 5. Check test set size
set.seed(123)
trainIndex <- createDataPartition(data$num, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]
cat("\nTrain size:", nrow(train), "\nTest size:", nrow(test), "\n")

# Convert num to factor if classification
train$num <- as.factor(train$num)
test$num <- as.factor(test$num)

# Train the Random Forest model
rf_model <- randomForest(num ~ ., data = train)

# Predict on test set
rf_pred <- predict(rf_model, test)

# Ensure predicted values and actual values are factors with the same levels
rf_pred <- factor(rf_pred, levels = levels(test$num))
test$num <- factor(test$num, levels = levels(rf_pred))

# Confusion matrix
confusion <- confusionMatrix(rf_pred, test$num)


cat("\nPrecision (Positive Class):", confusion$byClass["Precision"])
cat("\nRecall (Positive Class):", confusion$byClass["Recall"])
cat("\nF1-Score (Positive Class):", confusion$byClass["F1"])

knn_pred <- knn(train[ , -which(names(train) == "num")],
                test[ , -which(names(test) == "num")],
                cl = train$num, k = 5)
confusionMatrix(knn_pred, test$num)

kmeans_model <- kmeans(train[ , -which(names(train) == "num")], centers = 2)
table(kmeans_model$cluster, train$num)


dist_mat <- dist(train[ , -which(names(train) == "num")])
hc <- hclust(dist_mat)
plot(as.dendrogram(hc), main = "Hierarchical Clustering")
hc_clusters <- cutree(hc, k = 2)
table(hc_clusters, train$num)



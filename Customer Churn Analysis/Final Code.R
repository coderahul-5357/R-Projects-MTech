# Load necessary libraries
library(dplyr)
library(ggplot2)
library(caret)
library(cluster)
library(arules)
library(corrplot)
library(lattice)
library(Matrix)
# Load dataset
data <- read.csv("churn data.csv")
head(data)

sum(is.na(data))  # Count total missing values in the dataset

colSums(is.na(data))  # Show missing values per column
data$TotalCharges[is.na(data$TotalCharges)] <- median(data$TotalCharges, na.rm = TRUE)
colSums(is.na(data))  # Show missing values per column

sum(is.na(data))  # Check again to confirm all missing values are handled

ggplot(data, aes(x = Churn, fill = Churn)) +
  geom_bar() +
  ggtitle("Churn Distribution") +
  theme_minimal()


ggplot(data, aes(x = gender, fill = gender)) +
  geom_bar() +
  ggtitle("Gender Distribution") +
  theme_minimal()


ggplot(data, aes(x = tenure)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black", alpha = 0.7) +
  ggtitle("Distribution of Customer Tenure") +
  theme_minimal()


ggplot(data, aes(x = MonthlyCharges, y = TotalCharges, color = Churn)) +
  geom_point(alpha = 0.6) +
  ggtitle("Monthly Charges vs. Total Charges") +
  theme_minimal()


library(caret)

# Convert Churn to a factor
data$Churn <- as.factor(data$Churn)

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$Churn, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Train a logistic regression model
model <- glm(Churn ~ MonthlyCharges + tenure + TotalCharges + Contract, 
             data = trainData, family = "binomial")

# Summary of the model
summary(model)

# Make predictions
pred <- predict(model, testData, type = "response")
testData$Predicted <- ifelse(pred > 0.5, "Yes", "No")

# Confusion Matrix
confusionMatrix(factor(testData$Predicted), testData$Churn)


library(rpart)
library(rpart.plot)

# Train a decision tree model
tree_model <- rpart(Churn ~ MonthlyCharges + tenure + TotalCharges + Contract, 
                    data = trainData, method = "class")

# Plot the decision tree
rpart.plot(tree_model, extra = 104)


library(randomForest)

# Train the model
rf_model <- randomForest(Churn ~ ., data = trainData, ntree = 100, mtry = 3, importance = TRUE)

# Feature importance
importance(rf_model)

# Make predictions
rf_pred <- predict(rf_model, testData)

# Confusion Matrix
confusionMatrix(rf_pred, testData$Churn)


set.seed(123)
cluster_data <- data %>% select(tenure, MonthlyCharges)
kmeans_result <- kmeans(cluster_data, centers = 3)

# Add cluster labels
data$Cluster <- as.factor(kmeans_result$cluster)

# Visualize clusters
ggplot(data, aes(x = tenure, y = MonthlyCharges, color = Cluster)) +
  geom_point() +
  ggtitle("Customer Segmentation using K-Means")


library(arules)

# Convert categorical variables to factors
data_fact <- data %>% select(gender, Contract, PaymentMethod, Churn)
data_fact[] <- lapply(data_fact, as.factor)

# Convert to transactions
trans <- as(data_fact, "transactions")

# Apply Apriori algorithm
rules <- apriori(trans, parameter = list(supp = 0.1, conf = 0.6))

# Inspect top rules
inspect(head(sort(rules, by = "confidence"), 5))



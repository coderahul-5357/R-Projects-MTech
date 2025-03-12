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
data <- read.csv("loan_approval_dataset.csv")
head(data)

# Check for missing values in each column
colSums(is.na(data))

# Summary of dataset
summary(data)

# Structure of dataset
str(data)

# Convert categorical variables to factors
data$education <- as.factor(data$education)
data$self_employed <- as.factor(data$self_employed)
data$loan_status <- as.factor(data$loan_status)


# Replace missing values with median (for numerical) or mode (for categorical)
data$loan_amount[is.na(data$loan_amount)] <- median(data$loan_amount, na.rm = TRUE)

ggplot(data, aes(x = loan_status, fill = loan_status)) +
  geom_bar() +
  labs(title = "Loan Approval Distribution", x = "Loan Status", y = "Count") +
  theme_minimal()


ggplot(data, aes(x = income_annum, y = loan_amount, color = loan_status)) +
  geom_point() +
  labs(title = "Income vs Loan Amount", x = "Annual Income", y = "Loan Amount") +
  theme_minimal()


# Convert categorical columns to numeric before correlation
numeric_data <- data %>%
  select_if(is.numeric)

# Plot correlation matrix
corrplot(cor(numeric_data, use = "complete.obs"), method = "color", tl.cex = 0.7)

set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(data$loan_status, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
# Logistic Regression Model
model <- train(loan_status ~ ., data = trainData, method = "glm", family = "binomial")

# Model summary
summary(model)

# Select numeric columns for clustering
clustering_data <- data %>% select(income_annum, loan_amount, cibil_score)

# Standardize the data
clustering_data <- scale(clustering_data)

# Apply K-means clustering (Choose 3 clusters)
set.seed(123)
kmeans_result <- kmeans(clustering_data, centers = 3, nstart = 10)

# Add cluster labels to original data
data$Cluster <- as.factor(kmeans_result$cluster)

# Visualize Clusters
ggplot(data, aes(x = income_annum, y = loan_amount, color = Cluster)) +
  geom_point() +
  labs(title = "Customer Segmentation using K-Means Clustering") +
  theme_minimal()


# Convert categorical variables into factor
data$loan_status <- as.factor(data$loan_status)
data$education <- as.factor(data$education)
data$self_employed <- as.factor(data$self_employed)

# Convert dataframe to transaction format
trans_data <- as(data, "transactions")

# Generate association rules using Apriori algorithm
rules <- apriori(trans_data, parameter = list(supp = 0.1, conf = 0.8))

# Inspect top rules
inspect(rules[1:5])

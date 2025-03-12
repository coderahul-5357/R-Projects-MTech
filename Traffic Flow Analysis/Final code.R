# Load required libraries
library(dplyr)      # Data manipulation
library(ggplot2)    # Data visualization
library(caret)      # Machine learning & preprocessing
library(cluster)    # Clustering
library(arules)     # Association rule mining
library(corrplot)   # Correlation plot
library(lattice)    # Visualization
library(Matrix)     # Matrix operations


# Load dataset
data <- read.csv("traffic_flow_prediction_dataset.csv", stringsAsFactors = FALSE)

# Display first few rows
head(data)

# Check structure of dataset
str(data)

# Summary statistics
summary(data)


# Check for missing values
colSums(is.na(data))

# Fill missing values (if any)
data <- na.omit(data)  # Removes rows with any NA values



# Fill missing values with median for numerical columns
data$temperature_celsius[is.na(data$temperature_celsius)] <- median(data$temperature_celsius, na.rm = TRUE)
data$traffic_volume[is.na(data$traffic_volume)] <- median(data$traffic_volume, na.rm = TRUE)

# Fill missing values for categorical columns
data$weather_conditions[is.na(data$weather_conditions)] <- "Unknown"




# Convert categorical variables to factors
data$day_of_week <- as.factor(data$day_of_week)
data$weather_conditions <- as.factor(data$weather_conditions)

# Create a time category (Morning, Afternoon, Evening, Night)
data$time_category <- cut(
  data$time_of_day, 
  breaks = c(-1, 6, 12, 18, 24), 
  labels = c("Night", "Morning", "Afternoon", "Evening")
)

# Create traffic volume categories (Low, Medium, High)
data$traffic_category <- cut(
  data$traffic_volume, 
  breaks = quantile(data$traffic_volume, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE), 
  labels = c("Low", "Medium", "High"),
  include.lowest = TRUE
)




ggplot(data, aes(x = traffic_volume)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "black") +
  labs(title = "Traffic Volume Distribution", x = "Traffic Volume", y = "Count")


ggplot(data, aes(x = time_category, y = traffic_volume, fill = time_category)) +
  geom_boxplot() +
  labs(title = "Traffic Volume by Time of Day", x = "Time of Day", y = "Traffic Volume")



ggplot(data, aes(x = weather_conditions, y = traffic_volume, fill = weather_conditions)) +
  geom_boxplot() +
  labs(title = "Traffic Volume by Weather Condition", x = "Weather Condition", y = "Traffic Volume") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))





# Select numeric columns
numeric_data <- data %>%
  select_if(is.numeric)

# Compute correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")

# Display correlation matrix
print(cor_matrix)
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)



# Convert categorical target variable to factor
data$traffic_category <- as.factor(data$traffic_category)

# Split data into training (80%) and testing (20%)
set.seed(123)  # Ensures reproducibility
train_index <- createDataPartition(data$traffic_category, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]





# Load library
library(rpart)
library(rpart.plot)

# Train Decision Tree
model <- rpart(traffic_category ~ time_of_day + day_of_week + weather_conditions + temperature_celsius,
               data = train_data, method = "class")

# Plot the tree
rpart.plot(model)


# Make predictions
predictions <- predict(model, test_data, type = "class")

# Compute confusion matrix
confusionMatrix(predictions, test_data$traffic_category)



# Select numeric features for clustering
clustering_data <- scale(numeric_data)

# Choose optimal k using the Elbow Method
wss <- sapply(1:10, function(k){
  kmeans(clustering_data, centers = k, nstart = 10)$tot.withinss
})

# Plot Elbow Curve
plot(1:10, wss, type = "b", pch = 19, col = "red", xlab = "Number of Clusters", ylab = "WSS")



set.seed(123)
km_model <- kmeans(clustering_data, centers = 3, nstart = 10)

# Add cluster labels to dataset
data$Cluster <- as.factor(km_model$cluster)

# View cluster distribution
table(data$Cluster)



ggplot(data, aes(x = time_of_day, y = traffic_volume, color = Cluster)) +
  geom_point(alpha = 0.5) +
  labs(title = "Traffic Clusters", x = "Time of Day", y = "Traffic Volume")

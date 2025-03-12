# Load required libraries
library(ggplot2)       # Data visualization
library(corrplot)      # Correlation matrix visualization
library(caret)         # Machine learning (classification, regression)
library(randomForest)  # Random Forest model
library(e1071)         # Support Vector Machine
library(arules)        # Association Rule Mining
library(arulesViz)     # Visualization for Association Rules
library(Matrix)

# Load the dataset (update the file path if needed)
data <- read.csv("Data.csv", header = TRUE)

# View the first few rows
head(data)


# Check for missing values
colSums(is.na(data))

# Convert categorical columns to factors
data$type <- as.factor(data$type)
data$rating <- as.factor(data$rating)

# Convert release_year to numeric
data$release_year <- as.numeric(data$release_year)

# Extract duration in numeric form (movies in minutes, TV shows in seasons)
data$duration_num <- as.numeric(gsub(" min| Season| Seasons", "", data$duration))

# Handling missing values (remove rows or impute)
data <- na.omit(data)


library(ggplot2)

ggplot(data, aes(x = type, fill = type)) +
  geom_bar() +
  labs(title = "Distribution of Movies vs. TV Shows")



library(dplyr)
library(tidyr)

# Count the frequency of each genre
genre_count <- data %>%
  separate_rows(listed_in, sep = ", ") %>%
  count(listed_in, sort = TRUE)

# Bar plot for top 10 genres
ggplot(genre_count[1:10, ], aes(x = reorder(listed_in, -n), y = n, fill = listed_in)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top 10 Genres", x = "Genre", y = "Count")

ggplot(data, aes(x = release_year, fill = type)) +
  geom_histogram(binwidth = 5, position = "stack") +
  labs(title = "Content Release Trend Over the Years")



library(caret)
library(randomForest)

# Prepare data (select useful features)
df <- data[, c("release_year", "duration_num", "type")]
df <- na.omit(df)

# Split into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(df$type, p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]

# Train a Random Forest model
model <- randomForest(type ~ ., data = trainData, ntree = 100)

# Predict on test set
predictions <- predict(model, newdata = testData)

# Evaluate accuracy
confusionMatrix(predictions, testData$type)





library(cluster)

# Prepare data for clustering
clustering_data <- na.omit(data[, c("release_year", "duration_num")])

# Determine optimal number of clusters (Elbow Method)
wss <- (nrow(clustering_data)-1)*sum(apply(clustering_data, 2, var))
for (i in 2:10) {
  wss[i] <- sum(kmeans(clustering_data, centers = i)$withinss)
}
plot(1:10, wss, type="b", main="Elbow Method", xlab="Number of Clusters", ylab="WSS")

# Apply K-Means
set.seed(123)
kmeans_model <- kmeans(clustering_data, centers = 3)
clustering_data$cluster <- as.factor(kmeans_model$cluster)

# Visualize clusters
ggplot(clustering_data, aes(x = release_year, y = duration_num, color = cluster)) +
  geom_point() +
  labs(title = "Clustering of TV Shows & Movies")




install.packages("syuzhet")  # Install sentiment analysis package
library(syuzhet)



# Extract descriptions
descriptions <- data$description

# Compute sentiment scores
sentiment_scores <- get_sentiment(descriptions, method = "bing")  # Using Bing Lexicon

# Add sentiment scores to the dataset
data$sentiment_score <- sentiment_scores

# Categorize sentiment into Positive, Negative, or Neutral
data$sentiment <- ifelse(sentiment_scores > 0, "Positive",
                         ifelse(sentiment_scores < 0, "Negative", "Neutral"))

# View Sentiment Distribution
table(data$sentiment)

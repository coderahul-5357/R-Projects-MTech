
library(tm)            # Text mining
library(SnowballC)     # Stemming
library(wordcloud)     # Word cloud visualization
library(ggplot2)       # Data visualization
library(dplyr)         # Data manipulation
library(caret)         # Machine learning models
library(e1071)         # SVM & Naïve Bayes
library(randomForest)  # Random Forest
library(text2vec)      # Text vectorization
library(RColorBrewer)  # Colors for visualization
# Load dataset
df <- read.csv("sentiment.csv", stringsAsFactors = FALSE)

# View first few rows
head(df)
# Check structure of dataset
str(df)

# Check missing values
sum(is.na(df))

# Check unique labels
table(df$labels)

# View a sample of tweets
df$tweets[1:10]






# Load required libraries
library(tm)
library(SnowballC)

# Remove the index column
df <- df[, -1]

# Function for text cleaning
clean_text <- function(text) {
  text <- tolower(text)  # Convert to lowercase
  text <- gsub("http\\S+|www\\S+", "", text)  # Remove URLs
  text <- gsub("[^a-z ]", "", text)  # Remove special characters & numbers
  text <- removeWords(text, stopwords("en"))  # Remove stopwords
  text <- wordStem(text)  # Perform stemming
  return(text)
}

# Apply cleaning function to all tweets
df$tweets <- sapply(df$tweets, clean_text)

# View cleaned tweets
head(df$tweets, 10)









# Load library for text vectorization
library(tm)

# Create a Corpus
corpus <- VCorpus(VectorSource(df$tweets))

# Create a Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus,
                          control = list(weighting = weightTfIdf, 
                                         removePunctuation = TRUE,
                                         stopwords = TRUE,
                                         stemming = TRUE))

# Convert DTM to matrix format
dtm_matrix <- as.matrix(dtm)

# View dimensions (rows = tweets, cols = words)
dim(dtm_matrix)

# View a sample of TF-IDF features
dtm_matrix[1:5, 1:5]









# Load required library
library(caret)

# Set a seed for reproducibility
set.seed(123)

# Convert labels to factors (required for classification)
df$labels <- as.factor(df$labels)

# Split data into training (80%) and testing (20%)
trainIndex <- createDataPartition(df$labels, p = 0.8, list = FALSE)
train_data <- dtm_matrix[trainIndex, ]
test_data <- dtm_matrix[-trainIndex, ]
train_labels <- df$labels[trainIndex]
test_labels <- df$labels[-trainIndex]

# Check sizes
dim(train_data)  # Should be around 800 x 2939
dim(test_data)   # Should be around 200 x 2939















# Load Naïve Bayes library
library(e1071)

# Train the Naïve Bayes model
nb_model <- naiveBayes(train_data, train_labels)

# Make predictions
nb_predictions <- predict(nb_model, test_data)

# Evaluate the model
conf_matrix <- confusionMatrix(nb_predictions, test_labels)
print(conf_matrix)



# Load SVM library
library(e1071)

# Train the SVM model
svm_model <- svm(train_data, train_labels, kernel = "linear")

# Make predictions
svm_predictions <- predict(svm_model, test_data)

# Evaluate the model
svm_conf_matrix <- confusionMatrix(svm_predictions, test_labels)
print(svm_conf_matrix)




# Load Random Forest library
library(randomForest)

# Train the Random Forest model
rf_model <- randomForest(train_data, train_labels, ntree = 100)

# Make predictions
rf_predictions <- predict(rf_model, test_data)

# Evaluate the model
rf_conf_matrix <- confusionMatrix(rf_predictions, test_labels)
print(rf_conf_matrix)









# Function to extract accuracy, precision, recall, and F1-score
get_metrics <- function(conf_matrix) {
  accuracy <- conf_matrix$overall["Accuracy"]
  precision <- conf_matrix$byClass["Precision"]
  recall <- conf_matrix$byClass["Recall"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  return(c(accuracy, precision, recall, f1))
}

# Extract metrics for each model
nb_metrics <- get_metrics(conf_matrix)
svm_metrics <- get_metrics(svm_conf_matrix)
rf_metrics <- get_metrics(rf_conf_matrix)

# Combine results into a dataframe
results <- data.frame(
  Model = c("Naïve Bayes", "SVM", "Random Forest"),
  Accuracy = c(nb_metrics[1], svm_metrics[1], rf_metrics[1]),
  Precision = c(nb_metrics[2], svm_metrics[2], rf_metrics[2]),
  Recall = c(nb_metrics[3], svm_metrics[3], rf_metrics[3]),
  F1_Score = c(nb_metrics[4], svm_metrics[4], rf_metrics[4])
)

# Print the results
print(results)



# Step 1: Load required libraries
library(tidyverse)
library(tm)               # Text Mining package
library(caret)            # Machine Learning utilities
library(randomForest)     # Random Forest Classifier
library(e1071)            # SVM & Logistic Regression
library(ggplot2)         # Visualization
library(wordcloud)       # Word Cloud for Text Visualization
library(readxl)

# Step 2: Load dataset
df <- read.csv("sentiment.csv", stringsAsFactors = FALSE)


# Step 3: Drop unnecessary columns
df <- df[, c("tweets", "labels")]


# Step 3: Explore dataset
head(df)
str(df)
table(df$labels)  # Distribution of sentiment classes

# Step 4: Text Preprocessing Function
clean_text <- function(text) {
  text <- tolower(text)  # Convert to lowercase
  text <- gsub("http\\S+|www\\S+", "", text)  # Remove URLs
  text <- gsub("@\\w+|#", "", text)  # Remove mentions and hashtags
  text <- gsub("[^a-z\\s\\n]", "", text)  # Remove special characters
  return(text)
}

df$tweets <- sapply(df$tweets, clean_text)

head(df$tweets)  # Show first few cleaned tweets


# Step 5: Encode Labels (Good → 2, Neutral → 1, Bad → 0)
df$labels <- factor(df$labels, levels = c("bad", "neutral", "good"), labels = c(0, 1, 2))
head(df$labels)

# Step 6: Train-Test Split
library(caret)
set.seed(42)
trainIndex <- createDataPartition(df$labels, p = 0.8, list = FALSE)
train_data <- df[trainIndex, ]
test_data  <- df[-trainIndex, ]

# Step 7: TF-IDF Vectorization
library(tm)
corpus <- VCorpus(VectorSource(train_data$tweets))

corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)



# Convert corpus to document-term matrix
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.99)
train_matrix <- as.matrix(dtm)

# Apply the same transformation to the test set
test_corpus <- VCorpus(VectorSource(test_data$tweets))
test_corpus <- tm_map(test_corpus, content_transformer(tolower))
test_corpus <- tm_map(test_corpus, removeNumbers)
test_corpus <- tm_map(test_corpus, removePunctuation)
test_corpus <- tm_map(test_corpus, removeWords, stopwords("en"))
test_corpus <- tm_map(test_corpus, stripWhitespace)

test_dtm <- DocumentTermMatrix(test_corpus, control = list(dictionary = Terms(dtm)))
test_matrix <- as.matrix(test_dtm)

# Step 8: Train Logistic Regression Model
logistic_model <- glm(labels ~ ., data = data.frame(train_matrix, labelled = train_data$labels), 
                      family = binomial)

# Step 9: Train Random Forest Model
rf_model <- randomForest(x = train_matrix, y = train_data$labels, ntree = 100)

# Step 10: Make Predictions
logistic_preds <- predict(logistic_model, newdata = data.frame(test_matrix), type = "response")
logistic_preds <- as.factor(ifelse(logistic_preds > 0.5, 1, 0))

rf_preds <- predict(rf_model, newdata = test_matrix)

# Step 11: Evaluate Models
logistic_acc <- sum(logistic_preds == test_data$labels) / length(test_data$labels)
rf_acc <- sum(rf_preds == test_data$labels) / length(test_data$labels)

cat("Logistic Regression Accuracy:", logistic_acc, "\n")
cat("Random Forest Accuracy:", rf_acc, "\n")

# Step 12: Confusion Matrix
logistic_cm <- confusionMatrix(logistic_preds, test_data$labels)
rf_cm <- confusionMatrix(rf_preds, test_data$labels)

print("Logistic Regression Confusion Matrix:")
print(logistic_cm)

print("Random Forest Confusion Matrix:")
print(rf_cm)

# Step 13: Visualizations

# Plot Sentiment Distribution
ggplot(df, aes(x = label)) +
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Sentiment Distribution", x = "Sentiment", y = "Count")

# Confusion Matrix Heatmap (Random Forest)
rf_cm_table <- as.table(rf_cm$table)
ggplot(data = as.data.frame(rf_cm_table), aes(Prediction, Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal() +
  labs(title = "Random Forest Confusion Matrix", x = "Predicted", y = "Actual")

# Word Cloud
wordcloud(words = names(colSums(as.matrix(dtm))), 
          freq = colSums(as.matrix(dtm)), 
          max.words = 100, 
          random.order = FALSE, 
          colors = brewer.pal(8, "Dark2"))




# Load required libraries
library(ggplot2)       # Data visualization
library(corrplot)      # Correlation matrix visualization
library(caret)         # Machine learning (classification, regression)
library(randomForest)  # Random Forest model
library(e1071)         # Support Vector Machine
library(arules)        # Association Rule Mining
library(arulesViz)     # Visualization for Association Rules
library(Matrix)




# Load dataset as a single column
df <- read.delim("SMSSpamCollection.csv", header=FALSE, stringsAsFactors=FALSE)

# Check the structure
str(df)

# Split the first column into two: Label and Message
df <- data.frame(do.call("rbind", strsplit(df$V1, "\t", fixed=TRUE)))

# Rename columns
colnames(df) <- c("Label", "Message")

# Convert Label to factor for classification
df$Label <- factor(df$Label)

# Check data structure again
str(df)
head(df)



table(df$Label)
prop.table(table(df$Label))  # Get percentage distribution



ggplot(df, aes(x = Label, fill = Label)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Spam vs Ham Distribution", x = "Category", y = "Count")



library(tm)

# Convert to a proper corpus
corpus <- VCorpus(VectorSource(df$Message))

# Clean the text
corpus <- tm_map(corpus, content_transformer(tolower))  # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)  # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)  # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("en"))  # Remove stopwords
corpus <- tm_map(corpus, stripWhitespace)  # Remove extra spaces

# Convert to Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.99)  # Remove sparse terms

# Convert to data frame
df_clean <- as.data.frame(as.matrix(dtm))
df_clean$Label <- df$Label






library(caret)
library(e1071)

set.seed(123)
trainIndex <- createDataPartition(df_clean$Label, p=0.8, list=FALSE)
trainData <- df_clean[trainIndex, ]
testData <- df_clean[-trainIndex, ]

# Train Naïve Bayes
model <- naiveBayes(Label ~ ., data=trainData)
predictions <- predict(model, testData)

# Evaluate model
confusionMatrix(predictions, testData$Label)






library(e1071)

# Train SVM model
model_svm <- svm(Label ~ ., data=trainData, kernel="linear", cost=1)
predictions_svm <- predict(model_svm, testData)

# Evaluate model
confusionMatrix(predictions_svm, testData$Label)






# Natural Language Processing

# Importing the dataset
dataset_original = read.delim(file.choose(), quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
 install.packages('tm')
 install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm
View(dtm)
dtm=removeSparseTerms(dtm,0.999)
dtm
View(dtm)
dataset = as.data.frame(as.matrix(dtm))
dataset
View(dataset)
dataset$Liked = dataset_original$Liked


# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))
dataset$Liked


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)
classifier
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])
y_pred
# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
View(cm)

# Fake-News-Detection-using-Machine-Learning
 

Fake news on different platforms is spreading widely and is a matter of serious concern, as it causes social wars and permanent breakage of the bonds established among people. A lot of research is already going on focused on the classification of fake news. In this project, we aim to solve this issue using machine learning in Python.

## Table of Contents

- [Introduction](#introduction)
- [Importing Libraries and Datasets](#importing-libraries-and-datasets)
- [Data Preprocessing](#data-preprocessing)
- [Preprocessing and Analysis of News Column](#preprocessing-and-analysis-of-news-column)
- [Converting Text into Vectors](#converting-text-into-vectors)
- [Model Training, Evaluation, and Prediction](#model-training-evaluation-and-prediction)
- [Conclusion](#conclusion)

## Introduction

Fake news has become a widespread issue on various platforms, causing social conflicts and damaging relationships. In this project, we employ machine learning techniques in Python to classify news articles as fake or real.

## Importing Libraries and Datasets

We start by importing the necessary libraries and the dataset. We use Pandas for data handling, Seaborn/Matplotlib for data visualization, and various machine learning libraries.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

Let's import the dataset to get started:

```python
data = pd.read_csv('News.csv', index_col=0)
```

## Data Preprocessing

Data preprocessing involves cleaning and preparing the dataset for analysis. We remove unnecessary columns, check for missing values, shuffle the data to prevent bias, and perform other necessary data cleaning operations.

```python
# Removing unnecessary columns
data = data.drop(["title", "subject", "date"], axis=1)

# Check for missing values
data.isnull().sum()

# Shuffle the dataset and reset index
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)
```

## Preprocessing and Analysis of News Column

We preprocess the text data by removing stopwords, punctuation, and irrelevant spaces from the text. After this, we analyze the text data.

```python
# Preprocess text data
preprocessed_review = preprocess_text(data['text'].values)
data['text'] = preprocessed_review

# Visualize WordCloud for fake and real news
# Plot the top frequent words
```

## Converting Text into Vectors

Before feeding data into a machine learning model, we split the dataset into training and testing sets. Then, we convert the text data into numerical vectors using techniques like TF-IDF vectorization.

```python
# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25)

# Convert text data into vectors using TF-IDF vectorization
```

## Model Training, Evaluation, and Prediction

We train machine learning models, such as Logistic Regression or Decision Tree Classifier, on the vectorized text data. We evaluate the model's performance using metrics like accuracy and confusion matrices.

```python
# Train a machine learning model (e.g., Logistic Regression)
model.fit(x_train, y_train)

# Evaluate the model
train_accuracy = accuracy_score(y_train, model.predict(x_train))
test_accuracy = accuracy_score(y_test, model.predict(x_test))

# Print accuracy scores
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Plot confusion matrix (e.g., Decision Tree Classifier)
```

## Conclusion

In this project, we have implemented a machine learning solution for fake news detection. We've shown how to preprocess text data, convert it into numerical vectors, and train and evaluate machine learning models. Decision Tree Classifier and Logistic Regression are among the models used in this project, and their performance has been assessed.

Feel free to explore and adapt this project for your own use case or further research into fake news detection.

> Disclaimer: The accuracy and performance of the models may vary, and further improvements can be made to enhance the accuracy of fake news detection.

**Note:** Make sure to update this README with relevant results, insights, and additional details about your project as needed.

  

# SENTIMENT-ANALYSIS
Project Overview: Sentiment Analysis of Twitter Data Using NLP and Machine Learning
Introduction
With the rise of social media, platforms like Twitter have become key sources of public opinion on current events, brands, and global issues. This project focuses on building a sentiment analysis system that classifies tweets into positive or negative categories. Using natural language processing (NLP) techniques and machine learning algorithms, the system is designed to understand and interpret emotional tone in user-generated text.

Objectives
The primary goals of this sentiment analysis project are:

To clean and preprocess real-world Twitter data effectively.

To transform unstructured text into structured numerical features using vectorization techniques.

To train a machine learning model that can accurately classify the sentiment of unseen tweets.

To evaluate the model using standard performance metrics.

Tools and Technologies Used
Python: Core programming language for building the project.

Pandas: Used for data manipulation and preprocessing.

Regular Expressions (re): For cleaning text data by removing URLs, mentions, and special characters.

Scikit-learn: Provides tools for vectorization, model training, and evaluation.

Multinomial Naive Bayes: The chosen algorithm for classification due to its efficiency in handling text data.

Data Description
The dataset used is a CSV file named twitter.csv, containing two columns:

tweet: The original tweet text.

label: The sentiment label associated with the tweet (e.g., positive or negative).

The project begins by loading the dataset and retaining only the relevant columns. This ensures a focused approach on sentiment classification.

Text Preprocessing
To make the text data suitable for machine learning, a multi-step cleaning process is applied:

URL Removal: Eliminates links to prevent bias.

Mention Removal: Removes @username tags.

Special Character Removal: Strips punctuation and digits.

Tokenization: Splits the text into words using regular expressions.

Stopword Removal: Filters out common words that don't contribute to sentiment, such as "the", "is", and "you".

The cleaned and filtered tokens are then rejoined to form a processed version of the tweet.

Vectorization
To convert the cleaned text into numerical format suitable for model input, the CountVectorizer from scikit-learn is used. This transforms each tweet into a vector based on word frequency, forming a bag-of-words representation of the corpus.

Model Training and Evaluation
The dataset is split into training and testing subsets using an 80:20 ratio. A Multinomial Naive Bayes classifier is then trained on the vectorized data. This algorithm is known for its performance in text classification tasks due to its probabilistic approach.

After training, the model is tested on the unseen data. Performance is evaluated using:

Accuracy Score: Measures the proportion of correctly classified tweets.

Classification Report: Provides precision, recall, and F1-score for both positive and negative classes.

Conclusion
This sentiment analysis project successfully demonstrates how natural language processing and machine learning can be applied to social media data for valuable insights. By accurately classifying the emotional tone of tweets, the model can support applications in brand monitoring, political analysis, and customer feedback evaluation. Future enhancements may include adding deep learning models, incorporating emoji analysis, and deploying the system through a web interface.


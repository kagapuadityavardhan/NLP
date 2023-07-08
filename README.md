# Restaurant Review Sentiment Analysis

This project is a simple sentiment analysis model that predicts the sentiment (positive or negative) of restaurant reviews using the Naive Bayes algorithm. It is implemented in Python and utilizes the scikit-learn library for machine learning and natural language processing tasks.

## Dataset

The dataset used for training and testing the model is stored in the file `Restaurant_Reviews.tsv`. It is a tab-separated values (TSV) file containing a collection of restaurant reviews along with their corresponding sentiment labels.

## Dependencies

To run this project, you need to have the following dependencies installed:

- Python 3.x
- NumPy
- Matplotlib
- pandas
- NLTK (Natural Language Toolkit)

You can install the required Python packages by running the provided command.

After installing the packages, you also need to download the NLTK stopwords corpus by following the provided instructions.

## Preprocessing

Before training the model, the dataset undergoes preprocessing steps to clean and transform the reviews into a suitable format for machine learning. The preprocessing steps include removing non-alphabetic characters, converting text to lowercase, splitting reviews into individual words, removing stopwords, and applying word stemming.

The preprocessed reviews are then stored for further processing.

## Feature Extraction

To convert the preprocessed reviews into numerical feature vectors, the CountVectorizer from scikit-learn is utilized. It tokenizes the reviews and constructs a matrix where each row represents a review, and each column represents a unique word. The feature matrix is prepared for training the model.

## Model Training and Evaluation

The dataset is split into training and testing sets. The Gaussian Naive Bayes algorithm is employed for training the sentiment analysis model. Once trained, the model makes predictions on the testing set, and the predicted labels are compared with the actual labels to evaluate the model's performance.

The performance metrics, including the confusion matrix and accuracy score, are provided.

## Usage

To run the project, make sure you have the required dependencies installed. Then, execute the Python script. The predicted labels for the testing set and the corresponding actual labels will be displayed in the console, along with the confusion matrix and accuracy score.


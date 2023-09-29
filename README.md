# Amazon-product-Reviews-Machine-Learning
Analyzed 30,000 product reviews through machine learning utilizing Logistic Regression, Decision Tree, &amp;  Random Forest Classifier to derive insights &amp; categorize data

COSC 74/274: Machine Learning and Statistical Data Analysis (Spring 2023 Class Final Project) 

# 1. Objective
The goal of the course project is to implement machine learning models and concepts covered
in this course for a real-world dataset. The project will utilize the Amazon product review dataset
and focus on binary classification, multi-class classification, and clustering approaches to
analyze and categorize product reviews. All code must be implemented in Python and all
models must use the Scikit Learn toolkit - https://scikit-learn.org/stable/index.html. You are not
allowed to use other toolkits, such as NLTK or transformer network architectures, for your
project results.

Here are examples of some useful Scikit modules:

1. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
2. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer
3. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer

Projects will be individual: each student will work on their own project. Students can discuss with
each other for clarification, but they should make sure not to share codes. Any collaboration or
sharing of ideas should be acknowledged in the project reports with sufficient details.

# 2. Amazon Review Dataset
Link to dataset: https://tinyurl.com/22yau9r8
You will be given two files: Training.csv and Test.csv.

# Training.csv: This file is a CSV file consisting of review-related information with the following fields:

1. overall: This is the product’s rating on a scale of (1-5)
2. verified: A boolean variable denoting if the review has been verified by Amazon
3. reviewTime: time of review
4. reviewerID: The unique ID of the Amazon reviewer (some have left multiple reviews)
5. asin: Product ID. One product will have many different reviews
6. reviewerName: Encoding of the Amazon reviewer’s username
7. reviewText: The Amazon review
8. unixReviewTime: unix time of review
9. vote: How many people voted this review as being helpful
10. image: If there is an image, link to the image
11. style: If there is style information (e.g., size of shirt, color of phone), it is embedded in a dictionary here. Only available for some samples
12. Category: The Amazon product category of the product.

# Test.csv
This file contains all the same features as Training.csv, but the overall variable is withheld. You
will submit your predictions of the overall for each product using this file and we will compare
them with the true labels.


# 3. Tasks
# 3.1 Binary classification
In this task, you have to develop binary classifiers to classify product reviews as good or bad.
The cutoff of ‘goodness’ will be an input, i.e., you have to develop classifiers with the following
cutoffs of product rating: 1,2,3,4. Note: The cutoff is not an input to the model, but to the
experiment. For example, when cutoff=3, all samples with a rating <= 3 will have label 0, and all
samples with a rating > 3 have label 1. You are expected to report the performance of at least
three different classifiers for each of the four cutoffs. You need to perform cross-validation for
hyperparameter tuning. Your report should describe why certain model parameters help or hurt
the model performance. For each classifier, you should report in your report the confusion
matrix, ROC, AUC, macro F1 score, and accuracy for the best combination of hyperparameters
using 5-fold cross-validation. We will share a baseline macro F1 score for classification and at
least one of your classification models must achieve at least the baseline score for full credit.

# 3.2 Multiclass classification
Turn the above classifier into a multiclass classifier where the target classes are 1,2,3,4,5. In
other words, you want to classify the product rating on a five-class scale. For each classifier, you
should report the confusion matrix, ROC, AUC, macro F1 score, and accuracy for the best
combination of hyperparameters using 5-fold cross-validation. You are expected to report the
performance of at least three different classifiers. We will share a baseline macro F1 score for
classification, and at least one of your classification models must achieve at least the baseline
score for full credit.

# 3.3 Clustering
In this task, you will cluster the product reviews in the test dataset. You will need to create word
features from the data and use that for k-means clustering. Clustering will be done by product
types, i.e., in this case, the labels will be product categories. You will use the Silhouette score
and Rand index to analyze the quality of clustering. We will share a baseline silhouette score for
clustering, and your model must achieve at least the baseline score for full credit.

# 3.4 Supplementary Instructions for Classification Tasks
For classification tasks (4 binary classification tasks and 1 multiclass classification task), you
should work on at least three different classifiers. For each classifier, below is the expected
procedure.
1. You should decide which hyperparameters you want to tune.
2. For each combination, you should compute a 5-fold cross-validation score. (You should
report the mean of the five-fold validation scores.) For example, let’s say there are 100
different combinations of hyperparameters for a classifier. You will then have 100
cross-validation scores.
3. Now, you can choose the best combination of hyperparameters based on the
cross-validation scores. Note that you have to report all the validation scores and explain
how you come up with the best combination of hyperparameters.
4. You should report the best model (with the chosen hyperparameters) in multiple metrics
(e.g. confusion matrix, ROC, AUC, macro F1 score, and accuracy). You should report
them on one of the validation sets (i.e. 20% of your data). For the multiclass
classification task, you should show 6 curves in one plot: 5 curves from each category
and the average curve

# Starbucks Capstone Challenge
#### A capstone project as part of my Udacity Data Scientist Nanodegree Program


## Table of contents

1. [Project Overview](#overview)
2. [Project Motivation](#motivation)
3. [Libraries Used](#Libraries)
4. [Project Directory Contents](#Contents)
5. [Authors](#Authors)
6. [Acknowledgement](#Acknowledgement)
7. [Conclusion](#Conclusion)
8. [Reflection](#Reflection)
9. [Improvement](#Improvement)


<a name="overview"></a>
###  1. Project Overview

The objective of this project is to do a detailed analysis of the simulated data provided by Udacity (simplified version of the real Starbucks app data) that mimics customer behavior on the Starbucks rewards mobile app, along with a machine learning model that predicts whether a customer will respond to an offer sent to respective users. 

Once every few days, Starbucks sends out an offer to its users' mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. ALso, not all users receive the same offer, and that is the challenge to solve with this data set.

The goal here is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. Data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, it can be assumed the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

Also a transactional data is provided that contains user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer.

A blog post for this project explaining the observations from all analysis and about data modelling is publihed on [Medium Platform](https://siddharthsabat.medium.com/analyze-customer-behavior-on-starbucks-rewards-mobile-app-and-predict-customer-response-to-an-offer-48b81200e8d1) 

<a name="motivation"></a>
## 2. Project Motivation

The problem statement that we will be looking to solve for the this project is to combine transaction, demographic and offer data and to analyze Starbucks rewards mobile app data on how users respond to different offers and predict whether a user will respond to an offer or not using the demographics and offer reward data.

Below are the problem statements for which we need to find the answer as part of this project:


1. What percentage of offers viewed by the users from all the offers sent to them?

2. How many offers from each offer type were sent to Customers?

3. What percentage of Customers completed an offer after viewing it and what percentage of Customers completed an offer without viewing it?

4. How many transactions were completed/influenced because of the sent offers?

5. Find out the facts that are influenced by the offer?

    a. how gender affects the amount spent in a transaction and which type of discount attracts which gender types.

    b. find out the spread of data for amount spent by the users who completed the offer after viewing it and before it expires, and for users who didn’t complete the offer.
        
    c. analyze the correlation between Age Groups and Amount spent by customers for completed offers transactions with and without offers.

    d. find the correlation between Customer Income and Amount Spent by customers for the transactions with and without completed offers.
    
    e. how age plays a role in responding to offer rewards.
    
6. How customers responded to offers for the different advertisement channels used for both completed and uncomplete offers?

7. Predict whether a customer will respond to an offer or not using RandomForestClassifier and LinearSVC.

<a name="Libraries"></a>
## 3 Libraries Used
* Python 3.5+
* Python Pandas, NumPy, json, Matplotlib, Seaborn, pickle libraries
* Sklearn libraries such as train_test_split, GridSearchCV, svm.LinearSVC, Pipeline, classification_report, classification_report, preprocessing, RandomForestClassifier,       precision_score, recall_score, confusion_matrix

<a name="Contents"></a>
## 4 Project Directory Contents
* **Data:**

* There are 3 datasets provided by udacity as part of this project. 

    - portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
    - profile.json - demographic data for each customer
    - transcript.json - records for transactions, offers received, offers viewed, and offers completed

* **Screenshot:**

    Screenshots from the code visualization are captured 

* **Root Directory:**

    Starbucks_Capstone_notebook.ipynb: Contains the Jupiter notebook with detailed analysis and code for this project.
    Starbucks_Capstone_notebook.html: HTML version of the Jupiter notebook.       

<a name="Authors"></a>
## 5. Authors

#### Siddharth Sabat
* https://github.com/siddharthsabat
* https://www.linkedin.com/in/sidharthsabat88/
* Link to clone the GitHub Repo is [here](https://github.com/SiddharthSabat/Data-Science-and-ML-Portfolio/tree/main/Data_Scientist_Starbucks_Capstone)


<a name="Acknowledgement"></a>
## 6. Acknowledgements

* Thanks to [Udacity](https://www.udacity.com/) for providing the simulated datasets in collaboration with Starbucks and creating such a wonderful Data Scientist Nanodegree Program containing comprehensive course material for aspiring Data Scientists. 
* Here is the link for the data scientist nano degree program detail: [Click Here](https://www.udacity.com/course/data-scientist-nanodegree--nd025)


<a name="Conclusion"></a>
## 7. Conclusion

## Justification

Based on the analysis and exploration done, we identified how customer demographics and the offer rewards affect user response to offers/advertisements sent through various visualization.

First, we identified users for whom there are demographic information was missing, and we classified them into a separate group. There are 13% of the total users for which there is no demographic info available in the dataset. This helped us identify accurately the gender distribution in the dataset. We saw that males take up 57% of total users and females take up 41%, leaving 1% for others.


Then we saw income distribution for the users. Data suggests many users have an annual income in the range between 30000 USD and 50000 USD and majority of the customers having income in the range between 50000 USD to 75000 USD. 

Then we saw distribution of offer type events. Data contains 55% of records as  offer type events and 45% records are of transactions type data.

After that we identified the breakdown for no of offers sent to customers. Almost same number of Buy 1 Get 1 and Discount offers were sent to users which is almost double the number of total Advertisement offers sent.

Then we identified , out of total 76277 offers sent, 57725 offers were viewed by the users which is 75.68% of offers that is viewed by users from all the offers sent to them. And 71% of completed offers were made after users viewed them leaving 29% completed the offers without viewing it.


After that from many of the visualization techniques, we derived the facts that are influenced by the offer such as 

    a. how gender affects the amount spent in a transaction and which type of discount attracts which gender types.

    b. spread of data for amount spent by users who completed the offer after viewing it and before it expires and for users who didn't complete the offer.

    c. correlation between Age Groups and Amount spent by customers for the transactions with and without completed offers.

    d. find the correlation between Customer Income and Amount Spent by customers for the transactions with and without completed offers.

    e. how age plays a role in responding to offer rewards.

Then, we identified how customers responded to offers for the different advertisement channels used for both completed and uncomplete offers

Finally, we trained two supervised classification models that predicts whether an user will respond to an offer or not based on demographics and offer reward data. Both the models predicted user responses with an accuracy of 87%, F1-score of 0.93 by both models for those who won't respond to an offer, F1-score of 0.69 and 0.70 for identifying those who will respond to offers by RandomForestClassifier and LinearSVC respectively.

Random Forest creates as many trees on the subset of the data and combines the output of all the trees to reduces overfitting problem in decision trees and also reduces the variance and therefore improves the accuracy, hence this algorithm is chosen as one of the modelling technique for this project.

Since Linear SVC classifier is relatively faster and takes just one parameter to tune, hence we selected LinearSVC as our second modelling technique to predict user response as part of this project.

Since the f1-score and accuracy for both RandomForestClassifier and LinearSVC models are pretty much the same when tuned with the GridSearchCV, and since RandomForestClassifier is taking much time (around 1100 seconds) compared to LinearSVC when training the model, we can go ahead with the LinearSVC as our final model implementation to predict the user responses towards offer rewards.

<a name="Reflection"></a>
### Reflection

The problem that I chose to solve as part of this project is to build a model that predicts whether a customer will respond to an offer or not. The approach being used for solving this problem has mainly three steps. 

* First, after preprocessing portfolio, profile and transaction datasets, data sets were combined to get a final clean data containing relevant features which can be used to train our model.

* Second, After splitting data to train and test datasets, we choose the modelling techinqies "RandomForestClassifier" and LinearSVC with GridSearch classifier algorithms. 

* Third, we predicted the test target using test data, and plot a confusion matrix to evaluate the prformance of our model, and we found that our predictive model is well suited for this case.

The most interesting aspect of this project is the combination between different datasets, using predictive modeling techniques and analysis to provide better decisions and value to the business. The data exploration and wrangling steps were the longest and most challenging part. The toughest part of this entire analysis was to find right logic and strategy to answer the problem statements and conveying them with different visualization techniques.

<a name="Improvement"></a>
### Improvement

* Better predictions may have been deducted if there were more customer metrics available. For this analysis, I believe there are limited information available about customers which is just age, gender, and income. To find optimal customer demographics, it would be nice to have a few more features of a customer. These additional features may aid in providing better classification model results data in order to have more better model.


* To improve prediction results, we can consider increasing the data size by collecting data over a larger period of time. In this project, the data was collected over a period of one month, but it's not clear from which day the experiment started, some people may pay more in the start of the month than they would at the end because of salary dependecies. So collecting data over 3 months or more would produce a big improvement in the prediction results. Additionaly, after merged the data sets and removed duplicate entries, the data got even less records. So With more data, the classification models may have been able to produce better F1-score results.
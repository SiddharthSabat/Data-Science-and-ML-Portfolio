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

A blog post for this project explaing the observations from all analysis is publihed on [Medium Platform](https://siddharthsabat.medium.com/analyze-customer-behavior-on-starbucks-rewards-mobile-app-and-predict-customer-response-to-an-offer-48b81200e8d1) 

<a name="motivation"></a>
## 2. Project Motivation

Given the datasets below, the objective is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer reward, and to predict which users, who normally wouldn't make a purchase, would respond to a sent offer and make a purchase through it. This can be done by first answering a below questions:

1.  What percentage of Customers view offers and find out their characteristics?
2.  What percentage of Customers respond to offers and what percentage complete the offer after viewing it?
3.  How much do adevertisements contribute in user transactions?
4.  Predict whether a Customer will respond to an offer or not using demographics and offer reward data?


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

Based on the analysis and exploration done, how customer demographics and the offer reward affect their response to offers or advertisements sent are identified. 

First, we identified users who don't prefer to share their personal information, and we classified them into a separate group who provided the personal info. There are 13% of the total users who have prefered not to share their personal info. This helped us identify accurately the gender distribution in the dataset. We saw that males take up 57% of total users and females take up 41%, leaving 1% for others.

Next we observed that the majority of users are in their late 50's or eary 60's, and that the number of users decreases as we move away from the peak. And age doesn't affect the user attraction towards certain offer rewards. However, it was seen that the average amount spent per transaction increases as user age increases.

Many users have an annual income in the range between 30000 USD and 50000 USD, but the majority are in the range between 50000 USD and 75000 USD, and of course less users have high annual salary. The amount spent per transaction is more when the user income is more, which is expected.

After that we saw that 75% of all the received offers were actually viewed, and that 71% of users complete offers after they view them leaving 29% completing offers unintentionally.

It was seen that offers influence 19% of the total transactions or completed offers that occured, which is pretty big, and that users are 7% more likely to respond to offers when they are sent through social media.

We saw how gender plays role in the average amount spent by a user, and also in responding to which type of offer reward. Males responded more to the 2, 3, and 5 dollar rewards while females respond more to the 10 dollar rewards, and on average, females spend more than males.

People who chose to stay anonymous tend to spend more per transaction in the group that responded to offers, however, for the other group it's completely the opposite, known users spend a lot more than anonymous users.

Finally, we built a model that predicts whether a user will respond to an offer or not based of demographics and offer reward, and the model predicted this with an accuracy of 87%, a F1-score of 0.65 for identifing those who will repond to offers, and an F1-score of 0.92 for those who won't.

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
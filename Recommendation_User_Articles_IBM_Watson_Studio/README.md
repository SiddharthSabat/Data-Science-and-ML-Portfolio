# Recommendation of User Articles on IBM Watson Studio Platform 
#### As part of Udacity Data Scientist Nano Degree Program

## Table of Contents
1. [Project Introduction](#Introduction)
2. [Description](#Description)
	1. [Exploratory Data Analysis](#Exploratory_DA)
	2. [Rank Based Recommendations](#Rank_Based)
	3. [User-User Based Collaborative Filtering](#User)
	4. [Content Based Filtering](#Content)
    5. [Matrix Factorization](#Matrix)
3. [Dependencies](#Dependencies)
4. [Project Directory Contents](#Contents)
5. [Authors](#Authors)
6. [Acknowledgement](#Acknowledgement)
7. [Conclusion and Way Forward](#Conclusion_WF)


<a name="Introduction"></a>
## 1. Project Introduction

* This Project is part of Udacity Data Scientist Nanodegree Program. The Objective of this project is to analyze user article interactions on the real data retrieved from IBM Watson Studio platform and make different recommendations to the users with new articles based on their historical interactions. Also to recommend articles to new users on the platform.

* The dataset used for this project is facilitated by Udacity from IBM Watson Studio platform. Below is a preview of the article recommendations displaying just the newest articles to an user on the platform Dashboard, however the idea is to develop a recommendation engine based on data available to us and display articles on the platform Dashboard in similar fashion that are most pertinent to a specific user.

![Intro Pic](./screenshot/Preview.png)

<a name="Description"></a>
## 2. Description

* This project is divided into below sections. There are various recommendations developed as part of the project depending on the user profiles. Application and usage of each methods are explained as below.

<a name="Exploratory_DA"></a>
#### I. Exploratory Data Analysis
* This part does the pre-processing of data, remove duplicate records and makes the dataset ready for developing different recommendations.
* An email wrapper function is also implemented to convert the email ids (stored in hexadecimal forms in the original dataset representing unique id of users) to numerical ids for ease of further analysis 

<a name="Rank_Based"></a>
#### II. Rank Based Recommendations

* This type of recommendation normally based on the popularity of the items (like the items that are being the most trending on internet), often based on the ranks of most user to items ( articles in our case) interactions. For example, on Amazon account, the best seller books often gets listed/recommended to most of users if they browse the books section. This is because of their popularity among the readers.

* As part of this recommendation, top n number of articles are recommended for a given user. A function is created which takes no of articles and a dataframe as inputs, then the function recommends top n articles to the given user. Since there are not much info available for new users, this recommendation method is often used for new users on the platform based on the articles that are most interacted with or the most popular articles among user base.

<a name="User"></a>
#### III. User-User Based Collaborative Filtering

* User-User Based Collaborative Filtering is a technique which is widely used in recommendation systems and is rapidly advancing research area in many industry. Many websites like Amazon, Netflix, Facebook, Majority of e-commerce sites etc. use collaborative filtering method for building their recommendation system.

* This recommendation method is used to predict the items that an user might like based on ratings given to that item by his/her most similar users who have similar taste with that of the target user. The most similar user to a given target user is decided based on the dot product matrix of user item matrix and its transpose matrix. 

* The algorithm implemented here, takes target user_id as an input (for which the recommendation is intended), then it sorts each user based on similarity to the give user in descending order. The similarity is calculated based on the dot product of user matrix and its transpose matrix.

* For each sorted user, it finds articles that the sorted user has interacted with and adds that article to the recommendations list.

* Then it selects top m number of articles to recommend to the input/target user. Here m is the number of recommendations to provide for a specific user_id.

<a name="Content"></a>
#### IV. Content Based Filtering

* This part of the project is optional to implement. It will be implemented in coming weeks and this section will be updated accordingly.

<a name="Matrix"></a>
#### V. Matrix Factorization

* Matrix factorization is a class of collaborative based filtering algorithms used in recommender systems. 

* It is a way to generate latent features when multiplying two different kinds of entities (like user and movies or user and articles). Collaborative filtering is the application of matrix factorization to identify the relationship between items’ and users’ entities.
 
* SVD is a matrix factorization technique, which decomposes the number of features of a dataset by reducing the space dimension from n-dimension to k-dimension (where k<n). In the context of the recommender system, the SVD is used as a collaborative filtering technique. It uses a matrix structure where each row represents a user, and each column represents an item. The elements of this matrix are the ratings that are given to items by users.

* In this section, Singular Value Decomposition algorithm is used on the user_item interactions matrix. Then the behavior of accuracy vs the number of latent features is observed using visualization.

* From the Accuracy vs No of Latent Features visualization, its concluded that increasing latent features is causing overfitting problem. The accuracy becomes worser when the number of latent features increases. Since there are only 20 common users are shared between the train and test sets, results may not be significant in statistical terms, hence other recommendation methods can be used to improve the recommendation here, like collaborative filtering or content-based recommendation.

<a name="Dependencies"></a>
## 3 Dependencies
* Python 3.5+
* Python Pandas, NumPy, Matplotlib, Seaborn, pickle libraries

<a name="Contents"></a>
## 4 Project Directory Contents
* **Data:**

    articles_community.csv and user-item-interactions.csv: contains real data facilitated by Udacity from IBM Watson platform containing user article     interactions and article info.
    
    user_item_matrix.p: Pickle file containing the user item matrix for matrix factorization facilitated by Udacity.
   
* **Screenshot:**

    Preview screenshot of newest articles on IBM Watson platform dashboard as a reference. The recommended articles will be displayed to users in the     same way.
    Other screenshots are placed here to be used in my medium post for this project.

* **Root Directory:**

    Contains the Jupiter notebook of the various recommendation process and a test python script.
    
    project_tests.py: contain the tests functions that quality checks for right results generated from each recommendation algorithms which is           helpful evaluate the algorithm logic at each stage.
    
    top_5.p, top_10.p, top_20.p: These are the pickle files containing top 5, 10 and 20 article recommendations facilitated by Udacity. These files       are used along with the project_tests.py test script to check our algorithm logic for Rank based recommendations.

<a name="Authors"></a>
## 5. Authors

* [Siddharth Sabat](https://github.com/siddharthsabat)
* Link to clone the GitHub Repo is [here](https://github.com/SiddharthSabat/Data-Science-and-ML-Portfolio/tree/main/Recommendation_User_Articles_IBM_Watson_Studio)


<a name="Acknowledgement"></a>
## 6. Acknowledgements

* Thanks to [Udacity](https://www.udacity.com/) for providing the datasets and creating such a wonderful Data Scientist Nanodegree Program containing comprehensive course material for aspiring Data Scientists and awesome projects like this one. Here is the link for the data scientist program detail: [Click Here](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
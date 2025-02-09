# Prodigy InfoTech Internship Repository

This repository contains tasks completed during my internship at Prodigy InfoTech. Below is a breakdown of the tasks:

## Task 1: Linear Regression for House Price Prediction

**Dataset:** https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
<br>

This project is part of my internship at Prodigy InfoTech, where I implemented a machine learning model to predict house prices based on given features like square footage, number of bedrooms, and bathrooms. The project demonstrates the use of Linear Regression for predictive modeling.
<br>

**[Project Overview]**
<br>
The goal of this project is to build a regression model that accurately predicts house prices using a dataset from Kaggle. The dataset includes features such as:
<br>
:) Square Footage
<br>

:) Number of Bedrooms
<br>

:) Number of Bathrooms
<br>

:) stories (Number of House Stories)
<br>

:) mainroad (Whether connected to Main Road)
<br>

:) guestroom (Whether has a guest room)
<br>

:) basement (Whether has a basement)
<br>

:) hotwaterheating (Whether has a hotwater heater)
<br>

:) airconditioning (Whether has an airconditioning)
<br>

By training a linear regression model, we can identify the relationship between these features and house prices.
<br>

**[Tools Used]**
<br>
**Python** : For scripting and implementation.
<br>

**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**[Libraries Used]**
<br>
**numpy**: For numerical computations.
<br>

**pandas**: For data manipulation and preprocessing.
<br>

**scikit-learn**: For building and evaluating the linear regression model.
<br>

**[Implementation Steps]**
<br>
(1) **Data Preprocessing**: 
<br>
:) Handled missing values, if any.
<br>
:) Performed data exploration and visualization to understand the relationships between features and target values.
<br>

(2) **Feature Engineering**:
<br>
:) Selected relevant features for the prediction model.
<br>

(3) **Model Building**:
<br>
:) Implemented a Linear Regression model using scikit-learn.
<br>

(4) **Evaluation**:
<br>
:) Used metrics like Root Mean Squared Error (RMSE) and R² Score to evaluate model performance.
<br>

**[Results]**
<br>
The model successfully captured the relationship between house features and prices.
<br>

**[Evaluation Metrics]**
<br>
**R² Score**: [0.4559299118872445]
<br>

**Root Mean Squared Error (RMSE)**: [1658324.6001036866]
<br>



## Task 2: Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.

**Dataset:** https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
<br>

This project is part of my internship at Prodigy InfoTech, where i implemented a K-means clustering algorithm to group customers of a retail store based on their purchase history based on given features like CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100). 
<br>

**[Project Overview]**
<br>
The goal of this project is to create a K-means clustering algorithm to group customers of a retail store based on their purchase history using a dataset from Kaggle. The dataset includes features such as:
<br>
:) CustomerID : Unique ID assigned to the customer
<br>

:) Gender : Gender of the custome
<br>

:) Age : Age of the customer
<br>

:) Annual Income (k$) : Annual Income of the customee
<br>

:) Spending Score (1-100) : Score assigned by the mall based on customer behavior and spending nature
<br>


**[Tools Used]**
<br>
**Python** : For scripting and implementation.
<br>

**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**[Libraries Used]**
<br>
**numpy**: For numerical computations.
<br>

**pandas**: For data manipulation and preprocessing.
<br>

**matplotlib**: A widely used data visualization library.
<br>

**seaborn**: Built on top of Matplotlib, used for statistical data visualization.
<br>

**scikit-learn**: A machine learning library that provides simple and efficient tools for data analysis and modeling.
Includes KMeans for clustering and StandardScaler for feature scaling .
<br>

**[Implementation Steps]**
<br>

(1) **Data Preprocessing**: 
<br>
:) Load the dataset using Pandas.
<br>
:) Remove unnecessary columns like CustomerID and Gender, as they don't contribute to clustering.
<br>
:) Scale numerical features using StandardScaler to normalize values.
<br>


(2) **Finding Optimal Clusters (Elbow Method)** : 
<br>
:) Use the Elbow Method to determine the best number of clusters by plotting inertia (Sum of Squared Errors - SSE).
<br>

(3) **Applying K-Means Algorithm** : 
:) Train the K-Means model with the optimal number of clusters (determined by Elbow method).
<br>
:) Assign each customer a cluster.
<br>

**Results & Visualization** :
<br>
:) The Elbow Method plot suggests the optimal number of clusters is 5.
<br>
:) The final visualization shows customers grouped based on their Annual Income & Spending Score.
<br> 
:) Different clusters indicate different customer behaviors (e.g., high income & high spending vs. low income & low spending).
<br> 



**Author**: Simran Chaudhary
<br>
Role: Machine Learning Intern at Prodigy InfoTech
<br>
**LinkedIn**: https://www.linkedin.com/in/simran-chaudhary-5533b7308/
<br>
**GitHub**: https://github.com/Simran-ch
<br>

## Conclusion
<br>
This repository showcases my work on various ML tasks during my internship. Each task focuses on solving specific problems using different machine learning techniques.

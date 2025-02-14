# Prodigy InfoTech Internship Repository

This repository contains tasks completed during my internship at Prodigy InfoTech. Below is a breakdown of the tasks:

## Task 1: Linear Regression for House Price Prediction

**Dataset:** https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
<br>

This task is part of my internship at Prodigy InfoTech, where I implemented a machine learning model to predict house prices based on given features like square footage, number of bedrooms, and bathrooms. The task demonstrates the use of Linear Regression for predictive modeling.
<br>

**--Project Overview--**
<br>

The goal of this task is to build a regression model that accurately predicts house prices using a dataset from Kaggle. The dataset includes features such as:
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

**--Tools Used--**
<br>

**Python** : For scripting and implementation.
<br>
**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**--Libraries Used--**
<br>

**numpy**: For numerical computations.
<br>
**pandas**: For data manipulation and preprocessing.
<br>
**scikit-learn**: For building and evaluating the linear regression model.
<br>

**--Implementation Steps--**
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

**--Results--**
<br>

The model successfully captured the relationship between house features and prices.
<br>

**--Evaluation Metrics--**
<br>

**R² Score**: [0.4559299118872445]
<br>
**Root Mean Squared Error (RMSE)**: [1658324.6001036866]
<br>



## Task 2: Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.

**Dataset:** https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
<br>

This task is part of my internship at Prodigy InfoTech, where i implemented a K-means clustering algorithm to group customers of a retail store based on their purchase history based on given features like CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100). 
<br>

**--Project Overview--**
<br>

The goal of this task is to create a K-means clustering algorithm to group customers of a retail store based on their purchase history using a dataset from Kaggle. The dataset includes features such as:
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


**--Tools Used--**
<br>

**Python** : For scripting and implementation.
<br>
**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**--Libraries Used--**
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

**--Implementation Steps--**
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
<br>
:) Train the K-Means model with the optimal number of clusters (determined by Elbow method).
<br>
:) Assign each customer a cluster.
<br>

**--Results & Visualization--** 
<br>
:) The Elbow Method plot suggests the optimal number of clusters is 5.
<br>
:) The final visualization shows customers grouped based on their Annual Income & Spending Score.
<br> 
:) Different clusters indicate different customer behaviors (e.g., high income & high spending vs. low income & low spending).
<br> 







## Task 3: Implement a support vector machine (SVM) to classify images of cats and dogs .

**Dataset:** https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification/data
<br>

**--Dataset Structure--**
<br>
The dataset is well-structured, containing two folders:
<br>
:) **train** (for model training)
<br>
:) **test**  (for model evaluation)
<br>
Each folder has two subdirectories— cats and dogs, making it a great dataset for supervised learning tasks.
<br>


This task is part of my internship at Prodigy InfoTech, where i implemented a **support vector machine (SVM)** to classify images of cats and dogs.
<br>

**--Project Overview--**
<br>

This project implements an image classification model to distinguish between cats and dogs using a Support Vector Machine (SVM) classifier. Instead of training a Convolutional Neural Network (CNN) from scratch, we leverage MobileNetV2 as a feature extractor to improve efficiency and accuracy.
<br>


**--Tools Used--**
<br>

**Python** : For scripting and implementation.
<br>
**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**--Libraries Used--**
<br>

**numpy**: For numerical computations.
<br>
**os** : To navigate through dataset directories.
<br>
**cv2 (OpenCV)** : For image processing (resizing, reading images).
<br>
**pandas**: For data manipulation and preprocessing.
<br>
**tensorflow.keras.applications.MobileNetV2** : Pretrained CNN model for feature extraction.
<br>
**tensorflow.keras.applications.mobilenet_v2.preprocess_input** : To preprocess images before feeding them into MobileNetV2.
<br>
**sklearn.svm.SVC** : Support Vector Machine (SVM) classifier.
<br>
**sklearn.model_selection.train_test_split** : To split the dataset into training and testing sets.
<br>
**sklearn.preprocessing.LabelEncoder** : To encode categorical labels (cats & dogs).
<br>
**sklearn.metrics.accuracy_score** : To measure model accuracy.
<br>
**sklearn.metrics.confusion_matrix** : To analyze classification performance.
<br>
**sklearn.metrics.classification_report**  To generate precision, recall, and F1-score for model evaluation.
<br>
**matplotlib**: A widely used data visualization library.
<br>

**--Implementation Steps--**
<br>

(1)  **Load & Preprocess the Data**
<br>
:) Read images from directories using OpenCV (cv2).
<br>
:) Resize images to 224x224 pixels to match MobileNetV2 input requirements.
<br>
:) Normalize pixel values.
<br>
:) Encode labels ("Cat" → 0, "Dog" → 1).
<br>

(2) **Feature Extraction using MobileNetV2**
<br>
:) Load MobileNetV2 (pre-trained on ImageNet) without the top layer.
<br>
:) Extract deep learning features from images.
<br>

(3) **Train an SVM Classifier**
<br>
:) Use Scikit-Learn's SVM (Support Vector Machine) with a linear kernel.
<br>
:) Train on extracted features.
<br>

**--Results & Visualization--** 
<br>
:) 𝐎𝐯𝐞𝐫𝐚𝐥𝐥 𝐀𝐜𝐜𝐮𝐫𝐚𝐜𝐲: 90% on test data
<br>
:) 𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐜𝐚𝐭𝐢𝐨𝐧 𝐑𝐞𝐩𝐨𝐫𝐭 : Produced a detailed classification report:
<br>
 𝐏𝐫𝐞𝐜𝐢𝐬𝐢𝐨𝐧: 0.94 (cats), 0.87 (dogs)
<br>
 𝐑𝐞𝐜𝐚𝐥𝐥: 0.86 (cats), 0.94 (dogs)
<br>
 𝐅𝟏-𝐒𝐜𝐨𝐫𝐞: 0.90 (cats), 0.90 (dogs)
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

# Industrial-Machine-Failure-Prediction
This project uses machine learning to predict machine failures in manufacturing operations. The model analyzes sensor data and operational features to classify machines as either likely to fail or not, enabling proactive maintenance and reducing unexpected downtime.
# Video URL
https://www.loom.com/share/430637d68f664b0a83e3547e01e9172f?sid=30f69d27-8257-491d-b0f6-72f6c01fc7a8

Step 1: Understand the problem statement 
Step 2: Asking right Questions 
Step 3: Data Collection / Overview and Loading the data
Step 4: Preprocess data and perform exploratory data analysis
Step 5: Train and evaluate the models
Step 6: Handling Imbalanced Classes (if necessary) to Improve Model Performance
Step 7: Repeat step 5
------------------------------------------------------------------------------
Step 1: Understanding the Problem and Goal:
============================================

Customer Problem Context: Our manufacturing operations are experiencing unexpected machine failures, resulting in unplanned downtime, 
reduced production efficiency, and increased maintenance costs. These failures occur with little to no warning, and we do not have 
adequate insights into which machines are at risk of failure. As a result, we are unable to effectively plan for maintenance and prevent
production interruptions, leading to costly operational disruptions and delayed product deliveries.

Goal : The goal of this project is to develop a machine learning-based predictive maintenance model that accurately classifies 
whether a machine will fail or not in advance based on various operational data and sensor readings. 




---------------------------------------------------------------------------------
Step 2: Asking the right Questions 
==================================



What does "machine failure" mean in this context?
What is the operational impact of machine failure?
What are the business objectives?
Is the goal to minimize unplanned downtime, extend machine life, or reduce maintenance costs?
Is it the complete breakdown of the machine, a critical malfunction, or a degraded state that requires maintenance? 


By predicting potential machine failures before they occur, the aim is to minimize downtime, 
reduce maintenance costs, and optimize machine performance, ultimately improving overall operational efficiency in the manufacturing process


Data Understanding and Scope

Which sensor readings and operational data are most indicative of failure?
Which features are expected to have the greatest impact on machine health?
Do we have sufficient data to detect failure trends?
Are the data quality and quantity sufficient to detect meaningful patterns or trends of failure?
Is there a standard definition for failure conditions, or do we need to define one?
Which certain feature values indicate failure risk?

Customer already has data for 1,000 machines, we will use this data to predict machine failure before delivery. According to customer provided data , we will use various operational data and 
sensor readings such as air temperature, process temperature, rotational speed, torque, tool wear to predict the machine.


---------------------------------------------------------------------------------
Step 3: Data Collection / Overview and Loading the data
========================================================


Source :  https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
University of California Irvine machine learning repository

Dataset Columns :

UDI
Product ID	
Type	
Air temperature [K]	
Process temperature [K]	
Rotational speed [rpm]	
Torque [Nm]	
Tool wear [min]	
Machine failure


Explanation of the Dataset Features:

UDI (Unique Device Identifier): A unique identifier for each machine.
Product ID: The specific identifier for the product type or model of the machine.
Type: The type or category of machine, which may provide context for failure prediction.
Air Temperature [K]: Operational air temperature, potentially affecting machine performance.
Process Temperature [K]: Temperature during operation, which could influence wear and failure rates.
Rotational Speed [rpm]: The rotational speed of parts, often linked to mechanical stress and potential failure.
Torque [Nm]: The amount of force being applied to a rotating part, which could indicate wear or overloading.
Tool Wear [min]: Duration of tool use, a direct indicator of the condition of machine components.
Machine Failure: The target variable (1 for failure, 0 for no failure).



------------------------------------------------------------------------------------
Step 4 :Steps for Preprocessing and Modeling / EDA :
===================================================

1. Discard unnecessary column :

UDI: Unique identifier for each record. This column is typically not useful for prediction, we can discard this column.
Product ID: Likely a categorical feature (machine/product identifier). 

2. Check for missing values

3. Handle Categorical Features

Type: Categorical feature (type of machine or process). This should be encoded into numerical format.
In our case we have Type colum will map strings ('L', 'M', 'H') to numerical values (0, 1, 2).

4. Data type conversion


Following are numerical features, likely continuous. It can provide important information related to machine performance. We will convert into float.


Air temperature [K]: 
Process temperature [K]: 
Rotational speed [rpm]: 
Torque [Nm]: 
Tool wear [min]: 


We will convert specific columns of the dataset to floats or integer types as required

5. Feature Distributions / Feature Scaling / Feature Normalization

In our case we are going for binary classification and we will go with Random Forest so skewed data does not require transformation,
though outlier handling might still be important.

Also there is not need for feature scaling

5. Heatmap of correlations

To identifying and unserstand the relationships / Coorelation between the features and target variables


-------------------------------------------------------------------
Step 5: Train and evaluate the models
======================================

1. Split datasets into features and target

We have 4 features variables and our target variable is IsFail

2. Splitting the Data

Split the data into training and testing sets (e.g., 70% training, 30% testing) using train_test_split from scikit-learn.

3. Fit and predict using RandomForestClassifier


4. Check the balance of data 

We will draw the confusion matrix to check the balance of data 

5. Evaluate your model using metrics

Evaluation Metrics:
In the case of imbalanced data, accuracy is not the best metric because it could be misleading. Instead, focus on metrics like:

Precision: The proportion of true positives among all positive predictions.
Recall: The proportion of true positives among all actual positives.
F1-Score: The harmonic mean of precision and recall, which is more informative for imbalanced datasets.
ROC-AUC: The area under the receiver operating characteristic curve, which gives insight into how well the model distinguishes between classes.

from sklearn.metrics import accuracy_score , recall_score , f1_score,classification_report , roc_auc_score , precision_score

------------------------------------------------------------------------------
6. Handling Imbalanced Classes (if necessary) to Improve Model Performance:
============================================================================

Adjusting Your Model using class weight :

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

Adjusting Your Model using SMOT :

If target variable (Machine failure) is imbalanced , then we will resampling the dataset
using SMOTE for oversampling the minority class. SMOTE (Synthetic Minority Over-sampling Technique)

from imblearn.over_sampling import SMOTE


------------------------------------------------------------------------------
6. Repeat step 5  
============================================================================

Tuning: Consider tuning hyperparameters like the number of trees (n_estimators), maximum depth (max_depth), and other parameters to optimize performance.












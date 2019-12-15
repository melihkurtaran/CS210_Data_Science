"""

This is a Predictive Analytics Homework, please answer the questions and fill the functions only with using python.

In this assignment, you are going to practice your exploratory data analysis skills. Each question is accompanied by an incomplete function. Your task is to construct the body of those functions with appropriate return types.

In this homework, we'll use Random Forest for classification by  using a German Credit Approval dataset.

Description:

When a bank receives a loan application, based on the applicant’s profile the bank has to make a decision regarding whether to go ahead with the loan approval or not. Two types of risks are associated with the bank’s decision which are the followings:

-If the applicant is a good credit risk, i.e. is likely to repay the loan, then not approving the loan to the person results in a loss of business to the bank

-If the applicant is a bad credit risk, i.e. is not likely to repay the loan, then approving the loan to the person results in a financial loss to the bank

A description of the data can also be found at:
https://newonlinecourses.science.psu.edu/stat508/resource/analysis/gcd

# **Question 1**


In the shared folder, you will see german-credit dataset. The following questions require you to read the datasets into dataframes. Therefore, you need to add the datasets to your own drive (Right Click->Add to My Drive), in addition to the colab notebook as stated in the instructions. And lastly, please do not forget to mount to your drive by executing the cell below.
"""

from google.colab import drive
drive.mount('/content/drive')

"""In the below cell please read the german-credit-data from drive and **assign it to the variable named as "df"**"""

import pandas as pd 
import numpy as np

from os.path import join  # merging paths in a better way

data_path = "/content/drive/My Drive/"
filename = "german_credit.csv"

df = pd.read_csv(join(data_path, filename),header=0)

"""# **Question 2**

Please print the following attributes of the data:

-Shape

-Column names

-First 5 rows

Finally,one should see the data consists of 1000 entries and 21 attributes.
"""
df.shape

df.columns

df.head()

"""

There is a creditability column in the dataframe, and this is the column that corresponds credithworthiness of appliciants.

If this column is equal to 1 then this means the applicant is a good credit risk, if this columns is equalt to 0 then this means the applicant is a bad cred risk.

# **Question 3**

The "Creditability" column is the our target feature and we will use remaning features (columns) to predict creditability of applicants.

Please divide the "df" dataframe we created in the first question to 2 part named as  df_x and df_y. 

df_x : correspond remaining features.

df_y : correspond only  'Creditability' feature.

Afterwards, print both of them.
"""

df_x = df.drop('Creditability', axis=1) # these are features
df_y = df['Creditability']  # this is the target (what we want to predict)

df_x

df_y

"""# **Question 4**

As we discussed in the recent recitations, we should divide the data-set into the three sub-part, which corresponds **train**, **validation** and **test**.

Please the **df_x** and **df_y** into the three sub part, train,valid and test respectively. You should assign them on the variables named as **X_train**, **X_valid**, **X_test**, **y_train**,**y_test**,**y_valid**

**Notation:** Please careful to the sizes of train, validation and test.

Train must account for 75% percent of the data.

Validation must account for 12.5% percent of the data.

Test must account for 12.5% of the data.

Please **print the shapes as well.** 

**Hint:** If you are manually divide the data, do not forget to shuffle it first !!, try to use libraries we discussed in the recitation
"""

from sklearn.model_selection import train_test_split

test_ids = np.random.permutation(df_x.shape[0]) #for shuffling
dum_x=np.array(df_x)
dum_y=np.array(df_y)

X_tr = dum_x[test_ids[0:750]]
X_val = dum_x[test_ids[750:875]]
X_test=  dum_x[test_ids[875:]]

y_tr = dum_y[test_ids[0:750]]
y_val = dum_y[test_ids[750:875]]
y_test=  dum_y[test_ids[875:]]

print(X_tr.shape)
print(X_val.shape)
print(X_test.shape)
print(y_tr.shape)
print(y_val.shape)
print(y_test.shape)

#It corresponds X_tr,X_val,X_test,y_tr,y_val,y_test  respectively.

"""# **Question 5**
Use randomForest to predict the Creditability in the validation credit dataset , using all
the remaining predictor variables, with the default hyper-parameters. Please follow below guideline.

-Fit the model with train data.

-Then predict validation

-Calculate accuracy of validation
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
rf= RandomForestClassifier()

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_tr, y_tr)

rf_predictions = model_rf.predict(X_val)
rf_acc = accuracy_score(y_val, rf_predictions)

print("Random Forest Accuracy:"+str(rf_acc))

"""# **Question 6**

First of all, let's look at the hyper-paramaters that random forest take. For more info let's look at the  (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

You can see the hyperparameters that random forest take in the below.
"""

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
print(rf.get_params)

"""Hyper-parameter known as n_estimators corresponds to the  number of trees in the forest.

Please **tune the model according to the  "n_estimators" hyper-parameter** on the different values specifed at the next cell on the validation data-set, then **plot the Number of trees versus validation error and train error on a graph** , just like in the recitation. Afterwards, **find the best  n_estimators value** that provide lowest error on the validation data-set. Then **predict the "Credictability"** in the **test data** with selected n_estimator value on the validation, do not forget the **print test accuracy**.

Notation:  One could use the out-of-bag error instead of a valid-train error.

Hint: Error = 1-Accuracy
"""

import matplotlib.pyplot as plt
# %matplotlib inline 
n_estimators=[3,5,7,9,11,15,17,20,23]

# Uses the n_estimators values above

#Draws the graph below

error_list_tr=[]
error_list_val=[]

for any_n in n_estimators:
  new_model = RandomForestClassifier(n_estimators=any_n, random_state=42)
  new_model.fit(X_tr, y_tr)
  predicted_values_tr=new_model.predict(X_tr)
  error_list_tr.append(1-accuracy_score(y_tr, predicted_values_tr))

  predicted_values_val=new_model.predict(X_val)
  error_list_val.append(1-accuracy_score(y_val, predicted_values_val))

plt.figure(figsize=(12, 6))  
plt.plot(n_estimators, error_list_val, color='blue', linestyle='dashed', marker='o',markerfacecolor='green', markersize=10, label="validation_error")
plt.plot(n_estimators, error_list_tr, color='black', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10, label="train_error")
plt.title('Validation Error - Train Error vs # of trees')  
plt.xlabel('Number of Trees')  
plt.ylabel('Accuracy') 
plt.legend(loc="upper right")
plt.show()

# Finds the Test Accuracy below

accuracy_list_test=[]

for any_n in n_estimators:
  new_model = RandomForestClassifier(n_estimators=any_n, random_state=42)
  new_model.fit(X_tr, y_tr)
  predicted_values_test=new_model.predict(X_test)
  accuracy_list_test.append(accuracy_score(y_test, predicted_values_test))

plt.figure(figsize=(12, 6))  
plt.plot(n_estimators, accuracy_list_test, color='green', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10, label="test_accuracy")
plt.title('Test Accuracy vs # of trees')  
plt.xlabel('Number of Trees')  
plt.ylabel('Accuracy') 
plt.legend(loc="upper right")
plt.show()

"""# **Question 7**

After fitting, we can use "feature_importances_" attribute to get most important parameters of random forest model we created. Please find most three important parameters. You can look at the documentation.
"""

# Solution
feature_list = list(range(df.shape[1]))
feature_importance = list(model_rf.feature_importances_)
feature_importance = [(feature,round(importance,4)) for feature,importance in zip(feature_list,feature_importance)]
feature_importance = sorted(feature_importance, key = lambda x:x[1], reverse=True)

# Sort features by importance
feature_importance = sorted(feature_importance, key = lambda x:x[1], reverse=True)

# Print top 5 features 
feature_importance[0:5]

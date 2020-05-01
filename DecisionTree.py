"""
Decision Tree Classifier

@author: Dhairya
"""

#Importing the Libraries
import numpy as np
import pandas as pd
import seaborn as sns


#Load Data From CSV File
df = pd.read_csv('drug200.csv')
df.info()                #Seeing if there is any null value present

#Convert categorical variable into dummy/indicator variables.
X = df[['Age', 'Na_to_K']]
sex = pd.get_dummies(df['Sex'], drop_first=True)
bp = pd.get_dummies(df['BP'], drop_first=True)
cholesterol = pd.get_dummies(df['Cholesterol'], drop_first=True)


#setting the independent and dependent variable
X = pd.concat( [X, sex, bp, cholesterol], axis=1)
y = df['Drug']

#Splitting data into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

#Calling the decisionTreeClassifier for the model
from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default 

#Let's make some predictions on the testing dataset
drugTree.fit(X_train,y_train)
y_pred = drugTree.predict(X_test)

#lets see the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
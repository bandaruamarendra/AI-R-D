# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os #Directory operations
from tkinter import *
from tkinter import filedialog #To Open File from local machine
import numpy as np #n-dim array operations
import pandas as pd #Reading data as Dataframe & Data accessing purpose
import matplotlib.pyplot as plt #Data visualization
import seaborn as sns #Data Visualization more effectively
from sklearn.model_selection import cross_val_score #To implement Cross-Validation and KFold validation
from sklearn.preprocessing import LabelEncoder
sns.set() #To override the styles of matplotlib
sns.set_style("darkgrid") #Styling the grid in the plot
sns.set_palette("magma") #Setting the palette style in the plot

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


#Remove warnings
import warnings
warnings.filterwarnings('ignore')

def read_data(filepath):
    if os.path.basename(filepath).endswith('.csv'):
        print('File read successfully!')
        return pd.read_csv(filepath, index_col=0)
    elif os.path.basename(filepath).endswith('.xlsx'):
        print('File read successfully!')
        return pd.read_excel(filepath)
    else:
        print('Invalid file format')
        
def convert_category(data):
    for c in data.columns:
        if data[c].dtypes == 'O' or data[c].dtypes == 'object':
            if len(data[c].value_counts()) <=10:
                lenc = LabelEncoder()
                data[c] = lenc.fit_transform(data[c])
                print(c , dict(zip(lenc.classes_, lenc.transform(lenc.classes_))))
            else:
                print('Large number of categories in %s ----- excluding converting to labels'%(c))
    return data

def select_data(data, cols):
    columns = cols
    selected_data = data.loc[:, columns]
    return selected_data

def basic_stats(data):
    return pd.concat([pd.DataFrame(data.dtypes, columns=['Data Type']).T,data.describe(exclude=['O'])],axis=0)
def plot_hist_on_selected_data(data, cols):
    columns = cols
    return sns.distplot(data[columns], bins=50, hist=True, kde=False)


def cross_validate(model, predictors, target, scoring, cv = 10):
    cv_score = cross_val_score(estimator=model, X = predictors, y = target, cv=cv, scoring=scoring)
    return cv_score #Returning 10-Fold  score


def line(Y,x_label,y_label): #Y is for Y axis values
    X = np.arange(0,len(Y),1)
    var2 = np.polyfit(X,Y,1)
    plt.plot(X,Y)
    plt.plot(X,var2[1]+ var2[0]*X, color='r')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    return var2

def display_histogram(data, cols):
    columns = cols
    return sns.distplot(data[columns], bins=50, hist=True, kde=False) #data[columns].hist(stacked=True)

def scatter_plot_display(X,Y,x_label,y_label): # X is for X axis and Y is for Y axis values
    plt.scatter(X,Y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
def display_heatmap(data):
#     return data[columns].hist()
    return sns.heatmap(data, annot=True, cmap='viridis')

window = Tk()
data = read_data(filedialog.askopenfilename(filetypes=(('Comma seperated file','.csv'),('Excel sheet','.xlsx'))))
window.destroy()
data = convert_category(data)
selected_final_data = select_data(data, cols=['ALB'])
print(basic_stats(data))
plot_hist_on_selected_data(selected_final_data, cols=selected_final_data.columns)
plt.show()
#### RFC Model ###

rfc = RandomForestClassifier(n_estimators=10,criterion='gini',max_depth=5,min_samples_split=2,min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,
    bootstrap=True,oob_score=False,n_jobs=None,random_state=21,verbose=0,warm_start=False,class_weight='balanced_subsample',
    ccp_alpha=0.0,    max_samples=None)

data.dropna(subset=['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data[['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL']],data[['Category']], test_size=0.3, random_state=21 )

print('Train data shape',X_train.shape)
print('Test data shape',X_test.shape)

print('Training Class distribution\n', y_train.Category.value_counts(), '\nTesting Class distribution\n', y_test.Category.value_counts())
rfc.fit(X_train,y_train)
predicted = rfc.predict(X_test)
print('Confusion Matrix \n', confusion_matrix(predicted,y_test))

pd.DataFrame(classification_report(predicted,y_test, output_dict=True)).T.to_csv('Classification_report_gini.csv')
print('Classifcation report written to Classification_report.csv in the working directory')
print('Feature Importance:\n', "['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL']\n", rfc.feature_importances_)

plt.figure()
line(Y = data['AST'].dropna(), x_label='---> Z', y_label='AST')
plt.show()

display_histogram(data, cols=['Category', 'ALB'])
plt.show()


scatter_plot_display(X = data['ALB'], Y = data['Category'], x_label='ALB', y_label='Category')
plt.show()

display_heatmap(data[['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL']].corr())
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('loan.csv')
# loan_prediction.csv


# In[3]:


# Loan_ID : Unique Loan ID

# Gender : Male/ Female

# Married : Applicant married (Y/N)

# Dependents : Number of dependents

# Education : Applicant Education (Graduate/ Under Graduate)

# Self_Employed : Self employed (Y/N)

# ApplicantIncome : Applicant income

# CoapplicantIncome : Coapplicant income

# LoanAmount : Loan amount in thousands of dollars

# Loan_Amount_Term : Term of loan in months

# Credit_History : Credit history meets guidelines yes or no

# Property_Area : Urban/ Semi Urban/ Rural

# Loan_Status : Loan approved (Y/N) this is the target variable 


# # 1. Display Top 5 Rows of The Dataset

# In[4]:


data.head()


# # 2. Check Last 5 Rows of The Dataset

# In[5]:


data.tail()


# # 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)

# In[6]:


data.shape


# In[7]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# # 4. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement

# In[8]:


data.info()


# # 5. Check Null Values In The Dataset
# 

# In[9]:


data.isnull().sum()


# In[10]:


data.isnull().sum()*100 / len(data)


# # 6. Handling The missing Values
# 

# In[11]:


data = data.drop('Loan_ID',axis=1)


# In[12]:


data.head(1)


# In[13]:


columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']


# In[14]:


data = data.dropna(subset=columns)


# In[15]:


data.isnull().sum()*100 / len(data)


# In[16]:


data['Self_Employed'].mode()[0]


# In[17]:


data['Self_Employed'] =data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])


# In[18]:


data.isnull().sum()*100 / len(data)


# In[19]:


data['Gender'].unique()


# In[20]:


data['Self_Employed'].unique()


# In[21]:


data['Credit_History'].mode()[0]


# In[22]:


data['Credit_History'] =data['Credit_History'].fillna(data['Credit_History'].mode()[0])


# In[23]:


data.isnull().sum()*100 / len(data)


# # 7. Handling Categorical Columns
# 

# In[24]:


data.sample(5)


# In[25]:


data['Dependents'] =data['Dependents'].replace(to_replace="3+",value='4')


# In[26]:


data['Dependents'].unique()


# In[27]:


data['Loan_Status'].unique()


# In[28]:


data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')


# In[29]:


data.head()


# # 8. Store Feature Matrix In X And Response (Target) In Vector y
# 

# In[30]:


X = data.drop('Loan_Status',axis=1)


# In[31]:


y = data['Loan_Status']


# In[32]:


y


# # 9. Feature Scaling
# 

# In[33]:


data.head()


# In[34]:


cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']


# In[35]:


from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X[cols]=st.fit_transform(X[cols])


# In[36]:


X


# # 10. Splitting The Dataset Into The Training Set And Test Set & Applying K-Fold Cross Validation
# 

# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np


# In[38]:


model_df={}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.20, random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    
    score = cross_val_score(model,X,y,cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model]=round(np.mean(score)*100,2)
    


# In[39]:


model_df


# # 11. Logistic Regression

# In[40]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model_val(model,X,y)


# # 12. SVC

# In[41]:


from sklearn import svm
model = svm.SVC()
model_val(model,X,y)


# # 13. Decision Tree Classifier

# In[42]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model_val(model,X,y)


# # 14. Random Forest Classifier

# In[43]:


from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier()
model_val(model,X,y)


# # 15. Gradient Boosting Classifier

# In[44]:


from sklearn.ensemble import GradientBoostingClassifier
model =GradientBoostingClassifier()
model_val(model,X,y)


# # 16. Hyperparameter Tuning

# In[45]:


from sklearn.model_selection import RandomizedSearchCV

Logistic Regression

# In[46]:


log_reg_grid={"C":np.logspace(-4,4,20),
             "solver":['liblinear']}


# In[47]:


rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                   param_distributions=log_reg_grid,
                  n_iter=20,cv=5,verbose=True)


# In[48]:


rs_log_reg.fit(X,y)


# In[49]:


rs_log_reg.best_score_


# In[50]:


rs_log_reg.best_params_


# # SVC

# In[51]:


svc_grid = {'C':[0.25,0.50,0.75,1],"kernel":["linear"]}


# In[52]:


rs_svc=RandomizedSearchCV(svm.SVC(),
                  param_distributions=svc_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)


# In[53]:


rs_svc.fit(X,y)


# In[54]:


rs_svc.best_score_


# In[55]:


rs_svc.best_params_


# # Random Forest Classifier
# 

# In[56]:


RandomForestClassifier()


# In[57]:


rf_grid={'n_estimators':np.arange(10,1000,10),
  'max_features':['auto','sqrt'],
 'max_depth':[None,3,5,10,20,30],
 'min_samples_split':[2,5,20,50,100],
 'min_samples_leaf':[1,2,5,10]
 }


# In[58]:


rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                  param_distributions=rf_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)


# In[59]:


rs_rf.fit(X,y)


# In[60]:


rs_rf.best_score_


# In[61]:


rs_rf.best_params_


# LogisticRegression score Before Hyperparameter Tuning: 80.48
# LogisticRegression score after Hyperparameter Tuning: 80.48 
#     
# ------------------------------------------------------
# SVC score Before Hyperparameter Tuning: 79.38
# SVC score after Hyperparameter Tuning: 80.66
#     
# --------------------------------------------------------
# RandomForestClassifier score Before Hyperparameter Tuning: 77.76
# RandomForestClassifier score after Hyperparameter Tuning: 80.66 

# # 17. Save The Model
# 

# In[62]:


X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']


# In[63]:


rf = RandomForestClassifier(n_estimators=270,
 min_samples_split=5,
 min_samples_leaf=5,
 max_features='sqrt',
 max_depth=5)


# In[64]:


rf.fit(X,y)


# In[65]:


import joblib


# In[66]:


joblib.dump(rf,'loan_status_predict')


# In[67]:


model = joblib.load('loan_status_predict')


# In[68]:


import pandas as pd
df = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2889,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1
},index=[0])


# In[69]:


df


# In[70]:


result = model.predict(df)


# In[71]:


if result==1:
    print("Loan Approved")
else:
    print("Loan Not Approved")


# # GUI

# In[72]:


# from tkinter import *
# import joblib
# import pandas as pd


# In[73]:


# def show_entry():
    
#     p1 = float(e1.get())
#     p2 = float(e2.get())
#     p3 = float(e3.get())
#     p4 = float(e4.get())
#     p5 = float(e5.get())
#     p6 = float(e6.get())
#     p7 = float(e7.get())
#     p8 = float(e8.get())
#     p9 = float(e9.get())
#     p10 = float(e10.get())
#     p11 = float(e11.get())
    
#     model = joblib.load('loan_status_predict')
#     df = pd.DataFrame({
#     'Gender':p1,
#     'Married':p2,
#     'Dependents':p3,
#     'Education':p4,
#     'Self_Employed':p5,
#     'ApplicantIncome':p6,
#     'CoapplicantIncome':p7,
#     'LoanAmount':p8,
#     'Loan_Amount_Term':p9,
#     'Credit_History':p10,
#     'Property_Area':p11
# },index=[0])
#     result = model.predict(df)
    
#     if result == 1:
#         Label(master, text="Loan approved").grid(row=31)
#     else:
#         Label(master, text="Loan Not Approved").grid(row=31)
        
    
# master =Tk()
# master.title("Loan Status Prediction Using Machine Learning")
# label = Label(master,text = "Loan Status Prediction",bg = "black",
#                fg = "white").grid(row=0,columnspan=2)

# Label(master,text = "Gender [1:Male ,0:Female]").grid(row=1)
# Label(master,text = "Married [1:Yes,0:No]").grid(row=2)
# Label(master,text = "Dependents [1,2,3,4]").grid(row=3)
# Label(master,text = "Education").grid(row=4)
# Label(master,text = "Self_Employed").grid(row=5)
# Label(master,text = "ApplicantIncome").grid(row=6)
# Label(master,text = "CoapplicantIncome").grid(row=7)
# Label(master,text = "LoanAmount").grid(row=8)
# Label(master,text = "Loan_Amount_Term").grid(row=9)
# Label(master,text = "Credit_History").grid(row=10)
# Label(master,text = "Property_Area").grid(row=11)


# e1 = Entry(master)
# e2 = Entry(master)
# e3 = Entry(master)
# e4 = Entry(master)
# e5 = Entry(master)
# e6 = Entry(master)
# e7 = Entry(master)
# e8 = Entry(master)
# e9 = Entry(master)
# e10 = Entry(master)
# e11 = Entry(master)


# e1.grid(row=1,column=1)
# e2.grid(row=2,column=1)
# e3.grid(row=3,column=1)
# e4.grid(row=4,column=1)
# e5.grid(row=5,column=1)
# e6.grid(row=6,column=1)
# e7.grid(row=7,column=1)
# e8.grid(row=8,column=1)
# e9.grid(row=9,column=1)
# e10.grid(row=10,column=1)
# e11.grid(row=11,column=1)

# Button(master,text="Predict",command=show_entry).grid()

# mainloop()


# In[ ]:





# # LOAN APPROVAL PREDICTION

# In[74]:


## Install the libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('pinfo', 'inline')


# In[75]:


df=pd.read_csv("loan.csv")
df


# In[76]:


df.head()


# In[77]:


df.tail()


# In[78]:


df.isnull().sum()


# In[79]:


## Checking the outliers

plt.figure(figsize=(12,8))
sns.boxplot(data = df)


# In[80]:


## Fill the null values of numerical datatype
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())


# In[81]:


## Fill the null values of object datatype
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])


# In[82]:


df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[83]:


df.isnull().sum()


# In[84]:


print('Number of people who took loan by gender')
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data = df, palette='Set2')


# In[85]:


print('Number of people who took loan by Married')
print(df['Married'].value_counts())
sns.countplot(x='Married',data = df, palette='Set2')


# In[86]:


numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns

corr = numeric_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot = True, cmap = 'BuPu')
plt.show()


# In[87]:


## Total Applicant Income

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# In[88]:


## Apply Log Transformation

df['ApplicantIncomelog'] = np.log(df['ApplicantIncome'] + 1)
sns.distplot(df['ApplicantIncomelog'])


# In[89]:


df['LoanAmountlog'] = np.log(df['LoanAmount'] + 1)
sns.distplot(df['LoanAmountlog'])


# In[90]:


df['Loan_Amount_Term_log'] = np.log(df['Loan_Amount_Term'] + 1)
sns.distplot(df['Loan_Amount_Term_log'])


# In[91]:


df['Total_Income_log'] = np.log(df['Total_Income'] + 1)
sns.distplot(df['Total_Income_log'])


# In[92]:


## drop unnecessary columns
cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Total_Income','Loan_ID']
df = df.drop(columns = cols, axis = 1)
df.head()


# In[93]:


## Encoding Technique : Label Encoding, One Hot Encoding

from sklearn.preprocessing import LabelEncoder
cols = ['Gender','Married','Education','Dependents','Self_Employed','Property_Area','Loan_Status']
le =  LabelEncoder()
for col in cols:
  df[col] =  le.fit_transform(df[col])


# In[94]:


## Split Independent and dependent features

X = df.drop(columns = ['Loan_Status'],axis = 1)
y = df['Loan_Status']


# In[95]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[96]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state = 42)


# In[97]:


## Logistic Regression
model1 = LogisticRegression()
model1.fit(X_train,y_train)
y_pred_model1 = model1.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_model1)


# In[98]:


accuracy*100


# In[99]:


# pickle.dump(model1, open('model.pkl', 'wb'))
# model = pickle.load(open('model.pkl', 'rb'))


# In[100]:


score = cross_val_score(model1,X,y,cv=5)
score


# In[101]:


np.mean(score)*100


# In[102]:


# ## Logistic Regression
# model1 = LogisticRegression()
# model1.fit(X_train,y_train)
# y_pred_model1 = model1.predict(X_test)
# accuracy = accuracy_score(y_test,y_pred_model1)


# In[103]:


## Decision Tree Classifier

model2 = DecisionTreeClassifier()
model2.fit(X_train,y_train)
y_pred_model2 = model2.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_model2)
print("Accuracy score of Decision Tree: ", accuracy*100)


# In[104]:


score = cross_val_score(model2,X,y,cv=5)
print("Cross Validation score of Decision Tree: ",np.mean(score)*100)


# In[105]:


from sklearn.tree import DecisionTreeClassifier
dt_clf= DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)


# In[106]:


from sklearn import metrics
y_pred=dt_clf.predict(X_test)
print("acc of is Decisiontree is  ",metrics.accuracy_score(y_pred,y_test))


# In[107]:


y_pred


# In[108]:


score = cross_val_score(model2,X,y,cv=5)
print("Cross Validation score of Decision Tree: ",np.mean(score)*100)


# In[109]:


## Random Forest Classifier
model3 = RandomForestClassifier()
model3.fit(X_train,y_train)
y_pred_model3 = model3.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_model3)
print("Accuracy score of Random Forest: ", accuracy*100)


# In[110]:


from sklearn.metrics import classification_report
def generate_classification_report(model_name,y_test,y_pred):
  report = classification_report(y_test,y_pred)
  print(f"Classification Report For {model_name}:\n{report}\n")

generate_classification_report(model1,y_test,y_pred_model1)
generate_classification_report(model2,y_test,y_pred_model2)
generate_classification_report(model3,y_test,y_pred_model3)
# generate_classification_report(model4,y_test,y_pred_model4)


# In[111]:


pip install -U imbalanced-learn


# In[112]:


from imblearn.over_sampling import RandomOverSampler


# In[113]:


oversample = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversample.fit_resample(X,y)

df_resampled = pd.concat([pd.DataFrame(X_resampled,columns=X.columns),pd.Series(y_resampled,name="Loan_status")],axis=1)


# In[114]:


X_resampled


# In[115]:


y_resampled


# In[116]:


y_resampled.value_counts()


# In[117]:


X_resampled_train, X_resampled_test, y_resampled_train, y_resampled_test = train_test_split(X_resampled,y_resampled,test_size = 0.25,random_state=42)


# In[118]:


## Logistic Regression
model1 = LogisticRegression()
model1.fit(X_resampled_train,y_resampled_train)
y_pred_model1 = model1.predict(X_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model1)
accuracy*100


# In[119]:


## Decision Tree Classifier

model2 = DecisionTreeClassifier()
model2.fit(X_resampled_train,y_resampled_train)
y_pred_model2 = model2.predict(X_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model2)
print("Accuracy score of Decision Tree: ", accuracy*100)


# In[120]:


## Random Forest Classifier
model3 = RandomForestClassifier()
model3.fit(X_resampled_train,y_resampled_train)
y_pred_model3 = model3.predict(X_resampled_test)
accuracy = accuracy_score(y_resampled_test,y_pred_model3)
print("Accuracy score of Random Forest: ", accuracy*100)


# In[121]:


from sklearn.metrics import classification_report

def generate_classification_report(model_name,y_test,y_pred):
  report = classification_report(y_test,y_pred)
  print(f"Classification Report For {model_name}:\n{report}\n")

generate_classification_report(model1,y_resampled_test,y_pred_model1)
generate_classification_report(model2,y_resampled_test,y_pred_model2)
generate_classification_report(model3,y_resampled_test,y_pred_model3)
# generate_classification_report(model4,y_resampled_test,y_pred_model4)


# In[ ]:





# In[ ]:





# #  BANK CUSTOMER CHURN PREDICTION

# In[122]:


import pandas as pd
import numpy as np

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns
sns.set_style('darkgrid')

# store elements as dictionary keys and their counts as dictionary values
from collections import Counter


# In[123]:


# Load the dataset
df = pd.read_csv("loan.csv")
print(f"Dataframe dimensions: {df.shape}")
df.head()


# In[124]:


df.info()


# In[125]:


# List number of unique customer IDs
df.Loan_ID.nunique()


# In[126]:


df.duplicated().sum()


# In[127]:


# Drop unused features
df.drop(['Loan_ID', 'ApplicantIncome', 'Self_Employed'], axis=1, inplace=True)
print(f"Dataframe dimensions: {df.shape}")
df.head()


# In[128]:


# Plot histogram grid
df.hist(figsize=(4,4))

plt.show()


# In[129]:


df.describe()


# In[130]:


# Summarize categorical features
df.describe(include=['object'])


# In[131]:


# Bar plot for "Gender"
plt.figure(figsize=(4,4))
df['Gender'].value_counts().plot.bar(color=['b', 'g'])
plt.ylabel('Count')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.show()

# Display count of each class
Counter(df.Gender)


# In[132]:


# Bar plot for "Geography"
plt.figure(figsize=(6,4))
df['Education'].value_counts().plot.bar(color=['b', 'g', 'r'])
plt.ylabel('Count')
plt.xlabel('Education')
plt.xticks(rotation=0)
plt.show()

# Display count of each class
Counter(df.Education)


# In[133]:


# Segment "Exited" by gender and display the frequency and percentage within each class
grouped = df.groupby('Gender')['Education'].agg(Count='value_counts')
grouped


# In[134]:


# Reorganize dataframe for plotting count
dfgc = grouped
dfgc = dfgc.pivot_table(values='Count', index='Gender', columns=['Education'])
dfgc


# In[135]:


# Calculate percentage within each class
dfgp = grouped.groupby(level=[0]).apply(lambda g: round(g * 100 / g.sum(), 2))
dfgp.rename(columns={'Count': 'Percentage'}, inplace=True)
dfgp


# In[136]:


# Reorganize dataframe for plotting percentage
dfgp = dfgp.pivot_table(values='Percentage', index='Gender', columns=['Education'])
dfgp


# In[137]:


# Churn distribution by gender, count + percentage

labels= ['Stays', 'Exits']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

dfgc.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax1)
ax1.legend(labels)
ax1.set_title('Churn Risk per Gender (Count)', fontsize=14, pad=10)
ax1.set_ylabel('Count',size=12)
ax1.set_xlabel('Gender', size=12)


dfgp.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax2)
ax2.legend(labels)
ax2.set_title('Churn Risk per Gender (Percentage)', fontsize=14, pad=10)
ax2.set_ylabel('Percentage',size=12)
ax2.set_xlabel('Gender', size=12)

plt.show()


# In[138]:


# Segment "Exited" by geography and display the frequency and percentage within each class
grouped = df.groupby('Gender')['Education'].agg(Count='value_counts')
grouped


# In[139]:


# Reorganize dataframe for plotting count
dfgeoc = grouped
dfgeoc = dfgeoc.pivot_table(values='Count', index='Gender', columns=['Education'])
dfgeoc


# In[140]:


# Calculate percentage within each class
dfgeop = grouped.groupby(level=[0]).apply(lambda g: round(g * 100 / g.sum(), 2))
dfgeop.rename(columns={'Count': 'Percentage'}, inplace=True)
dfgeop


# In[141]:


# Reorganize dataframe for plotting percentage
dfgeop = dfgeop.pivot_table(values='Percentage', index='Gender', columns=['Education'])
dfgeop


# In[142]:


# Churn distribution by geography, count + percentage

labels= ['Stays', 'Exits']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

dfgeoc.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax1)
ax1.legend(labels)
ax1.set_title('Churn Risk per Gender (Count)', fontsize=14, pad=10)
ax1.set_ylabel('Count',size=12)
ax1.set_xlabel('Gender', size=12)


dfgeop.plot(kind='bar',
          color=['g', 'r'],
          rot=0, 
          ax=ax2)
ax2.legend(labels)
ax2.set_title('Churn Risk per Gender (Percentage)', fontsize=14, pad=10)
ax2.set_ylabel('Percentage',size=12)
ax2.set_xlabel('Gender', size=12)

plt.show()


# In[143]:


# Define our target variable
y = df.Education


# In[144]:


# Function to display count and percentage per class of target feature
def class_count(a):
    counter=Counter(a)
    kv=[list(counter.keys()),list(counter.values())]
    dff = pd.DataFrame(np.array(kv).T, columns=['Education','Count'])
    dff['Count'] = dff['Count'].astype('int64')
    dff['%'] = round(dff['Count'] / a.shape[0] * 100, 2)
    return dff.sort_values('Count',ascending=False)


# In[145]:


# Let's use the function
dfcc = class_count(y)
dfcc


# In[146]:


import warnings


# In[147]:


import pickle


# In[148]:


warnings.filterwarnings("ignore")


# In[ ]:





# In[ ]:





# #  CREDIT CARD FRAUD DETECTION

# In[150]:


# Import basic libraries 
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import ADASYN 
from collections import Counter
import seaborn as sn

# plot functions
# import plot_functions as pf

# scikit packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn import metrics

# settings
get_ipython().run_line_magic('matplotlib', 'inline')
sn.set_style("dark")
sn.set_palette("colorblind")


# In[151]:


df = pd.read_csv("loan.csv")


# In[152]:


df.head()


# In[153]:


print('The dataset contains {0} rows and {1} columns.'.format(df.shape[0], df.shape[1]))


# In[154]:


df.info()


# In[155]:


print('Normal transactions count: ', df['Gender'].value_counts().values[0])
print('Fraudulent transactions count: ', df['Gender'].value_counts().values[1])


# In[ ]:





# # LOAN STATUS PREDICTION

# In[156]:


obj = (data.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))


# In[158]:


# Dropping Loan_ID column 
# data.drop(['Loan_ID'],axis=1,inplace=True)


# In[159]:


obj = (data.dtypes == 'object') 
object_cols = list(obj[obj].index) 
plt.figure(figsize=(18,36)) 
index = 1
  
for col in object_cols: 
  y = data[col].value_counts() 
  plt.subplot(11,4,index) 
  plt.xticks(rotation=90) 
  sns.barplot(x=list(y.index), y=y) 
  index +=1


# In[160]:


# Import label encoder 
from sklearn import preprocessing 
    
# label_encoder object knows how  
# to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
obj = (data.dtypes == 'object') 
for col in list(obj[obj].index): 
  data[col] = label_encoder.fit_transform(data[col])


# In[161]:


obj = (data.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))


# In[162]:


plt.figure(figsize=(12,6)) 
  
sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f', 
            linewidths=2,annot=True)


# In[163]:


sns.catplot(x="Gender", y="Married", 
            hue="Loan_Status",  
            kind="bar",  
            data=data)


# In[164]:


for col in data.columns: 
  data[col] = data[col].fillna(data[col].mean())  
    
data.isna().sum()


# In[165]:


from sklearn.model_selection import train_test_split 
  
X = data.drop(['Loan_Status'],axis=1) 
Y = data['Loan_Status'] 
X.shape,Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1) 
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[169]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
  
from sklearn import metrics 
  
knn = KNeighborsClassifier(n_neighbors=3) 
rfc = RandomForestClassifier(n_estimators = 9, 
                             criterion = 'entropy', 
                             random_state =9) 
svc = SVC() 
lc = LogisticRegression() 
  
# making predictions on the training set 
for clf in (rfc, knn, svc,lc): 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_train) 
    print("Accuracy score of ", 
          clf.__class__.__name__, 
          "=",100*metrics.accuracy_score(Y_train,  
                                         Y_pred))


# In[170]:


for clf in (rfc, knn, svc,lc): 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test) 
    print("Accuracy score of ", 
          clf.__class__.__name__,"=", 
          100*metrics.accuracy_score(Y_test, 
                                     Y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





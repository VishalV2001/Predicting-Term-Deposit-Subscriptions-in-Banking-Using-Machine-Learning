#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ##### Read the data

# In[2]:


#create data drame to read data set
df = pd.read_csv('/Users/vishalvadhel/Desktop/AI/machine learning/ml coursework/term prediction/term prediction.csv') 


# In[3]:


df


# In[4]:


# check the df structure
df.info()


# In[5]:


# find number of rows and column
df.shape


# In[6]:


# describe df numerical columns
df.describe()


# In[7]:


for col in df.select_dtypes(include='object').columns:
    print(col)
    print(df[col].unique())


# # Data Analysis

# ##### Find Missing Values

# In[8]:


# find missing values
features_na = [features for features in df.columns if df[features].isnull().sum() > 0]
for feature in features_na:
    print(feature, np.round(df[feature].isnull().mean(), 4),  ' % missing values')
else:
    print("No missing value found")


# #### Find Features with One Value

# In[9]:


for column in df.columns:
    print(column,df[column].nunique())


# #### Explore the Categorical Features

# In[10]:


categorical_features=[feature for feature in df.columns if ((df[feature].dtypes=='O') & (feature not in ['deposit']))]
categorical_features


# In[11]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))


# #### Find Categorical Feature Distribution

# In[12]:


#check count based on categorical features
plt.figure(figsize=(15,80), facecolor='white')
plotnumber =1
for categorical_feature in categorical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.countplot(y=categorical_feature,data=df)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber+=1
plt.show()


# #### Relationship between Categorical Features and Label

# In[13]:


#check target label split over cate
#gorical features
#Find out the relationship between categorical variable and dependent variable
for categorical_feature in categorical_features:
    sns.catplot(x='y', col=categorical_feature, kind='count', data= df)
plt.show()


# In[14]:


#Check target label split over categorical features and find the count
for categorical_feature in categorical_features:
    print(df.groupby(['y',categorical_feature]).size())


# #### Explore the Numerical Features

# In[15]:


# list of numerical variables
numerical_features = [feature for feature in df.columns if ((df[feature].dtypes != 'O') & (feature not in ['deposit']))]
print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
df[numerical_features].head()


# #### Find Discrete Numerical Features

# In[16]:


discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# #### Find Continous Numerical Features

# In[17]:


continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['y']]
print("Continuous feature Count {}".format(len(continuous_features)))


# #### Distribution of Continous Numerical Features

# In[18]:


#plot a univariate distribution of continues observations
plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for continuous_feature in continuous_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.histplot(df[continuous_feature])
    plt.xlabel(continuous_feature)
    plotnumber+=1
plt.show()


# #### Relation between Continous numerical Features and Labels

# In[19]:


#boxplot to show target distribution with respect numerical features
plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for feature in continuous_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(x="y", y= df[feature], data=df)
    plt.xlabel(feature)
    plotnumber+=1
plt.show()


# #### Find Outliers in numerical features

# In[20]:


#boxplot on numerical features to find outliers
plt.figure(figsize=(15, 45), facecolor='white')
plotnumber = 1

for numerical_feature in numerical_features:
    ax = plt.subplot(12, 3, plotnumber)
    sns.boxplot(x=df[numerical_feature])
    #sns.boxplot(x=df[numerical_feature], orient='h') # Use orient='h' for horizontal boxplot
    plt.xlabel(numerical_feature)
    plotnumber += 1

plt.show()


# #### Explore the Correlation between numerical features

# In[21]:


## Checking for correlation
cor_mat=df.corr()
fig = plt.figure(figsize=(12,5))
sns.heatmap(cor_mat,annot=True)


# #### Check the Data set is balanced or not based on target values in classification

# In[22]:


sns.countplot(x='y',data=df)
plt.show()


# In[23]:


df['y'].groupby(df['y']).count()


# # Feature Engineering

# In[24]:


df2=df.copy()


# In[25]:


df2.head()


# In[26]:


df2.shape


# #### Drop unwanted Features

# In[27]:


#defaut features does not play imp role
df2.groupby(['y','default']).size()


# #### Handle Categorical Features

# In[28]:


df2.drop(['default'],axis=1, inplace=True)


# In[29]:


df2.groupby(['y','pdays']).size()


# #### Handle Feature Scalling

# In[30]:


# drop pdays as it has -1 value for around 40%+ 
df2.drop(['pdays'],axis=1, inplace=True)


# #### Remove Outliers

# In[31]:


# remove outliers in feature age...
df2.groupby('age',sort=True)['age'].count()
# these can be ignored and values lies in between 18 to 95


# In[32]:


# remove outliers in feature balance...
df2.groupby(['y','balance'],sort=True)['balance'].count()
# these outlier should not be remove as balance goes high, client show interest on deposit


# In[33]:


# remove outliers in feature duration...
df2.groupby(['y','duration'],sort=True)['duration'].count()
# these outlier should not be remove as duration goes high, client show interest on deposit


# In[34]:


# remove outliers in feature campaign...
df2.groupby(['y','campaign'],sort=True)['campaign'].count()


# In[35]:


df3 = df2[df2['campaign'] < 33]


# In[36]:


df3.groupby(['y','campaign'],sort=True)['campaign'].count()


# In[37]:


# remove outliers in feature previous...
df3.groupby(['y','previous'],sort=True)['previous'].count()


# In[38]:


df4 = df3[df3['previous'] < 31]


# In[39]:


cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
for col in  cat_columns:
    df4 = pd.concat([df4.drop(col, axis=1),pd.get_dummies(df4[col], prefix=col, prefix_sep='_',drop_first=True, dummy_na=False)], axis=1)


# In[40]:


bool_columns = ['housing', 'loan', 'y']
for col in  bool_columns:
    df4[col+'_new']=df4[col].apply(lambda x : 1 if x == 'yes' else 0)
    df4.drop(col, axis=1, inplace=True)


# In[41]:


df4.head()


# # Split Dataset into Training set and Test set

# In[42]:


X = df4.drop(['y_new'],axis=1)
y = df4['y_new']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


# In[43]:


len(X_train)


# In[44]:


len(X_test)


# # Model Selection

# In[45]:


pip install xgboost


# In[46]:


# will try to use below two models that are RandomForestClassifier and XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[47]:


# Cross-validation scores for RandomForestClassifier
rf_model_score = cross_val_score(estimator=RandomForestClassifier(), X=X_train, y=y_train, cv=5)
print("RandomForestClassifier Cross-Validation Scores:")
print(rf_model_score)
print("Mean Accuracy:", rf_model_score.mean())
print("\n")


# In[48]:


# Cross-validation scores for XGBClassifier
xgb_model_score = cross_val_score(estimator=XGBClassifier(), X=X_train, y=y_train, cv=5)
print("XGBClassifier Cross-Validation Scores:")
print(xgb_model_score)
print("Mean Accuracy:", xgb_model_score.mean())
print("\n")


# In[49]:


#create param
model_param = {
    'RandomForestClassifier':{
        'model':RandomForestClassifier(),
        'param':{
            'n_estimators': [10, 50, 100, 130], 
            'criterion': ['gini', 'entropy'],
            'max_depth': range(2, 4, 1), 
            'max_features': ['auto', 'log2']
        }
    },
    'XGBClassifier':{
        'model':XGBClassifier(objective='binary:logistic'),
        'param':{
           'learning_rate': [0.5, 0.1, 0.01, 0.001],
            'max_depth': [3, 5, 10, 20],
            'n_estimators': [10, 50, 100, 200]
        }
    }
}


# In[50]:


#gridsearch
scores =[]
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(estimator=mp['model'],param_grid=mp['param'],cv=5,return_train_score=False)
    model_selection.fit(X,y)
    scores.append({
        'model': model_name,
        'best_score': model_selection.best_score_,
        'best_params': model_selection.best_params_
    })


# In[51]:


scores


# # Model Building

# In[52]:


# RandomForest Model building
model_rf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
rf_accuracy = model_rf.score(X_test, y_test)
rf_training_accuracy = model_rf.score(X_train, y_train)
print(f"RandomForest Training Accuracy: {rf_training_accuracy}")
print(f"RandomForest Testing Accuracy: {rf_accuracy}")


# In[53]:


#xg boots model building
model_xgb = XGBClassifier(objective='binary:logistic',learning_rate=0.1,max_depth=10,n_estimators=100)
model_xgb.fit(X_train,y_train)
xgb_accuracy = model_xgb.score(X_test,y_test)
xgb_training_accuracy = model_xgb.score(X_train, y_train) 
print(f"XG Boost Training Accuracy: {xgb_training_accuracy}")
print(f"XG Boost Testing Accuracy: {xgb_accuracy}")


# In[54]:


#get feature importances from xgb boost
headers = ["name", "score"]
values = sorted(zip(X_train.columns, model_xgb.feature_importances_), key=lambda x: x[1] * -1)
xgb_feature_importances = pd.DataFrame(values, columns = headers)

#get feature Importances for RandomForest
headers_rf = ["name", "score"]
values_rf = sorted(zip(X_train.columns, model_rf.feature_importances_), key=lambda x: x[1] * -1)
rf_feature_importances = pd.DataFrame(values_rf, columns=headers_rf)


# In[55]:


# Plot Feature Importances for RandomForest
fig = plt.figure(figsize=(15,7))
x_pos_rf = np.arange(0, len(rf_feature_importances))
plt.bar(x_pos_rf, rf_feature_importances['score'])
plt.xticks(x_pos_rf, rf_feature_importances['name'])
plt.xticks(rotation=90)
plt.title('Feature importances (Random Forest)')
plt.show()


# In[56]:


#plot feature importances for xg boost
fig = plt.figure(figsize=(15,7))
x_pos = np.arange(0, len(xgb_feature_importances))
plt.bar(x_pos, xgb_feature_importances['score'])
plt.xticks(x_pos, xgb_feature_importances['name'])
plt.xticks(rotation=90)
plt.title('Feature importances (XGB)')
plt.show()


# In[57]:


from sklearn.metrics import confusion_matrix
# Confusion Matrix for RandomForest
cm_rf = confusion_matrix(y_test, model_rf.predict(X_test))
print("\nConfusion Matrix for RandomForest:")
print(cm_rf)

#Confusion Matrix for xg boost
from sklearn.metrics import confusion_matrix
cm_xgb = confusion_matrix(y_test,model_xgb.predict(X_test))
print("confusion matrix for XG Boost:")
print(cm_xgb)


# In[58]:


from matplotlib import pyplot as plt
import seaborn as sn

#plot the graph random forest
sn.heatmap(cm_rf, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True Value')
print("the graph Random Forest")
plt.show()

#plot the graph xg boost
sn.heatmap(cm_xgb, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True Value')
print("the graph XG Boost")
plt.show()


# In[59]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

predictions_rf = model_rf.predict(X_test)
predictions_xgb = model_xgb.predict(X_test)

# Calculate metrics for RandomForest
accuracy_rf = accuracy_score(y_test, predictions_rf)
precision_rf = precision_score(y_test, predictions_rf)
recall_rf = recall_score(y_test, predictions_rf)
f1_rf = f1_score(y_test, predictions_rf)

# Calculate metrics for XGBoost
accuracy_xgb = accuracy_score(y_test, predictions_xgb)
precision_xgb = precision_score(y_test, predictions_xgb)
recall_xgb = recall_score(y_test, predictions_xgb)
f1_xgb = f1_score(y_test, predictions_xgb)

print("\nMetrics for RandomForest:")
print(f"Accuracy: {accuracy_rf}, Precision: {precision_rf}, Recall: {recall_rf}, F1 Score: {f1_rf}")

# Print the metrics
print("Metrics for XGBoost:")
print(f"Accuracy: {accuracy_xgb}, Precision: {precision_xgb}, Recall: {recall_xgb}, F1 Score: {f1_xgb}")


# In[ ]:





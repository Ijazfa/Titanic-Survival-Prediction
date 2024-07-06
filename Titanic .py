#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("C:/Users/Ijaz khan/Downloads/Titanic-Dataset.csv")


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe(include='all')


# In[9]:


df.isnull().sum()


# In[10]:


print(df.index)
print(df.columns)


# In[11]:


drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']

df.drop(drop, axis=1, inplace = True)


# In[12]:


df.head()


# In[13]:


df[df['SibSp']==8]


# In[14]:


df = df.drop(df[df['SibSp']==8].index)
df.head(20)


# In[15]:


df.groupby(['Survived','SibSp','Pclass'])['Age'].mean()


# In[16]:


df['Age'] = df['Age'].fillna(df.groupby(['Survived','SibSp','Pclass'])['Age'].transform('mean'))


# In[17]:


df[df.isnull().any(axis=1)]


# In[18]:


df = df.dropna().reset_index(drop = True)
df.head()


# In[19]:


df.describe()


# In[20]:


plt.figure(figsize=(10,9))
for i,col in enumerate(['Age','Fare']):
    plt.subplot(2,1,i+1)
    sns.histplot(data=df, x= col, kde=True, color = 'red')


# In[21]:


sns.countplot(x = df['Embarked'])
plt.show()


# In[22]:


sns.countplot(x=df["Pclass"])
plt.show()


# In[23]:


sns.countplot(x = df['Sex'])


# In[24]:


sns.countplot(x =df['Survived'])


# In[25]:


sns.countplot(x = df['SibSp'])
plt.title('No of siblings / spouses aboard in the Titanic')
plt.xlabel('No of siblings / spouses')


# In[26]:


sns.countplot(x=df['Parch'])
plt.title('No of parents / children aboard in the Titanic')
plt.xlabel('No of parents / children')


# In[27]:


df.groupby('Survived')['Age'].mean()


# In[28]:


df.groupby('Survived')['Age'].mean().plot(kind = 'bar')
plt.xticks(rotation = 0)
plt.show()


# In[29]:


df.groupby('Survived')['Fare'].mean()


# In[30]:


df.groupby('Survived')['Fare'].mean().plot(kind = 'bar')
plt.xticks(rotation = 0)
plt.show()


# In[31]:


sns.countplot(x =df['Survived'], hue=df['Pclass'])
plt.legend(bbox_to_anchor=(1.5,0.8), title='Passenger ticket class')
plt.show()


# In[32]:


sns.scatterplot(x= df['Fare'], y=df['Age'], hue =df['Survived'])
plt.legend(bbox_to_anchor = (1.2,0.8), title = 'Survived')
plt.show()


# In[33]:


sns.countplot(x = df['Survived'], hue = df['Sex'])
plt.legend(bbox_to_anchor = (1.2,0.8),title = 'Sex')
plt.show()


# In[34]:


sns.countplot(x = df['Survived'], hue = df['SibSp'])
plt.legend(bbox_to_anchor = (1.2,0.8), title = 'Siblings/spouses')
plt.show()


# In[35]:


sns.countplot(x=df['Survived'], hue = df['Parch'])
plt.legend(bbox_to_anchor = (1.2,0.8), title = "parents/children")
plt.show()


# In[ ]:





# In[36]:


# importing libreries to transform the data

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split,cross_val_score


# In[37]:


def one_hot_encoding(df=None):
    dums = pd.get_dummies(df[['Sex', 'Embarked']], dtype=int)
    dums_data = pd.concat([dums,df], axis =1).drop(columns=['Sex','Embarked']).reset_index(drop=True)
    return dums_data


# In[38]:


model_data = one_hot_encoding(df)


# In[39]:


model_data


# In[40]:


plt.figure(figsize=(12,12))
sns.heatmap(model_data.corr(), annot=True, cmap='coolwarm')


# In[41]:


#spliting the data to train and test

x = model_data.drop(columns = 'Survived')
y = model_data['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=0)


# In[42]:


#normalizing the data using min max scaler

scaler = MinMaxScaler()
scaler.fit(x[['Fare', 'Age']])
x[['Fare','Age']] = scaler.transform(x[['Fare', 'Age']])


# In[43]:


gbc_model = GradientBoostingClassifier()


# In[44]:


gbc_model.fit(x_train,y_train)


# In[45]:


#evaluating Gradient Boosting Classifier

gbc_model.score(x_train,y_train)


# In[46]:


gbc_model.score(x_test,y_test)


# In[47]:


score = cross_val_score(gbc_model,x,y,cv=10)
avg = np.mean(score)
print(f'cross val score of gradient boosting : {score}')
print(f'avarage cross val score of gradient boosting : {avg}')


# In[48]:


y_predicted = gbc_model.predict(x_test)
y_proba = gbc_model.predict_proba(x_test)


# In[49]:


print(classification_report(y_test,y_predicted))


# In[50]:


cm = confusion_matrix(y_test,y_predicted)
sns.heatmap(cm, annot =True, fmt ='d')
plt.xlabel('Predict')
plt.ylabel('Truth')
plt.title('Gradient Boosting Confusion Matrix')
plt.show()


# In[62]:


new_data = pd.DataFrame({'Pclass':[3],'Sex':['male'],'Age':[25],'SibSp':[0],'Parch':[1],'Fare':[25],
                        'Embarked':['S']})


# In[63]:


#transfprming data

new_data = one_hot_encoding(new_data)
new_data = new_data.reindex(columns= x.columns, fill_value=0)
new_data[['Fare','Age']] = scaler.transform(new_data[['Fare','Age']])


# In[64]:


new_data


# In[65]:


prediction= gbc_model.predict(new_data)
if prediction == 0:
    print("Passenger didn't survive")
else:
    print("Passenger survived")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





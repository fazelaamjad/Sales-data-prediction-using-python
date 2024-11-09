#!/usr/bin/env python
# coding: utf-8

# # Step 1: Import Libraries and Load the Dataset

# In[1]:


# Import necessary libraries
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# Load the dataset
df = pd.read_csv('Advertising.csv')

df = df.drop(['Unnamed: 0'], axis=1)
df.head()


# # Step 2: Data Exploration and Cleaning

# In[3]:


print(df.columns)


# In[4]:


# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())



# In[5]:


# Display basic statistics
df.describe()


# # Step 3: Exploratory Data Analysis (EDA)

# In[6]:


# Scatter plots to show relationship between different variables and sales
plt.figure(figsize=(12, 8))
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.7)
plt.show()



# # Step 4: Feature Engineering and Selection

# In[7]:


# Define features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


# # Step 5: Train-Test Split

# In[8]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Step 6: Model Training

# In[9]:


# Initialize and train the model
model = RandomForestRegressor(random_state=42)
# Perform Cross-Validation (using 5 folds)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-Validation R-squared scores: {cv_scores}")
print(f"Average Cross-Validation R-squared: {cv_scores.mean()}")
model.fit(X_train, y_train)



# In[10]:


# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[11]:


# Feature Importance
feature_importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})


# In[12]:


# Sort by importance and plot
importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()


# In[14]:


from joblib import dump, load
# Test saving the model
try:
    dump(model, 'test_model.joblib')
    print("Model saved successfully.")
except Exception as e:
    print("Error saving the model:", e)

# Test loading the model
try:
    loaded_model = load('test_model.joblib')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", e)



# In[ ]:





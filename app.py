import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#load the dataset
cal=fetch_california_housing()
df=pd.DataFrame(data=cal.data, columns=cal.feature_names)
df['price']=cal.target
df.head()

#Title of the app
st.title("california House Price Prediction for XYZ Brokerage Company")

#Data overview
st.dataframe(df.head(10))

#split the data into train and test
X=df.drop(['price'],axis=1)   #input variables
Y=df['price'] #Target Variable
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.3,random_state=42)

#Standardize the data
scaler=StandardScaler()
X_train_sc=scaler.fit_transform(X_train)
X_test_sc=scaler.transform(X_test)

#Model Selection
st.subheader("## Select a Model")

model=st.selectbox("Choose a model",["Linear Regression","Ridge","Lasso","ElasticNet"])

models={"Linear Regression":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(),
        "ElasticNet":ElasticNet()}

#Train the selected model
selected_model=models[model]

#Train the model
selected_model.fit(X_train_sc,Y_train)

#predict the values
Y_pred=selected_model.predict(X_test_sc)

#evaluate the model
test_mse=mean_squared_error(Y_test,Y_pred)
test_mae=mean_absolute_error(Y_test,Y_pred)
test_rmse=np.sqrt(test_mse)
test_r2=r2_score(Y_test,Y_pred)

#Display the metrics for selected model
st.write("Test MSE",test_mse)
st.write("Test MAE",test_mae)
st.write("Test RMSE",test_rmse)
st.write("Test R2",test_r2)

#Promt the user to enter the input values
st.write("Enter the input to predict the house price")

user_input={}

for feature in X.columns:
    user_input[feature]=st.number_input(feature) 

#Convert the dictionary to a dataframe
user_input_df=pd.DataFrame([user_input])

#Scale the user input
user_input_sc=scaler.transform(user_input_df)

#predict the house value
predicted_price = selected_model.predict(user_input_sc)

#display the predicted house price
st.write(f"Predicted House Price is{predicted_price[0]*100000}")
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
pipe = pickle.load(open('best_model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Sales Predictor")

Item_Type = st.selectbox('Item_Type', df['Item_Type'].unique())

Item_Fat_Content = st.selectbox('Item_Fat_Content', df['Item_Fat_Content'].unique())

Item_Visibility = st.selectbox('Item_Visibility', df['Item_Visibility'].unique())

Item_MRP = st.number_input("Enter Item MRP", min_value=0.0)

Item_Weight = st.number_input("Enter Item Weight (in grams)", min_value=0.0)

Outlet_Establishment_Year = st.selectbox('Outlet_Establishment_Year', df['Outlet_Establishment_Year'].unique())

Outlet_Size = st.selectbox(	'Outlet_Size', df['Outlet_Size'].unique())

Outlet_Location_Type = st.selectbox('Outlet_Location_Type', df['Outlet_Location_Type'].unique())

Outlet_Type = st.selectbox('Outlet_Type', df['Outlet_Type'].unique())

if st.button('Predict Price'):

    query = pd.DataFrame({
        'Item_Weight': [Item_Weight],
        'Item_Fat_Content': [Item_Fat_Content],
        'Item_Visibility': [Item_Visibility],
        'Item_Type': [Item_Type],
        'Item_MRP': [Item_MRP],
        'Outlet_Establishment_Year': [Outlet_Establishment_Year],
        'Outlet_Size': [Outlet_Size],
        'Outlet_Location_Type': [Outlet_Location_Type],
        'Outlet_Type': [Outlet_Type]
    })

    predicted_price = pipe.predict(query)

    st.title("The predicted price of this configuration is " + str(int(np.exp(predicted_price[0]))))


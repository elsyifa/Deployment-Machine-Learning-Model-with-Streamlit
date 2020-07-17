#import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import time
import matplotlib .pyplot as plt
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler



#import data
data=pd.read_csv('MALTENG4g.csv')
data2=pd.read_csv('BANDUNG4g.csv')
data3=pd.read_csv('kota_bogor4g.csv')
data4=pd.read_csv('kota_medan4g.csv')
data5=pd.read_csv('kota_pontianak4g.csv')


#set_index
#data.set_index('week_date', inplace=True)
#data.index=pd.to_datetime(data.index)


#input the numbers
week 	 = st.sidebar.slider("How many weeks you want to forecast?",1,12,1)


#splitting data
#train = data.iloc[:96]
#test  = data.iloc[96:]

#data scaling
scaler=MinMaxScaler()
scaler.fit(data[['total_payload4G']])
scaled_data=scaler.transform(data[['total_payload4G']])

#data2 scaling
scaler2=MinMaxScaler()
scaler2.fit(data2[['total_payload4G']])
scaled_data2=scaler2.transform(data2[['total_payload4G']])

#data3 scaling
scaler3=MinMaxScaler()
scaler3.fit(data3[['total_payload4G']])
scaled_data3=scaler3.transform(data3[['total_payload4G']])

#data4 scaling
scaler4=MinMaxScaler()
scaler4.fit(data4[['total_payload4G']])
scaled_data4=scaler4.transform(data4[['total_payload4G']])


#data5 scaling
scaler5=MinMaxScaler()
scaler5.fit(data5[['total_payload4G']])
scaled_data5=scaler5.transform(data5[['total_payload4G']])


#convert scaled data to dataframe
scaled_data_df =pd.DataFrame(data=scaled_data, index=data.week_date, columns=['total_payload4G'])
scaled_data_df['week_date']=scaled_data_df.index
scaled_data_df=scaled_data_df.reset_index(drop=True)
scaled_data_df=scaled_data_df[['week_date', 'total_payload4G']]

#convert scaled data2 to dataframe
scaled_data_df2 =pd.DataFrame(data=scaled_data2, index=data2.week_date, columns=['total_payload4G'])
scaled_data_df2['week_date']=scaled_data_df2.index
scaled_data_df2=scaled_data_df2.reset_index(drop=True)
scaled_data_df2=scaled_data_df2[['week_date', 'total_payload4G']]


#convert scaled data3 to dataframe
scaled_data_df3 =pd.DataFrame(data=scaled_data3, index=data3.week_date, columns=['total_payload4G'])
scaled_data_df3['week_date']=scaled_data_df3.index
scaled_data_df3=scaled_data_df3.reset_index(drop=True)
scaled_data_df3=scaled_data_df3[['week_date', 'total_payload4G']]


#convert scaled data4 to dataframe
scaled_data_df4 =pd.DataFrame(data=scaled_data4, index=data4.week_date, columns=['total_payload4G'])
scaled_data_df4['week_date']=scaled_data_df4.index
scaled_data_df4=scaled_data_df4.reset_index(drop=True)
scaled_data_df4=scaled_data_df4[['week_date', 'total_payload4G']]


#convert scaled data5 to dataframe
scaled_data_df5 =pd.DataFrame(data=scaled_data5, index=data5.week_date, columns=['total_payload4G'])
scaled_data_df5['week_date']=scaled_data_df5.index
scaled_data_df5=scaled_data_df5.reset_index(drop=True)
scaled_data_df5=scaled_data_df5[['week_date', 'total_payload4G']]


#checking the data
st.title('Forecasting 4G Payload using RNN')
st.write("View of the data series!")
check_data = st.checkbox("See the sample data")
if check_data:
    st.write(scaled_data_df.head(10))
st.write("Now let's find out the forecast of the next period of the week.")


#define n_input and n_feature
n_input = 24
n_input2 = 12
n_feature = 1

#define the last 12 point of data test as first_evaluation_batch
first_eval_batch = scaled_data[-n_input:]
first_eval_batch2 = scaled_data2[-n_input:]
first_eval_batch3 = scaled_data3[-n_input2:]
first_eval_batch4 = scaled_data4[-n_input2:]
first_eval_batch5 = scaled_data5[-n_input2:]

#Reshaping the evaluation batch to match put NN model, the RNN need 3D array
current_batch = first_eval_batch.reshape((1, n_input, n_feature))
current_batch2 = first_eval_batch2.reshape((1, n_input, n_feature))
current_batch3 = first_eval_batch3.reshape((1, n_input2, n_feature))
current_batch4 = first_eval_batch4.reshape((1, n_input2, n_feature))
current_batch5 = first_eval_batch5.reshape((1, n_input2, n_feature))

#Load Model Keras
loaded_model = load_model('RNN_4G_Payload_MalukuTenggara.h5')
loaded_model2 = load_model('RNN_4G_Payload_Bandung.h5')
loaded_model3 = load_model('RNN_4G_Payload_KOTABOGOR_model2.h5')
loaded_model4 = load_model('RNN_4G_Payload_KOTAMEDAN_model2.h5')
loaded_model5 = load_model('RNN_4G_Payload_KOTAPONTIANAK_model5.h5')



#make forecasting
forecast=[]
for i in range(week):
    #get prediction 1 time stamp ahead ([0] is for grabbing just number instead of array)
    current_pred = loaded_model.predict(current_batch)[0]
    #store the prediction
    forecast.append(current_pred)
    #update batch -- include the predictions and drop the first value
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis =1)


#make forecasting
forecast2=[]
for i in range(week):
    #get prediction 1 time stamp ahead ([0] is for grabbing just number instead of array)
    current_pred2 = loaded_model2.predict(current_batch2)[0]
    #store the prediction
    forecast2.append(current_pred2)
    #update batch -- include the predictions and drop the first value
    current_batch2 = np.append(current_batch2[:,1:,:], [[current_pred2]], axis =1)
	
#make forecasting
forecast3=[]
for i in range(week):
    #get prediction 1 time stamp ahead ([0] is for grabbing just number instead of array)
    current_pred3 = loaded_model3.predict(current_batch3)[0]
    #store the prediction
    forecast3.append(current_pred3)
    #update batch -- include the predictions and drop the first value
    current_batch3 = np.append(current_batch3[:,1:,:], [[current_pred3]], axis =1)
	
#make forecasting
forecast4=[]
for i in range(week):
    #get prediction 1 time stamp ahead ([0] is for grabbing just number instead of array)
    current_pred4 = loaded_model4.predict(current_batch4)[0]
    #store the prediction
    forecast4.append(current_pred4)
    #update batch -- include the predictions and drop the first value
    current_batch4 = np.append(current_batch4[:,1:,:], [[current_pred4]], axis =1)
	
#make forecasting	
forecast5=[]
for i in range(week):
    #get prediction 1 time stamp ahead ([0] is for grabbing just number instead of array)
    current_pred5 = loaded_model5.predict(current_batch5)[0]
    #store the prediction
    forecast5.append(current_pred5)
    #update batch -- include the predictions and drop the first value
    current_batch5 = np.append(current_batch5[:,1:,:], [[current_pred5]], axis =1)



#reinverse to the true value
#true_forecast = scaler.inverse_transform(forecast)
#true_forecast2 = scaler2.inverse_transform(forecast2)
#true_forecast3 = scaler3.inverse_transform(forecast3)
#true_forecast4 = scaler4.inverse_transform(forecast4)
#true_forecast5 = scaler5.inverse_transform(forecast5)

#create index timestap for forecast
index=pd.date_range(max(data.week_date), periods=week, freq='W')
index2=pd.date_range(max(data2.week_date), periods=week, freq='W')
index3=pd.date_range(max(data3.week_date), periods=week, freq='W')
index4=pd.date_range(max(data4.week_date), periods=week, freq='W')
index5=pd.date_range(max(data5.week_date), periods=week, freq='W')

#create data frame from forecast array
forecast=pd.DataFrame(data=forecast, index=index,columns=['Forecast'])
forecast['week_date']=forecast.index
forecast['week_date']=forecast['week_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
forecast=forecast.reset_index(drop=True)
forecast=forecast[['week_date', 'Forecast']]

#create data2 frame from forecast array
forecast2=pd.DataFrame(data=forecast2, index=index2,columns=['Forecast'])
forecast2['week_date']=forecast2.index
forecast2['week_date']=forecast2['week_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
forecast2=forecast2.reset_index(drop=True)
forecast2=forecast2[['week_date', 'Forecast']]

#create data frame from forecast array
forecast3=pd.DataFrame(data=forecast3, index=index3,columns=['Forecast'])
forecast3['week_date']=forecast3.index
forecast3['week_date']=forecast3['week_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
forecast3=forecast3.reset_index(drop=True)
forecast3=forecast3[['week_date', 'Forecast']]

#create data frame from forecast array
forecast4=pd.DataFrame(data=forecast4, index=index4,columns=['Forecast'])
forecast4['week_date']=forecast4.index
forecast4['week_date']=forecast4['week_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
forecast4=forecast4.reset_index(drop=True)
forecast4=forecast4[['week_date', 'Forecast']]


#create data frame from forecast array
forecast5=pd.DataFrame(data=forecast5, index=index5,columns=['Forecast'])
forecast5['week_date']=forecast5.index
forecast5['week_date']=forecast5['week_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
forecast5=forecast5.reset_index(drop=True)
forecast5=forecast5[['week_date', 'Forecast']]




#join dataframe data and forecast
df= scaled_data_df.append(forecast, ignore_index=True, sort=False)
df.set_index('week_date', inplace=True)
df.index=pd.to_datetime(df.index)

#join dataframe data and forecast
df2= scaled_data_df2.append(forecast2, ignore_index=True, sort=False)
df2.set_index('week_date', inplace=True)
df2.index=pd.to_datetime(df2.index)

#join dataframe data and forecast
df3= scaled_data_df3.append(forecast3, ignore_index=True, sort=False)
df3.set_index('week_date', inplace=True)
df3.index=pd.to_datetime(df3.index)

#join dataframe data and forecast
df4= scaled_data_df4.append(forecast4, ignore_index=True, sort=False)
df4.set_index('week_date', inplace=True)
df4.index=pd.to_datetime(df4.index)

#join dataframe data and forecast
df5= scaled_data_df5.append(forecast5, ignore_index=True, sort=False)
df5.set_index('week_date', inplace=True)
df5.index=pd.to_datetime(df5.index)


#plot the graphic of data
if st.sidebar.button("Forecast!"):
	st.markdown('**Forecasting of 4G Payload MALUKU TENGGARA**')
	st.line_chart(df)
	st.markdown('**Forecasting of 4G Payload BANDUNG**')
	st.line_chart(df2)
	st.markdown('**Forecasting of 4G Payload KOTA BOGOR**')
	st.line_chart(df3)
	st.markdown('**Forecasting of 4G Payload KOTA MEDAN**')
	st.line_chart(df4)
	st.markdown('**Forecasting of 4G Payload KOTA PONTIANAK**')
	st.line_chart(df5)
	
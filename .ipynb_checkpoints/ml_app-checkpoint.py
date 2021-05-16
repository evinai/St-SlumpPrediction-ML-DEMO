import streamlit as st
import os

# Load EDA Pkgs
import pandas as pd
import joblib
import pickle
from pickle import load

# Load EDA Pkgs
import numpy as np
from sklearn.svm import SVR,LinearSVR
from sklearn.preprocessing import StandardScaler
import regex

attrib_info = """

#### Attribute Information:
	- ACement
	- Slag
	- Fly ash
	- Water
	- SP
	- Coarse Aggr.
    - Fine Aggr.
 	- Output variables (3):
	- SLUMP (cm)
	- FLOW (cm)
 	- 28-day Compressive Strength (Mpa)


"""


# Load Models
def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model
  

@st.cache
def load_data():
	df = pd.read_csv('cement_slump.csv')
	df = df.astype('int64', copy=False)
	df.rename(columns={'Fly ash':'Fly_ash'}, inplace=True)
	df.rename(columns={'Coarse Aggr.':'Coarse_Aggr'}, inplace=True)
	df.rename(columns={'Fine Aggr.':'Fine_Aggr'}, inplace=True)
	return df
df= load_data()



def run_ml_app():
	
	st.subheader("From ML Section")

	st.sidebar.subheader('Data Input')
	Cement = st.sidebar.slider("Cement", 0, 500, step=1, key='C')
	Slag = st.sidebar.slider('Slag', 0,300 ,step=1, key='S')
	Fly_ash  = st.sidebar.slider('Fly_ash',100,300,step=1,key='F')
	Water = st.sidebar.slider('Water',100,300,step=1,key='W')
	SP = st.sidebar.slider('SP',0,30,step=1,key='P')
	Coarse_Aggr = st.sidebar.slider('Coarse_Aggr',700,1000,step=1,key='C')
	Fine_Aggr = st.sidebar.slider('Fine_Aggr',600,1000,step=1,key='F')
	Slump_cm = st.sidebar.slider('Slump_cm',0,40,step=1,key='L')
	Flow_cm = st.sidebar.slider('Flow_cm',0,40,step=1,key='M')

	selected_options = [Cement, Slag, Fly_ash, Water, SP, Coarse_Aggr, Fine_Aggr, Slump_cm, Flow_cm]
	vectorized_data = np.array(selected_options).reshape(1, -1)
	st.write(vectorized_data)

	with st.beta_expander("Attribute Info"):
		st.write(attrib_info)
		

	# Layout
	col1,col2 = st.beta_columns(2)

	with st.beta_expander("Your Selected Options"):
		result = {'Cement':Cement,
		'Slag':Slag,
		'Fly_ash':Fly_ash,
		'Water':Water,
		'SP':SP,
		'Coarse_Aggr':Coarse_Aggr,
		'Fine_Aggr':Fine_Aggr,
		'Slump_cm':Slump_cm,
		}
		st.info(selected_options)
	
		st.write(result)
	
	st.sidebar.subheader('Prediction')
	if st.sidebar.checkbox("Make Prediction"):
		all_ml_list = ['SVR']

		# Model Selection
		model_choice = st.selectbox("Model Choice", all_ml_list)
		if st.button("Predict"):
			if model_choice == 'SVR':
				model_predictor = load_prediction_models("svr_slump_model.joblib")
				pred_df = pd.DataFrame(vectorized_data,columns=['Cement', 'Slag', 'Fly_ash', 'Water', 'SP', 'Coarse_Aggr', 'Fine_Aggr',
       'Slump_cm', 'Flow_cm']).astype('float64')
				# pred_df
				st.table(pred_df)
				scaler = joblib.load('scaler_svr.gz')   
				scaled_data = scaler.transform(pred_df)
				# st.write(scaled_data)
				scaled_data = pd.DataFrame(scaled_data)
				prediction = model_predictor.predict(scaled_data)
				predicted = pd.DataFrame(prediction, index=None,columns=['Compressive Strength 28-day: Mpa'])
				# st.write(predicted)
				st.text(predicted.to_string(index=False))



	# 	# Load from file

		# Load ML Models





	





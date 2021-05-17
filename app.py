# Core pkgs
import streamlit as st
import streamlit.components.v1 as stc

# Import Our Mini Apps
from eda_app import run_eda_app
from ml_app import run_ml_app



# EDA
import pandas as pd
import numpy as np


# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Sci-kit Learn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR,LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error,mean_squared_error
import json

html_temp = """
		
		<h1 style="color:black;text-align:center;">Slump Test Predictor </h1>
		<h2 style="color:black;text-align:center;"> Beton Çökme Test Tahmin Uygulaması</h2>
			"""

		

attribute_list = """
		#### - Introduction
			Our data set consists of various cement properties and the resulting slump test metrics in cm. 
			Later on the set concrete is tested for its compressive strength 28 days later.

	
		#### - Attribute List
			- Cement
			- Slag
			- Fly ash
			- SP
			- Coarse Aggr
			- Fine Aggr
			- Cement
			- SLUMP (cm)
			- FLOW (cm)
			- 28-day Compressive Strength (Mpa)

			- Input variables (7)(component kg in one M^3 concrete):

			"""

			
data_source = """
			
			#### - Data Source

			https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test
			Credit: Yeh, I-Cheng, "Modeling slump flow of concrete using second-order regressions and artificial neural networks," Cement and Concrete Composites, Vol.29, No. 6, 474-480, 2007.

			#### - Method
			
			SVR - Support Vector Regression Model

			"""
	
 
desc_temp = """
			This dataset contains information to measure slump
			#### Content
				
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section: ML Predictor App

			"""
from PIL import Image 
img = Image.open("togaylogogri.png")
st.sidebar.image(img,width=270, caption='Demo-Lab')# 
# st.sidebar.image('st-sidebar.jpg', width=200, align=Left)

def main():
	# st.title("Main App")

	stc.html(html_temp)

	#st.title("Slump Test Predictor - Çökme Test Tahmin Uygulaması")
	st.write('Demo Uygulaması')

	menu = ["Ana","Exploratory Data Analysis","Makine Öğretisi","Hakkında"]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice == "Ana":
		st.subheader("Home")
		st.write(desc_temp)
		st.markdown(data_source)
		st.markdown(attribute_list)

	elif choice == "Exploratory Data Analysis":
		run_eda_app()

	elif choice == "Makine Öğretisi":
		run_ml_app()

	else:
		st.subheader("Hakkında")
		st.text('10-02-2021 - Togay Tunca')






if __name__ == '__main__':
	main()
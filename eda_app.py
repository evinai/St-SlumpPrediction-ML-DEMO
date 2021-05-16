import streamlit as st



# Load EDA Pkgs
import pandas as pd

# Load Data Viz pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px



desc_temp = """
			### This is the area where the insights could e written
			This dataset contains information to measure slump
			#### Insights
				
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section: ML Predictor App

			"""

# Load Data

@st.cache
def load_data():
	df = pd.read_csv('cement_slump.csv')
	df = df.astype('int64', copy=False)
	df.rename(columns={'Fly ash':'Fly_ash'}, inplace=True)
	df.rename(columns={'Coarse Aggr.':'Coarse_Aggr'}, inplace=True)
	df.rename(columns={'Fine Aggr.':'Fine_Aggr'}, inplace=True)
	return df


def run_eda_app():
	st.subheader("EDA - Exploratory Data Analysis")
	df = load_data()

	submenu = st.sidebar.selectbox("Submenu",["Descriptive","Plots"])
	if submenu == "Descriptive":
		with st.beta_expander("Data Frame"):
			st.dataframe(df)

		with st.beta_expander("Data Types"):
			st.dataframe(df.dtypes)

		with st.beta_expander("Descriptive Summary"):
			st.dataframe(df.describe().round(0))



	elif submenu == "Plots":
		st.subheader("Plots")

		# Correlation Plot
		with st.beta_expander("Correlation Plot"):
			corr_matrix = df.corr()
			fig = plt.figure(figsize=(20,15), dpi=130)
			sns.heatmap(corr_matrix,annot=True)
			st.pyplot(fig)

			p4 = px.imshow(corr_matrix)
			st.plotly_chart(p4)

		## add INSIGHTS html and correlation 
		## detailed info chart

























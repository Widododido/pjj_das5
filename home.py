import streamlit as st
import pandas as pd
import streamlit.components.v1 as stc 
import joblib

from eda import run_eda_app
from prediction import run_predict_app


st.set_page_config(page_title="Titanic ML",
		   page_icon="🚢",
		   layout="wide")

html_temp = """
		<div style="background-color:#3872fb;padding:5px;border-radius:10px">
		<h3 style="color:white;text-align:center;font-family:arial;">Titanic Machine Learning </h3>
		<h3 style="color:white;text-align:center;font-family:arial;">PJJ DAS V 2023</h3>
		</div>
		"""

def main():
    stc.html(html_temp)
    menu=["Home","EDA","Prediction"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Kelompok I")
        st.write('1. Ratu Najmil Huda')
        st.write('2. Ratih Nurjanah')
        st.write('3. Ayu Dharmasari')
        st.write('4. Nur Afni Kurniasari')
        st.write('5. Widodo')
    elif choice == "EDA":
        run_eda_app()
    
    elif choice =="Prediction":
        run_predict_app()


if __name__=='__main__':
    main()
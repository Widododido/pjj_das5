import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image


def load_model():
    model_gb=joblib.load('model.joblib')
    return model_gb

def run_predict_app():
    st.subheader('Prediction Section')
    model=joblib.load('model.joblib')
    with st.sidebar:
        st.title("Features")
        kelas=st.selectbox('Pclass :',('Excecutive','Bisnis','Ekonomi'))
        if kelas=='Excecutive':
            pclass=1
        elif kelas=='Bisnis':
            pclass=2
        else:
            pclass=3
        fare= st.number_input('Fare :', min_value=1.0, max_value=263.0)
        sex_label= st.selectbox('Sex :',('Male', 'Female'))
        if sex_label=='Male':
            sex=1
        else:
            sex=0

        Embarked = st.selectbox('Embarked :', ('Cherbourg','Queenstown','Southampton'))
        if Embarked == 'Cherbourg':
            embarked_enc=0
        elif Embarked == 'Queenstown':
            embarked_enc=1
        else:
            embarked_enc=2


        initial= st.selectbox('Initial :', ('Mr','Mrs','Miss','Master','Other'))
        if initial=='Mr':
            mr=1
            mrs=0
            miss=0
            master=0
            other=0
        elif initial=='Mrs':
            mr=0
            mrs=1
            miss=0
            master=0
            other=0
        elif initial=='Miss':
            mr=0
            mrs=0
            miss=1
            master=0
            other=0
        elif initial=='Master':
            mr=0
            mrs=0
            miss=0
            master=1
            other=0
        else:
            mr=0
            mrs=0
            miss=0
            master=0
            other=1

        
        age = st.slider('Age :',1,100)
        if age <=15:
            child=1
            young_adult=0
            adult=0
            elderly=0
        elif age<=30:
            child=0
            young_adult=1
            adult=0
            elderly=0
        elif age<=50:
            child=0
            young_adult=0
            adult=1
            elderly=0
        else:
            child=0
            young_adult=0
            adult=0
            elderly=1


    if st.button('Predict'):
        st.info('Input :')
        st.write('Class : {}'.format(pclass))
        st.write('Fare : ${} '.format(fare))
        st.write('Sex : ', sex_label)
        st.write('Embarked :', Embarked)
        st.write('Initial :', initial)
        st.write('Age : {}'.format(age))

        dfvalues = pd.DataFrame(list(zip([pclass],[fare],[sex],[embarked_enc],[mr],[mrs],[miss],[master],[other],
                                         [young_adult],[adult],[elderly],[child])),columns =['pclass', 'fare', 'sex', 'embarked','mr','mrs',
                                                           'miss','master','other','young_adult','adult',
                                                           'elderly','child'])
        input_variables=np.array(dfvalues)
        st.dataframe(dfvalues)
        st.info('Result :')
        col1, col2=st.columns([1,2])
        prediction=model.predict(input_variables)
        pred_prob=model.predict_proba(input_variables)
        with col1:
            st.write('Prediction :')
            hasil=prediction[0]
            if hasil==1:
                st.success('Survived')
            else:
                st.warning('Not Survived')
            

        with col2:
            if prediction == 1:
                pred_probability_score = pred_prob[0][1]*100
                st.write("Prediction Probability Score :")
                st.success("There is a : {:.2f} % you will survived like Rose".format(pred_probability_score))
                    # image_rose = Image.open('rose.jpeg')
                    # st.image(image_rose,width=460)
                img="https://www.thelist.com/img/gallery/things-only-adults-notice-in-titanic/why-would-rose-display-paintings-for-a-week-long-trip-on-the-titanic-1575316189.jpg"
                st.image(img,width=460)
            else:
                pred_probability_score = pred_prob[0][0]*100
                st.write("Prediction Probability Score")
                st.warning("There is a : {:.2f} % you will end up like Jack".format(pred_probability_score))
                    # image_jack = Image.open('jack.jpeg')
                image_jack="https://imgix.bustle.com/uploads/image/2017/7/4/e60d1805-b01f-4f02-a155-13f33814639a-jack-dawson-dirtbag.jpg?w=800&fit=crop&crop=faces&auto=format%2Ccompress&q=50&dpr=2"
                st.image(image_jack, width=460)
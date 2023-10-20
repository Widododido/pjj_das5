import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler




def load_data(data):
    return pd.read_csv(data)


def normalize(data):
	scaler = MinMaxScaler()
	ndf = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
	return ndf

def run_eda_app():
    st.write('Halaman EDA')
    df = load_data('titanic.csv')
    
    submenu = st.sidebar.selectbox("Submenu",["Descriptive","Visualization"])
    if submenu == 'Descriptive':
        with st.expander('Data Frame'):
            st.dataframe(df)

        with st.expander('Data Type'):
            st.write(df.dtypes)

        with st.expander('Data Shape'):
            st.write(df.shape)

        with st.expander('Descriptive Statistic'):
            st.write(df.describe().transpose())

        with st.expander('Null'):
            st.dataframe(df.isna().sum().transpose())

    elif submenu=='Visualization':
        st.write('Visualization')

        with st.expander('Pie chart passenger belonging to Emabarked'):
            embarked_counts = df['Embarked'].value_counts()
            fig, ax = plt.subplots()
            sns.set(style="whitegrid")
            ax.pie(embarked_counts, labels=embarked_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
            ax.axis('equal')
            ax.legend(title='Embarked', loc='upper center' ,bbox_to_anchor=(1, 0.5))
            st.pyplot(fig)

        with st.expander('Pclass Barchart'):
            fig, ax = plt.subplots()
            sns.histplot(data=df, x="Pclass", hue="Sex", multiple="dodge", shrink=.8)
            st.pyplot(fig)

        with st.expander('Fare Histogram'):
            fig, ax = plt.subplots()
            sns.histplot(data=df, x="Fare", kde=True, palette="deep")
            st.pyplot(fig)

        with st.expander('Violin Chart'):
            fig, ax = plt.subplots()
            sns.violinplot(data=df, x="Survived", y="Sex")
            plt.title('Violin Plot - Survived vs Sex')
            plt.xlabel('Survived')
            plt.ylabel('Sex')
            st.pyplot(fig)

        with st.expander('Matrix Correlation'):
            corr = df.select_dtypes(include=['number']).corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr,  annot=True, ax=ax, annot_kws={"size":5})
            ax.set_title('Matriks Correlation')
            plt.xticks(rotation=80)
            st.pyplot(fig)

        with st.expander('Outlier Detection'):
            data_kolom=df[['Age', 'Fare']]
            df_norm = normalize(data_kolom)
            fig, ax = plt.subplots()
            sns.boxplot(data=df_norm, orient='h', ax=ax)
            ax.set_title('Boxplot Normalisasi')
            st.pyplot(fig)

        

        


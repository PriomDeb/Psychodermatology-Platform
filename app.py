import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model # type: ignore

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu

from random_data import prediction_from_user_data
from priom_encryption import decrypt
# from password import PASSWORD



try:
    from password import PASSWORD
    try:
        password = PASSWORD
    except:
        pass
except:
    password = st.secrets["PASSWORD"]


file_path = "/Users/priom/Desktop/Psychodermatology"

def load_df():
    return pd.read_excel(f"{file_path}/PsyDerm_new_final.xlsx")

def initialize_model(model_path=f"{file_path}/model_01_30-09-2024_00-26-21.keras"):
    print(f"Attempting to load model from: {model_path}")
    try:
        loaded_model = load_model(model_path)
        print(f"Model Summary: \n{loaded_model.summary()}\n")
        return loaded_model
    except Exception as e:
        print(e)
        print("Error while loading the model.")


# initialize_model()

with st.sidebar:
    selected = option_menu(
    menu_title = "Main Menu",
    options = ["Home","Stats", "Contact Us"],
    icons = ["house","gear","envelope"],
    menu_icon = "cast",
    default_index = 0,
)
    
if selected == "Home":
    tab1, tab2 = st.tabs(["Custom Data", "Get Random Data from Dataset"])
    model = initialize_model()
    # df = load_df()
    df = decrypt(password=password, object="encrypted_df.joblib")
    
    with tab1:
     prediction_from_user_data(df=df, model=model, keys="tab1")
    
    with tab2:
     prediction_from_user_data(df=df, model=model, keys="tab2", random=True)

    

if selected == "Stats":
    st.header('Stats')
    df = decrypt(password=password, object="encrypted_df.joblib")
    # Create a row layout
    c1, c2= st.columns(2)
    c3, c4= st.columns(2)

    with st.container():
        c1.write("c1")
        c2.write("c2")

    with st.container():
        c3.write("c3")
        c4.write("c4")

    with c1:
        chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
        st.area_chart(chart_data)
           
    with c2:
        chart_data = pd.DataFrame(np.random.randn(20, 3),columns=["a", "b", "c"])
        st.bar_chart(chart_data)

    with c3:
        chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
        st.line_chart(chart_data)

    with c4:
        chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
        st.line_chart(chart_data)


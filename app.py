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


file_path = "/Users/priom/Desktop/Psychodermatology"
# df = pd.read_excel(f"{file_path}/PsyDerm_new_final.xlsx")

def initialize_model(model_path=f"{file_path}/model_01_30-09-2024_00-26-21.keras"):
    try:
        loaded_model = load_model(model_path)
        print(f"Model Summary: \n{loaded_model.summary()}\n")
        return loaded_model
    except:
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
    col1, col2, col3 = st.columns(3)
    
    # Input fields for each feature with corresponding labels
    with col1:
        sex = st.number_input('Sex (0 or 1)', min_value=0, max_value=1, value=1)
        imped_lat = st.number_input('Impedance Latency', min_value=0.0, value=1.5)
        bas_entertainment_sum = st.number_input('BAS Entertainment Sum', min_value=0, value=5)
        hq_fear_level_and_sensitivity_left = st.number_input('HQ Fear Level Left', min_value=0, value=3)
        hq_pair_bonding_and_spouse_dominance_style_right = st.number_input('HQ Pair Bonding Right', min_value=0, value=1)
    
    with col2:
        r_lfa = st.number_input('r_LFA', min_value=0.0, value=0.5)
        bacq_withdrawal_coping_sum = st.number_input('BACQ Withdrawal Coping Sum', min_value=0, value=4)
        hq_logical_orientation_right = st.number_input('HQ Logical Orientation Right', min_value=0, value=1)
        hq_fear_level_and_sensitivity_right = st.number_input('HQ Fear Level Right', min_value=0, value=2)
        wapi_verbal_left = st.number_input('WAPI Verbal Left', min_value=0, value=4)
        
    
    with col3:
        pns = st.number_input('PNS', min_value=0.0, value=3.2)
        bas_drive_sum = st.number_input('BAS Drive Sum', min_value=0, value=2)
        hq_type_of_consciousness_right = st.number_input('HQ Type of Consciousness Right', min_value=0, value=2)
        hq_pair_bonding_and_spouse_dominance_style_left = st.number_input('HQ Pair Bonding Left', min_value=0, value=3)
        rsq_anxious_attachment = st.number_input('RSQ Anxious Attachment', min_value=0, value=2)
        

    # Example of custom data based on selected features
    custom_data = np.array([
        sex, r_lfa, pns, imped_lat, bacq_withdrawal_coping_sum,
        bas_drive_sum, bas_entertainment_sum, hq_logical_orientation_right,
        hq_type_of_consciousness_right, hq_fear_level_and_sensitivity_left,
        hq_fear_level_and_sensitivity_right, hq_pair_bonding_and_spouse_dominance_style_left,
        hq_pair_bonding_and_spouse_dominance_style_right, wapi_verbal_left, rsq_anxious_attachment
    ]).reshape(1, -1)
    
    model = initialize_model()
    predictions = model.predict(custom_data)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    print(f"Predicted class: {predicted_class[0]}")
    st.write(f"Predicted Class: {predicted_class[0]}")
        

    
    
    pass

if selected == "Stats":
    st.header('Snowflake Healthcare App')
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


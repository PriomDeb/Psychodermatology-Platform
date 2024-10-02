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
import os

from random_data import prediction_from_user_data
from priom_encryption import decrypt
# from password import PASSWORD
from dataset_stats import basic_dataset_stats, training_visuals

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access environment variables



try:
    from password import PASSWORD
    try:
        password = PASSWORD
        file_path = "/Users/priom/Desktop/Psychodermatology"
        model_name = f"{file_path}/model_01_30-09-2024_00-26-21.keras"
    except:
        pass
except:
    try:
        model_name = "model.keras"
        password = st.secrets["PASSWORD"]
    except:
        model_name = "model.keras"
        password = os.getenv('PASSWORD')
        



def load_df():
    return pd.read_excel(f"{file_path}/PsyDerm_new_final.xlsx")

def initialize_model(model_path="model.h5"):
    absolute_model_path = os.path.abspath(model_path)
    print(f"Attempting to load model from: {absolute_model_path}")
    print(os.listdir(os.getcwd()))
    print(tf.__version__)
    try:
        loaded_model = load_model(absolute_model_path)
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
    model_1 = initialize_model()
    # df = load_df()
    df = decrypt(password=password, object="encrypted_df.joblib")
    
    with tab1:
     prediction_from_user_data(df=df, model=model_1, keys="tab1")
    
    with tab2:
     prediction_from_user_data(df=df, model=model_1, keys="tab2", random=True)

    

if selected == "Stats":
    c1, c2= st.columns(2)
    
    with c1:
        basic_dataset_stats()
    
    with c2:
        training_visuals()

    



def contact_page():
    st.header("Ticket System (Contact Us)")

    # Ticket Submission Form
    with st.form(key='ticket_form'):
        name = st.text_input("Name")
        email = st.text_input("Email")
        problem = st.text_area("Problem Description")

        # Create a submit button for the form
        submit_button = st.form_submit_button(label="Submit Ticket")

        # When the submit button is pressed
        if submit_button:
            if not name or not email or not problem:
                st.error("Please fill out all fields.")
            else:
                st.success(f"Thank you {name}! Your ticket has been submitted. Ticket ID:")

    
    
    




if selected == "Contact Us":
    contact_page()




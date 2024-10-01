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
from dataset_stats import basic_dataset_stats

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
        basic_dataset_stats()
    
    # st.header('Stats')
    # features = ['sex', 'r_lfa', 'pns', 'imped_lat', 
    #             'bacq_withdrawal_coping_sum', 'bas_drive_sum', 'bas_entertainment_sum', 
    #             'hq_logical_orientation_right', 'hq_type_of_consciousness_right', 
    #             'hq_fear_level_and_sensitivity_left', 'hq_fear_level_and_sensitivity_right', 
    #             'hq_pair_bonding_and_spouse_dominance_style_left', 
    #             'hq_pair_bonding_and_spouse_dominance_style_right', 'wapi_verbal_left', 
    #             'rsq_anxious_attachment']
    # df = decrypt(password=password, object="encrypted_df.joblib")
    # df.columns = df.columns.str.lower()
    # # Create a row layout
    # c1, c2= st.columns(2)
    # c3, c4= st.columns(2)
    
    # with c1:
    #     st.subheader('Class Distribution')
    #     fig, ax = plt.subplots()
    #     sns.countplot(x='group', data=df, palette='Set2', ax=ax)
    #     ax.set_title('Class Distribution')
    #     ax.set_xlabel('Group (0: Healthy, 1: AD, 2: Psoriasis)')
    #     ax.set_ylabel('Count')
    #     st.pyplot(fig)
    
    # # with c2:
    # #     st.subheader('Correlation Heatmap')
    # #     corr = df.corr()
    # #     fig, ax = plt.subplots(figsize=(10, 8))
    # #     sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    # #     ax.set_title('Feature Correlation Matrix')
    # #     st.pyplot(fig)
    
    
    # # with c3:
    # #     model_1 = initialize_model()
    # #     importances = model_1.feature_importances_ 
        
    # #     st.subheader('Feature Importances')
    # #     fig, ax = plt.subplots()
    # #     sns.barplot(x=importances, y=features, palette='Blues_d', ax=ax)
    # #     ax.set_title('Top Features by Importance')
    # #     ax.set_xlabel('Importance Score')
    # #     st.pyplot(fig)
    
    
    # with c4:
    #     st.subheader('Boxplot of Key Features by Group')
    #     feature_to_plot = 'r_lfa'  # Choose a feature dynamically if required
    #     fig, ax = plt.subplots()
    #     sns.boxplot(x='group', y=feature_to_plot, data=df, palette='Set3', ax=ax)
    #     ax.set_title(f'Boxplot of {feature_to_plot} by Group')
    #     ax.set_xlabel('Group (0: Healthy, 1: AD, 2: Psoriasis)')
    #     ax.set_ylabel(feature_to_plot)
    #     st.pyplot(fig)

    



def load_tickets():
    try:
        tickets_df = pd.read_csv('tickets.csv')
    except FileNotFoundError:
        # Create a new DataFrame with necessary columns if the file does not exist
        tickets_df = pd.DataFrame(columns=['Ticket ID', 'Name', 'Email', 'Problem', 'Status', 'Response'])
    # Ensure 'Response' column exists
    if 'Response' not in tickets_df.columns:
        tickets_df['Response'] = ''
    return tickets_df

# Helper function to save tickets to a CSV file
def save_tickets(tickets_df):
    tickets_df.to_csv('tickets.csv', index=False)

def contact_page():
    st.header("Ticket System (Contact Us)")
    
    # Load existing tickets
    tickets_df = load_tickets()

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

    # Display the tickets in a table
    st.subheader("All Tickets")
    
    




if selected == "Contact Us":
    contact_page()




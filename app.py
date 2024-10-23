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

st.set_page_config(page_title="Predict Psychodermatological Risk", layout="wide")

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
    menu_title = "Psychodermatology AI",
    options = ["Home","Visual Analysis", "Inspired From", "Contact"],
    icons = ["house","gear", "star", "envelope"],
    menu_icon = "robot",
    default_index = 0,
)
    
if selected == "Home":
    st.header("""🧠 Try this AI Model to Predict Psychodermatological Risk of Psoriasis or Atopic Dermatitis""")
    st.markdown("""
                - 0: Healthy Control 
                - 1: Atopic Dermatitis
                - 2: Psoriasis
                """)
    tab1, tab2 = st.tabs(["Custom Data", "Get Random Data from Dataset"])
    model_1 = initialize_model()
    # df = load_df()
    df = decrypt(password=password, object="encrypted_df.joblib")
    
    with tab1:
     prediction_from_user_data(df=df, model=model_1, keys="tab1")
    
    with tab2:
     prediction_from_user_data(df=df, model=model_1, keys="tab2", random=True)

    

if selected == "Visual Analysis":
    c1, c2= st.columns(2)
    
    with c1:
        basic_dataset_stats()
    
    with c2:
        training_visuals()

if selected == "Inspired From":
    st.header("Assessment of Frontal Hemispherical Lateralization in Plaque Psoriasis and Atopic Dermatitis", help="Read the original paper from where I got the motivation to build this Machine Learning application. It is a great work!")
    
    st.markdown("""
    **Citation:** Bozsányi, S.; Czurkó, N.; Becske, M.; Kasek, R.; Lázár, B.K.; Boostani, M.; Meznerics, F.A.; Farkas, K.; Varga, N.N.; Gulyás, L.; et al. 
    Assessment of Frontal Hemispherical Lateralization in Plaque Psoriasis and Atopic Dermatitis. *J. Clin. Med.* 2023, 12, 4194. 
    
    [Read the full paper](https://doi.org/10.3390/jcm12134194)
    
    Published: June 21, 2023.
    """)

    st.subheader("Abstract", help=f"Abstract of original [paper](https://doi.org/10.3390/jcm12134194)")
    st.markdown("""
    **Background:** Each brain hemisphere plays a specialized role in cognitive and behavioral processes, known as hemispheric lateralization. 
    In chronic skin diseases, such as plaque psoriasis (Pso) and atopic dermatitis (AD), the degree of lateralization between the frontal hemispheres 
    may provide insight into specific connections between skin diseases and the psyche.

    **Methods:** The study included 46 patients with Pso, 56 patients with AD, and 29 healthy control (Ctrl) subjects. The participants underwent frontal electroencephalogram (EEG) measurement, heart rate variability (HRV) assessment, and psychological tests. Statistical analyses were performed using ANOVA, with Bonferroni correction applied for multiple comparisons.

    **Results:** This study shows significant right-lateralized prefrontal activity in both AD patients (p < 0.001) and Pso patients (p = 0.045) compared with Ctrl, with no significant difference between the AD and Pso groups (p = 0.633). AD patients with right-hemispheric dominant prefrontal activation exhibited increased inhibition and avoidance markers, while Pso patients showed elevated sympathetic nervous system activity.

    **Conclusion:** Psychophysiological and psychometric data suggest a shared prevalence of right-hemispheric dominance in both AD and Pso patient groups. However, the findings indicate distinct psychodermatological mechanisms in AD and Pso.
    """)

    st.subheader("Acknowledgment and Note")
    st.warning("""
    I am not the author of the paper. However, with the help of the authors, I took the data from this **[original](https://doi.org/10.3390/jcm12134194)** work and trained a machine learning model to classify different groups based on the provided psychophysiological and psychometric data. This platform showcases the potential machine learning application and analysis based on the data from the authors' work.

    I would like to especially thank the first author, ***Szabolcs Bozsányi***. Also, ***Prof. Imre Lázár***, he was the author of the previous [article](https://doi.org/10.3390/jcm12134194) and it was actually his courtesy letting us using the data. I would also like to thank him espicially for the dataset. Lastly, thank you the rest of the research team for sharing and explaining the data, which made this platform possible.
    
    Thank you all once again!
    """)


    



def contact_page():
    st.header("Contact")
    
    st.subheader("Contact Details", help="Contact developer of this platform.")
    st.write("**Priom Deb**")
    st.write("Computer Science Graduate, BRAC University")
    st.write("**Email:** priom@priomdeb.com | priom.deb@g.bracu.ac.bd")
    st.write("**Web:** https://priomdeb.com")
    
    st.markdown("""
                #### Contact Authors of the Paper 
                [Original Paper](https://doi.org/10.3390/jcm12134194)
                """)

    # Ticket Submission Form
    # with st.form(key='ticket_form'):
    #     name = st.text_input("Name")
    #     email = st.text_input("Email")
    #     problem = st.text_area("Problem Description")

    #     # Create a submit button for the form
    #     submit_button = st.form_submit_button(label="Submit Ticket", disabled=True)
    #     st.warning("Ticket system for support not available right now. Use contact details to report anything or share feedback.\nThank you for trying this platform.")

    #     # When the submit button is pressed
    #     if submit_button:
    #         if not name or not email or not problem:
    #             st.error("Please fill out all fields.")
    #         else:
    #             st.success(f"Thank you {name}! Your ticket has been submitted. Ticket ID:")
        

    
    
    




if selected == "Contact":
    contact_page()




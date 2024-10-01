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



try:
    from password import PASSWORD
    try:
        password = PASSWORD
        file_path = "/Users/priom/Desktop/Psychodermatology"
        model = f"{file_path}/model_01_30-09-2024_00-26-21.keras"
    except:
        pass
except:
    model = "model.keras"
    password = st.secrets["PASSWORD"]



def load_df():
    return pd.read_excel(f"{file_path}/PsyDerm_new_final.xlsx")

def initialize_model(model_path=model):
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
                # Generate a new Ticket ID
                ticket_id = len(tickets_df) + 1
                new_ticket = {
                    'Ticket ID': ticket_id, 
                    'Name': name, 
                    'Email': email, 
                    'Problem': problem, 
                    'Status': 'Open',
                    'Response': ''
                }
                
                # Append the new ticket to the DataFrame
                tickets_df = pd.concat([tickets_df, pd.DataFrame([new_ticket])], ignore_index=True)
                save_tickets(tickets_df)
                
                st.success(f"Thank you {name}! Your ticket has been submitted. Ticket ID: {ticket_id}")

    # Display the tickets in a table
    st.subheader("All Tickets")
    
    # Define table columns
    columns = ["Ticket ID", "Name", "Email", "Problem", "Status", "Response", "Action"]
    ticket_rows = []

    # Allow the user to change the status of a ticket and enter a response
    for index, row in tickets_df.iterrows():
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 2, 2, 5, 2, 3, 1])
        
        col1.write(row['Ticket ID'])
        col2.write(row['Name'])
        col3.write(row['Email'])
        col4.write(row['Problem'])
        
        # Dropdown to change the status
        new_status = col5.selectbox("Status", options=['Open', 'In Progress', 'Resolved'], 
                                      index=['Open', 'In Progress', 'Resolved'].index(row['Status']), 
                                      key=f"status_{row['Ticket ID']}")
        
        # Input for response
        response = col6.text_area(f"Response for Ticket ID {row['Ticket ID']}", 
                                   value=row.get('Response', ''),  # Use .get to avoid KeyError
                                   key=f"response_{row['Ticket ID']}")
        
        # Input for PIN
        pin = col7.text_input("PIN", type="password", key=f"pin_{row['Ticket ID']}")

        # Create action button
        if col7.button("Update", key=f"update_{row['Ticket ID']}"):
            if pin == '1234':
                # Update status if changed
                if new_status != row['Status']:
                    tickets_df.at[index, 'Status'] = new_status
                
                # Update response if changed
                if response != row.get('Response', ''):  # Use .get to avoid KeyError
                    tickets_df.at[index, 'Response'] = response
                
                save_tickets(tickets_df)
                st.success(f"Ticket ID {row['Ticket ID']} updated successfully.")
            else:
                st.error("Incorrect PIN. Please try again.")

        # Append row data for display
        ticket_rows.append(row)

    # Convert updated DataFrame back to DataFrame and display
    updated_tickets_df = pd.DataFrame(ticket_rows)
    st.dataframe(updated_tickets_df[['Ticket ID', 'Name', 'Email', 'Problem', 'Status', 'Response']], use_container_width=True)




if selected == "Contact Us":
    contact_page()




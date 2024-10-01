import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from priom_encryption import decrypt

from dotenv import load_dotenv
import os


load_dotenv()

try:
    from password import PASSWORD
    try:
        password = PASSWORD
    except:
        pass
except:
    try:
        password = st.secrets["PASSWORD"]
    except:
        password = os.getenv('PASSWORD')

df = decrypt(password=password, object="encrypted_df.joblib")
df.columns = df.columns.str.lower()



def basic_dataset_stats():
    st.header("📊 Dataset", help="This section provides an overview of your dataset with various visualizations and statistics.")
    
    
    count_tab, bar_chart_tab, pie_chart_tab, info_tab = st.tabs(["🔢 Counts", "📊 Bar Charts", "🥧 Pie Charts", "📝 Info"])
    
    group_counts = df['group'].value_counts()

    # Counts Tab
    with count_tab:
        st.subheader("Counts")
        
        # Show basic row and column count
        st.write(f"""Total Rows: **`{df.shape[0]}`**""")
        st.write(f"""Total Columns: **`{df.shape[1]}`**""")
        
        st.write("""  """)
        
        # # Show group counts (assuming 'group' column has AD, Psoriasis, and Healthy control data)
        # st.write("""Group Distribution:""")
        # group_counts_df = pd.DataFrame({
        #     'Group': group_counts.index,
        #     'Count': group_counts.values
        #     }).reset_index(drop=True)
        # st.table(group_counts_df.style)
        
        
        # Plot class distribution
        fig, ax = plt.subplots()
        sns.countplot(x='group', data=df, palette='Set2', ax=ax)
        ax.set_title('Class Distribution \n(Healthy Control-0, Atopic Dermatitis-1, Psoriasis-2)')
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])
        ax.bar_label(ax.containers[2])
        st.pyplot(fig)
        
        
        
        # Plot class distribution
        fig, ax = plt.subplots()
        sns.countplot(x='sex', data=df, palette='Set2', ax=ax)
        ax.set_title('Sex Distribution \n(Male-1, Female-2)')
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])
        st.pyplot(fig)
        
        
        
        # Plot class distribution
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.countplot(x='age', data=df, palette='Set2', ax=ax)
        ax.set_title('Age Distribution')
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        
        for i in range(len(ax.containers)):
            ax.bar_label(ax.containers[i])
            
        
        st.pyplot(fig)

    # Bar Charts Tab
    with bar_chart_tab:
        st.subheader("Bar Charts")
        
        # Display bar charts for categorical variables (e.g., Sex, group, etc.)
        st.write("Bar chart for gender distribution:")
        fig, ax = plt.subplots()
        sns.countplot(x='sex', data=df, palette='coolwarm', ax=ax)
        ax.set_title('Gender Distribution')
        ax.set_xlabel('Sex (1: Male, 2: Female)')
        st.pyplot(fig)

        st.write("Bar chart for age distribution:")
        fig, ax = plt.subplots()
        df['age'].value_counts().sort_index().plot(kind='bar', ax=ax, color='orange')
        ax.set_title('Age Distribution')
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    # Pie Charts Tab
    with pie_chart_tab:
        st.subheader("Pie Charts")
        
        # Pie chart for group distribution
        st.write("Group Distribution:")
        fig, ax = plt.subplots()
        group_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ff9999'], ax=ax)
        ax.set_ylabel('')
        ax.set_title('Group Distribution (AD, Psoriasis, Healthy Control)')
        st.pyplot(fig)

        # Pie chart for gender distribution
        st.write("Gender Distribution:")
        fig, ax = plt.subplots()
        df['sex'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ffcc99','#ff9999'], ax=ax)
        ax.set_ylabel('')
        ax.set_title('Gender Distribution')
        st.pyplot(fig)
    
    # Info Tab
    with info_tab:
        st.subheader("Dataset Information")
        
        # Display dataset overview
        st.write("Dataset Overview:")
        st.write(df.describe())

        # Show first few rows of the dataset
        st.write("Sample Data:")
        st.write(df.head())

        # Check for missing values
        st.write("Missing Values:")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

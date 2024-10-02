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
    st.header("📊 Dataset", help=f"""This section provides an overview of the dataset with various visualizations and statistics.\nTotal Rows: **`{df.shape[0]}`** Total Columns: **`{df.shape[1]}`**.""")
    
    
    count_tab, bar_chart_tab, pie_chart_tab, info_tab = st.tabs(["📊 Bar Charts", "📈 Violin Charts", "🥧 Pie Charts", "📝 Info"])
    
    group_counts = df['group'].value_counts()

    # Counts Tab
    with count_tab:
        st.write("""  """)
        
        fig, ax = plt.subplots()
        sns.countplot(x='group', data=df, palette='Set2', ax=ax)
        ax.set_title('Class Distribution \n(0: Healthy Control, 1: Atopic Dermatitis, 2: Psoriasis)')
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])
        ax.bar_label(ax.containers[2])
        ax.set_ylabel("Count")
        ax.set_xlabel("Group")
        st.pyplot(fig)
        
        
        
        fig, ax = plt.subplots()
        sns.countplot(x='sex', data=df, palette='Set2', ax=ax)
        ax.set_title('Gender Distribution \n(Male-1, Female-2)')
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])
        ax.set_ylabel("Count")
        ax.set_xlabel("Gender")
        st.pyplot(fig)
        
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.countplot(x='age', data=df, palette='Set2', ax=ax)
        ax.set_title('Age Distribution')
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        
        for i in range(len(ax.containers)):
            ax.bar_label(ax.containers[i])
        
        ax.set_ylabel("Count")
        ax.set_xlabel("Age")
        
        st.pyplot(fig)

        
        fig, ax = plt.subplots()
        pd.crosstab(df['group'], df['sex']).plot(kind='bar', stacked=True, ax=ax, color=['#66b3ff', '#ffcc99'])
        ax.set_title("Gender Distribution within Class")
        ax.set_xlabel("Class (0: Healthy Control, 1: Atopic Dermatitis, 2: Psoriasis)")
        ax.set_ylabel("Count")
        ax.legend(labels=["Male", "Female"])
        st.pyplot(fig)
        
        
        
    with bar_chart_tab:
        st.write("""  """)

        
        fig, ax = plt.subplots()
        sns.violinplot(x='group', y='age', data=df, palette='Set3', ax=ax)
        ax.set_title("Age Distribution by Class (0: Healthy Control, 1: Atopic Dermatitis, 2: Psoriasis)")
        ax.set_xlabel("Group/Class")
        ax.set_ylabel("Age")
        st.pyplot(fig)
        
        st.write("""  """)
        
        
        
        fig, ax = plt.subplots()
        sns.violinplot(x='group', y='sex', data=df, palette='Set3', ax=ax)
        ax.set_title("Gender Distribution by Class (Male-1, Female-2)")
        ax.set_xlabel("Group/Class")
        ax.set_ylabel("Sex")
        st.pyplot(fig)

        
    
    # Pie Charts Tab
    with pie_chart_tab:
        st.subheader("Pie Charts")
        
        # Pie chart for group distribution
        fig, ax = plt.subplots()
        group_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ff9999'], ax=ax)
        ax.set_ylabel('')
        ax.set_title('Group Distribution \n(Healthy Control-0, Atopic Dermatitis-1, Psoriasis-2)')
        ax.legend(labels=['Atopic Dermatitis', 'Psoriasis', 'Healthy Control'])
        st.pyplot(fig)

        # Pie chart for gender distribution
        fig, ax = plt.subplots()
        df['sex'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ffcc99','#ff9999'], ax=ax)
        ax.set_ylabel('')
        ax.set_title('Gender Distribution')
        ax.legend(labels=["Female", "Male"])
        st.pyplot(fig)
    
    # Info Tab
    with info_tab:
        st.subheader("Dataset Information")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"""Total Rows: **`{df.shape[0]}`**""")
        with c2:
            st.write(f"""Total Columns: **`{df.shape[1]}`**""")
            
        # Display dataset overview
        st.write("Dataset Overview:")
        st.write(df.describe())
        

        # Show first few rows of the dataset
        st.write("Sample Data:")
        n = st.selectbox("Select the number of rows to display:", [i for i in range(1, 11)], index=4)
        st.info("Dataset is encrypted using PBKDF2 Method. Users can only see first 10 data.")
        st.write(df.head(n))

        st.write("Missing Values:")
        missing_values = df.isnull().sum()
        missing_values_df = missing_values[missing_values > 0].reset_index()
        missing_values_df.columns = ['Columns', 'Missing Values']
        st.dataframe(missing_values_df, use_container_width=True)


def training_visuals():
    st.header("🧠 Model Training Visuals", help=f"""This section provides an overview of the model's training process and performance metrics.\nHere, you can visualize accuracy, loss, and classification results for your trained model.""")
    
    
    loss_acc_tab, confusion_matrix_tab, report_tab = st.tabs(["📉 Loss/Accuracy Curves", "🔢 Confusion Matrix", "📋 Classification Report"])
    
    
    with loss_acc_tab:
        st.subheader("Training & Validation Curves")
        # Call your function to plot training and validation curves
        # plot_training_curves(history)
        st.write("Placeholder for training and validation accuracy/loss curves.")
    
    
    with confusion_matrix_tab:
        st.subheader("Confusion Matrix")
        # Call your function to plot the confusion matrix
        # plot_confusion_matrix(y_true, y_pred, labels=["Healthy", "AD", "Psoriasis"])
        st.write("Placeholder for confusion matrix.")
    
    
    with report_tab:
        st.subheader("Classification Report")
        # Here you can display precision, recall, and F1-score
        st.write("Placeholder for classification report (precision, recall, F1).")



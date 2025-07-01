import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib

# from PIL import Image
# Image.MAX_IMAGE_PIXELS = None

# Load the trained ML model
def load_model():
    model = joblib.load('rf_model.pkl')
    return model


def load_scaler():
    scaler = joblib.load('scaler.pkl')
    return scaler

# Preprocessing function
def preprocess_data(data, scaler):
    # Compute new features
    data['Total_Income']=data['ApplicantIncome']+data['CoapplicantIncome']
    data['EMI']=data['LoanAmount']/data['Loan_Amount_Term'] 
    data['Balance_Income']=data['Total_Income']-(data['EMI']*1000) 
    
    # Log transformations
    data['Total_Income_log'] = np.log(data['Total_Income'])
    data['EMI_log'] = np.log(data['EMI'])
    c = 1769
    data['Balance_Income_log'] = np.log(data['Balance_Income'] + c)
    
    # Drop uncessary columns
    data = data.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                      'Loan_Amount_Term', 'Total_Income', 'EMI', 'Balance_Income'], 
                     axis=1)
    
    # Reorder columns
    col = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Credit_History', 'Property_Area', 'Total_Income_log',
       'EMI_log', 'Balance_Income_log']
    data = data[col]
    
    # Apply the scaler only to the numerical columns in X
    numerical_columns_log = ['Total_Income_log', 'EMI_log', 'Balance_Income_log']
    data[numerical_columns_log] = scaler.transform(data[numerical_columns_log])
    
    # Keep only required features
    feature_column = ['Credit_History', 'Total_Income_log', 'EMI_log', 'Balance_Income_log']
    data = data[feature_column]
    
    return data


# App Layout
def main():
   # Set page layout to wide
   st.set_page_config(layout="wide")
   
   # Create a centered layout
   col1, col2, col3 = st.columns([1,3,1])
   with col2:
       st.title('Loan Status Prediction App Built by Subhiksha') 
       st.image("Loan_Prediction_Image.jpg", 
                caption="Make smarter loan decisions with AI",
                use_container_width=True)  # Adjusts image width automatically to the column size
       st.write("""
        Welcome to the Loan Status Prediction App! This tool allows you to predict whether your loan will be **Approved** or **Rejected** based on your provided details.
        
        👉 **How It Works:**
        - Fill in the form on the left.
        - Get a prediction on your loan status.
        - Understand why the prediction was made with an interactive **SHAP Explanation**.
        """)
        
   # Sidebar setup
   st.sidebar.header("User Input Features")    
   
   # Useer input in the sidebar
   def user_input_features():
       st.sidebar.subheader('Applicant Details')
       Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
       Married = st.sidebar.selectbox("Marital Status", ["Yes", "No"])
       Dependents = st.sidebar.text_input("Number of Dependents", "0")
       Education = st.sidebar.selectbox("Education Level", ["Graduate", "Not Graduate"])
       Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
       
       st.sidebar.subheader("Income Details")
       ApplicantIncome = st.sidebar.text_input(" Monthly Applicant Income (in USD)", "1000")
       CoapplicantIncome = st.sidebar.text_input("Monthly Coapplicant Income (in USD)", "0")
       
       st.sidebar.subheader("Loan Details")
       LoanAmount = st.sidebar.text_input("Loan Amount (in USD thousands)","100")
       Loan_Amount_Term = st.sidebar.slider("Loan Term (in months)", 12, 420, 120)
       Credit_History = st.sidebar.selectbox("Credit History (Good: 1, Bad: 0)", [1, 0])
       Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])   
      
       data = {
            "Gender": 1 if Gender == "Male" else 0,
            "Married": 1 if Married == "Yes" else 0,
            "Dependents": int(Dependents),
            "Education": 0 if Education == "Graduate" else 1,
            "Self_Employed": 1 if Self_Employed == "Yes" else 0,
            "ApplicantIncome": float(ApplicantIncome),
            "CoapplicantIncome": float(CoapplicantIncome),
            "LoanAmount": float(LoanAmount),
            "Loan_Amount_Term": float(Loan_Amount_Term),
            "Credit_History": Credit_History,
            "Property_Area": ["Urban", "Semiurban", "Rural"].index(Property_Area),
        }
       
       return pd.DataFrame(data, index = [0]) # create a pandas DataFrame with one row of data and a specified index
           

   with col2: 
       user_input =  user_input_features()
       
       st.subheader("User Input Features")
       st.write(user_input)
   
       
   # Submit button to trigger the prediction
   if st.sidebar.button("Submit"): 
   
       # Load model and scaler
       model = load_model()
       scaler = load_scaler()
       
       # Preprocess data
       processed_data  =  preprocess_data(user_input, scaler)
       
       # Make prediction
       prediction = model.predict(processed_data)
       # prediction_proba = model.predict_proba(processed_data)
    
       with col2:    
           st.subheader("Prediction Result")
           loan_status = "Approved" if prediction[0] == 1 else "Rejected"
           st.write(f"Your loan is **{loan_status}**.")
           # st.write("Prediction Probability:")
           # st.write(prediction_proba)
           
           # Shap interpretation
           st.subheader("Model Explanation using SHAP")
           st.write("""
        SHAP (SHapley Additive exPlanations) values explain the impact of each feature on the model's prediction:
        
        - The Waterfall Plot provides a detailed breakdown of how each feature influenced the prediction.
        - The base value is the average model output over the training data.
        - Positive and negative contributions from individual features are combined to adjust the base value, culminating in the model’s final prediction.
        """)
        
       # Fit the explainer
       explainer = shap.Explainer(model)
        
       # Calculates the SHAP values
       shap_values = explainer(processed_data)
    
     
       # Choose the first sample and the first output
       shap_values_sample_output = shap_values.values[0, :, 0]  # 0 for the first output
    
       # Get the corresponding base value for the first output
       base_value = shap_values.base_values[0, 0]
    
       # Get the data for the first sample
       data = shap_values.data[0]
       
       with col2:
           # Use waterfall plot for detailed explanation
           st.write("The waterfall Plot shows the contribution of each feature to the final prediction for the loan application.")
       
           # Plot the waterfall chart
           fig_inc = plt.figure()
           shap.waterfall_plot(shap.Explanation(values=shap_values_sample_output,
                                         base_values=base_value,
                                         data=data,
                                         feature_names=["Credit History", "Total Income", "EMI", "Balance Income"]))
           st.pyplot(fig_inc)
           
           st.subheader("Feature Insights")
           st.markdown("""
            - **Total Income**: Sum of applicant income and co-applicant income. Higher total income generally increases the chances of loan approval.
            - **EMI**: Fixed monthly payment calculated as loan amount divided by loan term. Lower EMI reduces financial burden.
            - **Balance Income**: Income remaining after paying EMI. Higher balance income indicates better repayment capability.
            """)


if __name__ == "__main__":
    main()
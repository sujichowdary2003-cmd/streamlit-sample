
import pickle
import numpy as np

# Load the saved model and scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))
lr = pickle.load(open('lr.pkl', 'rb'))

# Streamlit UI
st.title("🩺 Diabetes Prediction using Logistic Regression")

st.write("Enter the following details to predict diabetes status:")

# Input fields
Age = st.number_input("Age", min_value=1, max_value=120, value=30)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)

# Predict button
if st.button("Predict"):
    try:
        # Prepare input data
        temp_arr = [Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        data = np.array([temp_arr])

        # Scale and predict
        temp_sc = scaler.transform(data)
        pred = int(lr.predict(temp_sc)[0])

        # Display result
        if pred == 0:
            st.success("✅ The result **does not indicate diabetes**.")
        else:
            st.error("⚠️ The result **indicates diabetes**.")

    except Exception as e:
        st.error(f"Error: {str(e)}")


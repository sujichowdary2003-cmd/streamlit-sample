import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# Load Model and Scaler
# -----------------------------
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('logistic_model.pkl', 'rb'))

# -----------------------------
# File to store user credentials
# -----------------------------
USER_FILE = "users.pkl"

# Load existing users or create empty
if os.path.exists(USER_FILE):
    with open(USER_FILE, "rb") as f:
        users = pickle.load(f)
else:
    users = {"admin": "1234"}  # default admin account
    with open(USER_FILE, "wb") as f:
        pickle.dump(users, f)

# -----------------------------
# Save updated users
# -----------------------------
def save_users():
    with open(USER_FILE, "wb") as f:
        pickle.dump(users, f)

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Diabetes Detection", page_icon="🩺", layout="centered")

# -----------------------------
# Initialize login session
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# -----------------------------
# LOGIN PAGE
# -----------------------------
def login_page():
    st.title("🔐 Login to Diabetes Detection App")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}! ✅")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password ❌")

    st.info("Don't have an account?")
    if st.button("Sign Up"):
        st.session_state.page = "signup"
        st.experimental_rerun()

# -----------------------------
# SIGN UP PAGE
# -----------------------------
def signup_page():
    st.title("📝 Create a New Account")

    new_user = st.text_input("Choose a Username")
    new_pass = st.text_input("Choose a Password", type="password")
    confirm_pass = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if new_user in users:
            st.warning("Username already exists. Please choose another.")
        elif new_pass != confirm_pass:
            st.error("Passwords do not match ❌")
        elif new_user == "" or new_pass == "":
            st.error("Please fill all fields.")
        else:
            users[new_user] = new_pass
            save_users()
            st.success("Account created successfully! 🎉 You can now log in.")
            st.session_state.page = "login"
            st.experimental_rerun()

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.experimental_rerun()

# -----------------------------
# MAIN PREDICTION PAGE
# -----------------------------
def prediction_page():
    st.title("🩺 Diabetes Detection App")
    st.write(f"Welcome, **{st.session_state.username}** 👋")
    st.write("Enter the following details to predict diabetes status:")

    Age = st.number_input("Age", min_value=1, max_value=120, value=30)
    Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)

    if st.button("Predict"):
        try:
            # Prepare input
            input_data = np.array([[Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            scaled_data = scaler.transform(input_data)
            pred = int(model.predict(scaled_data)[0])

            if pred == 0:
                st.success("✅ The result indicates the person is **not diabetic**.")
            else:
                st.error("⚠️ The result indicates the person is **diabetic**.")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("You have been logged out.")
        st.experimental_rerun()

# -----------------------------
# PAGE NAVIGATION
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"

if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
else:
    prediction_page()



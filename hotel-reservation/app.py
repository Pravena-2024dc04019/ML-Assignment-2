import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hotel Predictor Pro", page_icon="🏨")

# --- ASSET LOADING ---
@st.cache_resource
def load_selected_model(model_name):
    # Name of the models avaialable
    model_mapping = {
        "Random Forest": "Random_Forest.pkl",
        "XGBoost": "XGBoost.pkl",
        "Logistic Regression": "Logistic_Regression.pkl",
        "Decision Tree": "Decision_Tree.pkl",
        "K-Nearest Neighbors": "KNN.pkl",
        "Naive Bayes": "Naive_Bayes.pkl"
    }
    file_path = f"models/{model_mapping[model_name]}"
    return joblib.load(file_path)

# --- SIDEBAR: MODEL SETTINGS ---
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=50)
st.sidebar.header("Model Settings")

selected_model_name = st.sidebar.selectbox(
    "Choose Prediction Engine",
    ["Random Forest", "XGBoost", "Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes"]
)

st.sidebar.info(f"Currently using: **{selected_model_name}**")

# --- MAIN UI ---
st.title("🏨 Smart Hotel Reservation Analysis")
st.write(f"Predicting cancellation risk using the **{selected_model_name}** model.")

# --- INPUT SECTION (Grouped by Category) ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🕒 Timing")
        lead_time = st.number_input("Lead Time (Days)", 0, 500, 30)
        arrival_month = st.slider("Arrival Month", 1, 12, 10)
        
    with col2:
        st.markdown("### 💰 Pricing & Requests")
        avg_price = st.number_input("Avg Price per Room (€)", 0.0, 500.0, 120.0)
        special_requests = st.selectbox("Special Requests", [0, 1, 2, 3, 4, 5])

# Expandable section for less common features
with st.expander("More Booking Details"):
    c1, c2, c3 = st.columns(3)
    with c1:
        adults = c1.number_input("Adults", 1, 4, 2)
        children = c1.number_input("Children", 0, 3, 0)
    with c2:
        weekend_nights = c2.number_input("Weekend Nights", 0, 10, 1)
        week_nights = c2.number_input("Week Nights", 0, 10, 2)
    with c3:
        market_segment = c3.selectbox("Market Segment", [0, 1, 2, 3, 4])
        parking = c3.selectbox("Parking?", [0, 1])

# --- PREDICTION LOGIC ---
if st.button("RUN PREDICTION"):
    try:
        # 1. Load the specific model chosen
        model = load_selected_model(selected_model_name)
        
        # 2. Prepare Data (Match your X_train column order exactly)
        input_data = pd.DataFrame([[
            adults, children, weekend_nights, week_nights, 0, parking, 
            0, lead_time, 2018, arrival_month, 15, market_segment, 
            0, 0, 0, avg_price, special_requests
        ]], columns=['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month', 'arrival_date', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests'])

        # 3. Predict
        prediction = model.predict(input_data)[0]
        
        # 4. Display Results
        st.markdown("---")
        if prediction == 1:
            st.error(f"### Result: High Cancellation Risk")
            st.write(f"The {selected_model_name} algorithm predicts this guest will likely cancel.")
        else:
            st.success(f"### Result: Low Cancellation Risk")
            st.write(f"The {selected_model_name} algorithm predicts this booking is secure.")
            
    except Exception as e:
        st.error(f"Error: {e}. Make sure the file for {selected_model_name} exists in your 'models/' folder.")

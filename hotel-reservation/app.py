import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hotel Predictor Pro", page_icon="🏨", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_assets(model_name):
    base_path = os.path.dirname(__file__)
    model_mapping = {
        "XGBoost": "XGBoost.pkl",
        "Random Forest": "Random_Forest.pkl",
        "Logistic Regression": "Logistic_Regression.pkl",
        "Decision Tree": "Decision_Tree.pkl",
        "K-Nearest Neighbors": "KNN.pkl",
        "Naive Bayes": "Naive_Bayes.pkl"
    }
    
    model_path = os.path.join(base_path, "model", model_mapping[model_name])
    scaler_path = os.path.join(base_path, "model", "standard_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or Scaler files missing in 'model/' folder!")
        return None, None
        
    return joblib.load(model_path), joblib.load(scaler_path)

# --- SIDEBAR: MODEL SELECTION ---
st.sidebar.header("⚙️ Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose Prediction Engine",
    ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes"]
)

GITHUB_CSV_URL = "https://github.com/Pravena-2024dc04019/ML-Assignment-2/blob/main/hotel-reservation/test.csv"
st.sidebar.markdown(f"[🔗 Click here to download test.csv]({GITHUB_CSV_URL})")

# --- MAIN UI ---
st.title("🏨 Hotel Reservation Analysis & Evaluation")
tabs = st.tabs(["Single Prediction", "Batch Evaluation (CSV)"])

# --- TAB 1: SINGLE PREDICTION ---
with tabs[0]:
    st.subheader("Individual Booking Analysis")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            lead_time = st.number_input("Lead Time (Days)", 0, 500, 30)
            month_options = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            arrival_month = month_options.index(st.selectbox("Arrival Month", month_options, index=9)) + 1
        with col2:
            avg_price = st.number_input("Avg Price per Room (€)", 0.0, 500.0, 120.0)
            special_requests = st.selectbox("Special Requests", [0, 1, 2, 3, 4, 5])

    with st.expander("More Details"):
        c1, c2, c3 = st.columns(3)
        adults = c1.number_input("Adults", 1, 4, 2)
        children = c1.number_input("Children", 0, 3, 0)
        weekend_nights = c2.number_input("Weekend Nights", 0, 10, 1)
        week_nights = c2.number_input("Week Nights", 0, 10, 2)
        market_map = {"Offline TA/TO": 0, "Online TA": 1, "Corporate": 2, "Direct": 3, "Aviation": 4}
        market_segment = market_map[c3.selectbox("Market Segment", list(market_map.keys()))]
        parking = 1 if c3.selectbox("Parking Required?", ["NO", "YES"]) == "YES" else 0

    if st.button("Predict Single"):
        model, scaler = load_assets(selected_model_name)
        features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month', 'arrival_date', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']
        input_df = pd.DataFrame([[adults, children, weekend_nights, week_nights, 0, parking, 0, lead_time, 2018, arrival_month, 15, market_segment, 0, 0, 0, avg_price, special_requests]], columns=features)
        
        scaled_data = scaler.transform(input_df)
        pred = model.predict(scaled_data)[0]
        
        if pred == 1: st.error("Outcome: Likely to Cancel")
        else: st.success("Outcome: Likely to Stay")

# --- TAB 2: BATCH EVALUATION (CSV) ---
with tabs[1]:
    st.subheader("📈 Model Evaluation via Test Data")
    uploaded_file = st.file_uploader("Upload Test CSV (must contain 'booking_status')", type="csv")
    
    if uploaded_file:
        test_df = pd.read_csv(uploaded_file)
        if 'booking_status' not in test_df.columns:
            st.error("CSV must contain a 'booking_status' column for evaluation.")
        else:
            model, scaler = load_assets(selected_model_name)
            
            # Preprocessing
            y_true = test_df['booking_status']
            X_raw = test_df.drop(columns=['booking_status', 'Booking_ID'], errors='ignore')
            
            # Ensure Column Match
            X_scaled = scaler.transform(X_raw)
            y_pred = model.predict(X_scaled)
            y_probs = model.predict_proba(X_scaled)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_probs)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            st.write("### Model Evaluation Metrics")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.2%}")
            col2.metric("AUC Score", f"{auc:.3f}")            
            col3.metric("Precision", f"{prec:.2%}")
            col4.metric("Recall", f"{rec:.2%}")
            col5.metric("F1 Score", f"{f1:.3f}")
            col6.metric("MCC", f"{mcc:.3f}")
            # col_m1.metric("Model Accuracy", f"{acc:.2%}")
            # col_m2.metric("Matthews Correlation (MCC)", f"{mcc:.3f}")
            
            # Confusion Matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(1.5,1))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig, use_container_width=False)
            
            # Classification Report
            st.write("### Classification Report")
            st.text(classification_report(y_true, y_pred))



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="🏥 Smart Health Diagnosis Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for amazing UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-in;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Dataset Creation Functions
@st.cache_data
def create_diabetes_dataset():
    """Create PIMA Indian Diabetes Dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
        'BloodPressure': np.random.normal(69, 19, n_samples).clip(0, 122),
        'SkinThickness': np.random.normal(20, 15, n_samples).clip(0, 99),
        'Insulin': np.random.normal(79, 115, n_samples).clip(0, 846),
        'BMI': np.random.normal(32, 7, n_samples).clip(0, 67),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples)
    }
    
    # Create target based on realistic relationships
    diabetes_risk = (
        (data['Glucose'] > 140) * 0.4 +
        (data['BMI'] > 30) * 0.3 +
        (data['Age'] > 45) * 0.2 +
        (data['Pregnancies'] > 5) * 0.1
    )
    
    data['Outcome'] = (diabetes_risk + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    return pd.DataFrame(data)

@st.cache_data
def create_heart_disease_dataset():
    """Create Heart Disease Dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.normal(131, 17, n_samples).clip(94, 200),
        'chol': np.random.normal(246, 51, n_samples).clip(126, 564),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.normal(149, 22, n_samples).clip(71, 202),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6.2, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 3, n_samples)
    }
    
    # Create target based on realistic relationships
    heart_risk = (
        (data['age'] > 55) * 0.25 +
        (data['trestbps'] > 140) * 0.2 +
        (data['chol'] > 240) * 0.2 +
        (data['thalach'] < 120) * 0.15 +
        (data['exang'] == 1) * 0.2
    )
    
    data['target'] = (heart_risk + np.random.normal(0, 0.1, n_samples) > 0.4).astype(int)
    
    return pd.DataFrame(data)

@st.cache_data
def create_kidney_disease_dataset():
    """Create Kidney Disease Dataset"""
    np.random.seed(42)
    n_samples = 800
    
    data = {
        'age': np.random.randint(20, 90, n_samples),
        'bp': np.random.normal(76, 13, n_samples).clip(50, 180),
        'sg': np.random.uniform(1.005, 1.025, n_samples),
        'al': np.random.randint(0, 6, n_samples),
        'su': np.random.randint(0, 6, n_samples),
        'bgr': np.random.normal(148, 79, n_samples).clip(22, 490),
        'bu': np.random.normal(57, 50, n_samples).clip(1.5, 391),
        'sc': np.random.normal(3, 5, n_samples).clip(0.4, 76),
        'sod': np.random.normal(137, 10, n_samples).clip(4.5, 163),
        'pot': np.random.normal(4.6, 3.2, n_samples).clip(2.5, 47),
        'hemo': np.random.normal(12.5, 2.9, n_samples).clip(3.1, 17.8),
        'pcv': np.random.normal(38, 8, n_samples).clip(9, 54),
        'wc': np.random.normal(8400, 2570, n_samples).clip(2200, 26400),
        'rc': np.random.normal(4.7, 1, n_samples).clip(2.1, 8)
    }
    
    # Create target based on realistic relationships
    kidney_risk = (
        (data['sc'] > 1.2) * 0.3 +
        (data['bu'] > 20) * 0.2 +
        (data['hemo'] < 10) * 0.2 +
        (data['pcv'] < 30) * 0.15 +
        (data['age'] > 60) * 0.15
    )
    
    data['classification'] = (kidney_risk + np.random.normal(0, 0.1, n_samples) > 0.4).astype(int)
    
    return pd.DataFrame(data)

# Model Training Functions
@st.cache_resource 
def train_diabetes_model():
    """Train diabetes prediction model"""
    df = create_diabetes_dataset()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    return model, scaler, accuracy, X.columns.tolist()

@st.cache_resource
def train_heart_model():
    """Train heart disease prediction model"""
    df = create_heart_disease_dataset()
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    return model, scaler, accuracy, X.columns.tolist()

@st.cache_resource
def train_kidney_model():
    """Train kidney disease prediction model"""
    df = create_kidney_disease_dataset()
    X = df.drop('classification', axis=1)
    y = df['classification']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    return model, scaler, accuracy, X.columns.tolist()

# Prediction Functions
def predict_diabetes(input_data):
    """Predict diabetes risk"""
    model, scaler, accuracy, features = train_diabetes_model()
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    return prediction, probability, accuracy

def predict_heart_disease(input_data):
    """Predict heart disease risk"""
    model, scaler, accuracy, features = train_heart_model()
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    return prediction, probability, accuracy

def predict_kidney_disease(input_data):
    """Predict kidney disease risk"""
    model, scaler, accuracy, features = train_kidney_model()
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    return prediction, probability, accuracy

# Utility Functions
def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return "Low", "risk-low"
    elif probability < 0.7:
        return "Medium", "risk-medium"
    else:
        return "High", "risk-high"

def generate_recommendations(disease, risk_level, prediction):
    """Generate health recommendations"""
    recommendations = {
        'diabetes': {
            'Low': [
                "🥗 Maintain a balanced diet with low sugar content",
                "🏃‍♂️ Regular exercise (30 mins daily)",
                "⚖️ Monitor your weight regularly",
                "🩺 Annual health checkups"
            ],
            'Medium': [
                "🍎 Follow a strict diabetic diet plan",
                "💊 Consider consulting an endocrinologist",
                "📊 Monitor blood glucose levels regularly",
                "🚶‍♀️ Increase physical activity",
                "🧘‍♀️ Practice stress management"
            ],
            'High': [
                "🚨 Consult a doctor immediately",
                "💉 Regular blood sugar monitoring required",
                "💊 Follow prescribed medication strictly",
                "🏥 Regular medical supervision needed",
                "🚫 Avoid high-carb and sugary foods"
            ]
        },
        'heart': {
            'Low': [
                "❤️ Maintain heart-healthy diet",
                "🏃‍♂️ Regular cardio exercise",
                "🚭 Avoid smoking and excessive alcohol",
                "😴 Get adequate sleep (7-8 hours)"
            ],
            'Medium': [
                "🩺 Consult a cardiologist",
                "💊 Monitor blood pressure regularly",
                "🧂 Reduce sodium intake",
                "🏊‍♀️ Engage in low-impact exercises",
                "📊 Regular cholesterol checkups"
            ],
            'High': [
                "🚨 Seek immediate medical attention",
                "💊 Follow cardiac medication strictly",
                "🏥 Regular cardiac monitoring required",
                "🚫 Avoid strenuous activities without approval",
                "📞 Keep emergency contacts handy"
            ]
        },
        'kidney': {
            'Low': [
                "💧 Stay well hydrated",
                "🧂 Limit salt intake",
                "🥗 Maintain healthy diet",
                "🩺 Regular health checkups"
            ],
            'Medium': [
                "🩺 Consult a nephrologist",
                "💊 Monitor kidney function regularly",
                "💧 Adequate but not excessive fluid intake",
                "🚫 Avoid nephrotoxic medications"
            ],
            'High': [
                "🚨 Urgent medical consultation required",
                "🏥 Regular kidney function monitoring",
                "💊 Strict medication compliance",
                "🍎 Follow renal diet plan",
                "📊 Regular lab tests required"
            ]
        }
    }
    
    return recommendations.get(disease, {}).get(risk_level, ["Consult healthcare professional"])

# Main Application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Smart Health Diagnosis Agent</h1>
        <p>AI-Powered Health Risk Assessment & Preventive Care Guidance</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("## 🎯 Select Disease")
    disease_option = st.sidebar.selectbox(
        "Choose disease to predict:",
        ["Diabetes", "Heart Disease", "Kidney Disease"]
    )
    
    # Main content based on selection
    if disease_option == "Diabetes":
        diabetes_prediction_page()
    elif disease_option == "Heart Disease":
        heart_disease_prediction_page()
    elif disease_option == "Kidney Disease":
        kidney_disease_prediction_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;">
        <h3>⚠️ Medical Disclaimer</h3>
        <p>This tool is for educational purposes only and should not replace professional medical advice. 
        Always consult with healthcare professionals for proper diagnosis and treatment.</p>
        <p><strong>Developed with ❤️ using AI & Machine Learning</strong></p>
    </div>
    """, unsafe_allow_html=True)

def diabetes_prediction_page():
    st.markdown("## 🍎 Diabetes Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Enter Your Information")
        
        pregnancies = st.number_input("Number of Pregnancies", 0, 17, 0)
        glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
        bp = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Additional Parameters")
        
        insulin = st.slider("Insulin Level (mu U/ml)", 0, 846, 80)
        bmi = st.slider("BMI", 0.0, 67.0, 25.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5)
        age = st.slider("Age", 21, 81, 30)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🔍 Predict Diabetes Risk", key="diabetes_predict"):
        input_data = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
        prediction, probability, accuracy = predict_diabetes(input_data)
        
        risk_prob = probability[1] if prediction == 1 else probability[0]
        risk_level, risk_class = get_risk_level(risk_prob)
        
        # Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>🎯 Model Accuracy</h3>
                <h2>{accuracy:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h3>📊 Risk Probability</h3>
                <h2>{risk_prob:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h3>⚠️ Risk Level</h3>
                <h2>{risk_level}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction Result
        result_text = "High Risk of Diabetes" if prediction == 1 else "Low Risk of Diabetes"
        st.markdown(f"""
        <div class="prediction-card {risk_class}">
            <h2>🏥 Prediction Result</h2>
            <h1>{result_text}</h1>
            <p>Risk Level: {risk_level} ({risk_prob:.1%})</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        recommendations = generate_recommendations('diabetes', risk_level, prediction)
        st.markdown("### 💡 Health Recommendations")
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Diabetes Risk Gauge"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def heart_disease_prediction_page():
    st.markdown("## ❤️ Heart Disease Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Basic Information")
        
        age = st.slider("Age", 29, 77, 50)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", 
                         ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
        chol = st.slider("Cholesterol (mg/dl)", 126, 564, 245)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LVH"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Advanced Parameters")
        
        thalach = st.slider("Maximum Heart Rate", 71, 202, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0)
        slope = st.selectbox("Slope of Peak Exercise ST", 
                           ["Upsloping", "Flat", "Downsloping"])
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🔍 Predict Heart Disease Risk", key="heart_predict"):
        # Convert categorical to numerical
        sex_val = 1 if sex == "Male" else 0
        cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
        fbs_val = 1 if fbs == "Yes" else 0
        restecg_val = ["Normal", "ST-T Abnormality", "LVH"].index(restecg)
        exang_val = 1 if exang == "Yes" else 0
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
        thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
        
        input_data = [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, 
                     thalach, exang_val, oldpeak, slope_val, ca, thal_val]
        
        prediction, probability, accuracy = predict_heart_disease(input_data)
        
        risk_prob = probability[1] if prediction == 1 else probability[0]
        risk_level, risk_class = get_risk_level(risk_prob)
        
        # Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>🎯 Model Accuracy</h3>
                <h2>{accuracy:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h3>📊 Risk Probability</h3>
                <h2>{risk_prob:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h3>⚠️ Risk Level</h3>
                <h2>{risk_level}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction Result
        result_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
        st.markdown(f"""
        <div class="prediction-card {risk_class}">
            <h2>❤️ Prediction Result</h2>
            <h1>{result_text}</h1>
            <p>Risk Level: {risk_level} ({risk_prob:.1%})</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        recommendations = generate_recommendations('heart', risk_level, prediction)
        st.markdown("### 💡 Health Recommendations")
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Heart Disease Risk Gauge"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def kidney_disease_prediction_page():
    st.markdown("## 🫘 Kidney Disease Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Basic Parameters")
        
        age = st.slider("Age", 20, 90, 50)
        bp = st.slider("Blood Pressure (mm Hg)", 50, 180, 80)
        sg = st.slider("Specific Gravity", 1.005, 1.025, 1.020)
        al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
        su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
        bgr = st.slider("Blood Glucose Random (mgs/dl)", 22, 490, 120)
        bu = st.slider("Blood Urea (mgs/dl)", 1.5, 391.0, 30.0)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Lab Results")
        
        sc = st.slider("Serum Creatinine (mgs/dl)", 0.4, 76.0, 1.0)
        sod = st.slider("Sodium (mEq/L)", 4.5, 163.0, 137.0)
        pot = st.slider("Potassium (mEq/L)", 2.5, 47.0, 4.5)
        hemo = st.slider("Hemoglobin (gms)", 3.1, 17.8, 12.5)
        pcv = st.slider("Packed Cell Volume", 9, 54, 38)
        wc = st.slider("White Blood Cell Count (cells/cumm)", 2200, 26400, 8000)
        rc = st.slider("Red Blood Cell Count (millions/cmm)", 2.1, 8.0, 4.7)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🔍 Predict Kidney Disease Risk", key="kidney_predict"):
        input_data = [age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc]
        
        prediction, probability, accuracy = predict_kidney_disease(input_data)
        
        risk_prob = probability[1] if prediction == 1 else probability[0]
        risk_level, risk_class = get_risk_level(risk_prob)
        
        # Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>🎯 Model Accuracy</h3>
                <h2>{accuracy:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h3>📊 Risk Probability</h3>
                <h2>{risk_prob:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h3>⚠️ Risk Level</h3>
                <h2>{risk_level}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction Result
        result_text = "High Risk of Kidney Disease" if prediction == 1 else "Low Risk of Kidney Disease"
        st.markdown(f"""
        <div class="prediction-card {risk_class}">
            <h2>🫘 Prediction Result</h2>
            <h1>{result_text}</h1>
            <p>Risk Level: {risk_level} ({risk_prob:.1%})</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        recommendations = generate_recommendations('kidney', risk_level, prediction)
        st.markdown("### 💡 Health Recommendations")
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Kidney Disease Risk Gauge"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Additional Features
def show_model_performance():
    """Display model performance metrics"""
    st.markdown("## 📊 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🍎 Diabetes Model")
        _, _, accuracy, _ = train_diabetes_model()
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.info("Random Forest Classifier with feature scaling")
        
    with col2:
        st.markdown("### ❤️ Heart Disease Model")
        _, _, accuracy, _ = train_heart_model()
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.info("Random Forest Classifier with comprehensive features")
        
    with col3:
        st.markdown("### 🫘 Kidney Disease Model")
        _, _, accuracy, _ = train_kidney_model()
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.info("Random Forest with lab parameter analysis")

def show_health_tips():
    """Display general health tips"""
    st.markdown("## 💡 General Health Tips")
    
    tips = {
        "🥗 Nutrition": [
            "Eat a balanced diet rich in fruits and vegetables",
            "Limit processed foods and added sugars",
            "Stay hydrated with 8-10 glasses of water daily",
            "Include lean proteins and whole grains"
        ],
        "🏃‍♂️ Exercise": [
            "Aim for 150 minutes of moderate exercise weekly",
            "Include both cardio and strength training",
            "Take regular breaks from sitting",
            "Find activities you enjoy to stay consistent"
        ],
        "😴 Sleep": [
            "Get 7-9 hours of quality sleep nightly",
            "Maintain a consistent sleep schedule",
            "Create a relaxing bedtime routine",
            "Limit screen time before bed"
        ],
        "🧘‍♀️ Mental Health": [
            "Practice stress management techniques",
            "Stay socially connected",
            "Engage in hobbies and activities you enjoy",
            "Seek professional help when needed"
        ]
    }
    
    for category, tip_list in tips.items():
        with st.expander(category):
            for tip in tip_list:
                st.markdown(f"• {tip}")

def show_emergency_contacts():
    """Display emergency contact information"""
    st.markdown("## 🚨 Emergency Contacts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🏥 Emergency Services
        - **Emergency**: 911 (US) / 108 (India)
        - **Poison Control**: 1-800-222-1222
        - **Crisis Hotline**: 988
        """)
        
    with col2:
        st.markdown("""
        ### 👨‍⚕️ When to Seek Help
        - Chest pain or pressure
        - Difficulty breathing
        - Severe abdominal pain
        - Loss of consciousness
        - Severe bleeding
        """)

# Sidebar Navigation
def sidebar_navigation():
    """Enhanced sidebar with additional features"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔧 Additional Features")
    
    if st.sidebar.button("📊 Model Performance"):
        show_model_performance()
        
    if st.sidebar.button("💡 Health Tips"):
        show_health_tips()
        
    if st.sidebar.button("🚨 Emergency Info"):
        show_emergency_contacts()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📱 Export Results")
    
    if st.sidebar.button("📄 Generate PDF Report"):
        st.sidebar.success("Feature coming soon!")
        
    if st.sidebar.button("📧 Email Results"):
        st.sidebar.success("Feature coming soon!")

def symptom_checker():
    """Simple symptom checker with NLP-like functionality"""
    st.markdown("## 🔍 Symptom Checker")
    
    symptoms_input = st.text_area(
        "Describe your symptoms in natural language:",
        placeholder="E.g., I feel tired, have frequent urination, and increased thirst"
    )
    
    if st.button("🔍 Analyze Symptoms"):
        if symptoms_input:
            # Simple keyword mapping
            diabetes_keywords = ['thirsty', 'urination', 'tired', 'fatigue', 'hungry', 'weight loss', 'blurred vision']
            heart_keywords = ['chest pain', 'shortness of breath', 'fatigue', 'dizziness', 'palpitations', 'swelling']
            kidney_keywords = ['back pain', 'urination changes', 'swelling', 'fatigue', 'nausea', 'metallic taste']
            
            symptoms_lower = symptoms_input.lower()
            
            diabetes_score = sum(1 for keyword in diabetes_keywords if keyword in symptoms_lower)
            heart_score = sum(1 for keyword in heart_keywords if keyword in symptoms_lower)
            kidney_score = sum(1 for keyword in kidney_keywords if keyword in symptoms_lower)
            
            st.markdown("### 📊 Symptom Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>🍎 Diabetes Match</h3>
                    <h2>{diabetes_score}/7</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>❤️ Heart Disease Match</h3>
                    <h2>{heart_score}/6</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>🫘 Kidney Disease Match</h3>
                    <h2>{kidney_score}/6</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations based on highest score
            max_score = max(diabetes_score, heart_score, kidney_score)
            if max_score > 0:
                if diabetes_score == max_score:
                    st.warning("Consider taking the Diabetes Risk Assessment")
                elif heart_score == max_score:
                    st.warning("Consider taking the Heart Disease Risk Assessment")
                else:
                    st.warning("Consider taking the Kidney Disease Risk Assessment")
            else:
                st.info("No specific disease indicators detected. Maintain regular health checkups.")

if __name__ == "__main__":
    main()
    
    # Add sidebar navigation
    sidebar_navigation()
    
    # Add symptom checker in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🔍 Symptom Checker"):
            symptom_checker()
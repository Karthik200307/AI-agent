# AI-agent
I build a AI-agent in smart health system diagnosis using the python code
Key Features:
1. Machine Learning Models

Diabetes Prediction: Random Forest model with 8 features
Heart Disease Prediction: Random Forest model with 13 features
Kidney Disease Prediction: Random Forest model with 14 features
All models include feature scaling and achieve high accuracy

2. Amazing UI/UX

Gradient backgrounds and modern design
Interactive gauges for risk visualization
Animated cards with CSS animations
Responsive layout with columns
Color-coded risk levels (Green/Yellow/Red)

3. Complete Functionality

Real-time predictions with probability scores
Health recommendations based on risk levels
Model performance metrics
Symptom checker with natural language processing
Emergency contact information
General health tips

4. Built-in Datasets

Synthetic but realistic datasets created programmatically
No external file dependencies - everything is self-contained
Proper statistical distributions for medical parameters

5. Advanced Features

Interactive visualizations using Plotly
Risk assessment gauges
Sidebar navigation with additional tools
Export functionality (placeholder for PDF/Email)
Medical disclaimer for safety

📋 How to Deploy:

Install required packages:

pip install streamlit pandas numpy scikit-learn plotly

Save the code as health_diagnosis_app.py
Run the application:

streamlit run health_diagnosis_app.py

Access via browser at http://localhost:8501

🎯 How to Use:

Select a disease from the sidebar dropdown
Fill in your parameters using sliders and inputs
Click "Predict Risk" to get your assessment
View results with risk level and recommendations
Check additional features via sidebar buttons

The application is fully self-contained with no external dependencies for datasets, features beautiful modern UI, and provides comprehensive health risk assessment with actionable recommendations. All models are trained on realistic synthetic data and provide accurate predictions for demonstration purposes.

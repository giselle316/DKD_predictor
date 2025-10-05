DKD Predictor Web Tool

A Flask-based web application for predicting kidney disease risk using pathology images and clinical data.

Features:
- Crescent image analysis
- Fibrosis image analysis
- Clinical risk prediction 
- Interactive web interface: upload images and enter clinical data to get predictions

Usage:
1. Run the application:
   python app.py
2. Open your browser and go to:
   http://127.0.0.1:5000/
3. Upload images and enter clinical data to view predictions.

Notes:
- Uploaded images are temporarily stored in the uploads/ folder
- For production deployment, set debug=False in app.py

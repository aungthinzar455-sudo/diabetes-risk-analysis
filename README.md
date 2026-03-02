# 🩺 Diabetes Risk Prediction – Clinical Decision Support System

An AI-powered full-stack web application designed to assist healthcare professionals in assessing diabetes risk using machine learning.

This system allows secure login, real-time patient risk prediction, analytics dashboard monitoring, patient profile tracking, and automated PDF report generation.

---

## 🔥 Key Features

- 🔐 Secure Admin Login Authentication
- 🧠 Machine Learning Risk Prediction (Scikit-learn)
- 📊 Interactive Analytics Dashboard
- 📈 Risk Distribution Visualization (Chart.js)
- 👤 Patient Profile View
- 📄 Automated PDF Medical Report Generation
- 🗂 Historical Prediction Storage (CSV-based persistence)
- 🎨 Modern Responsive UI Design

---

## 🏗 System Architecture

- Machine Learning model trained on health dataset
- Flask backend API for prediction handling
- JavaScript Fetch API for dynamic frontend updates
- Pandas for analytics processing
- ReportLab for PDF report generation
- Session-based authentication system

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Backend Logic |
| Flask | Web Framework |
| Scikit-learn | Machine Learning Model |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| HTML/CSS | Frontend UI |
| JavaScript | Dynamic Updates |
| Chart.js | Dashboard Visualization |
| ReportLab | PDF Report Generation |

---

## 📷 Application Screenshots

### 🔐 Login Page
![Login](login.png)

### 📊 Analytics Dashboard
![Dashboard](dashboard.png)

### 🧠 Risk Prediction
![Prediction](prediction.png)

### 👤 Patient Profile
![Profile](patient profile.png)

### 📄 Generated PDF Report
![PDF](record.png)

---

## 📊 Risk Classification Logic

- Low Risk → Probability < 30%
- Moderate Risk → 30% – 69%
- High Risk → ≥ 70%

---

## 🚀 How To Run Locally

```bash
git clone https://github.com/aungthinzar455-sudo/diabetes-risk-analysis.git
cd diabetes-risk-analysis
pip install -r requirements.txt
python app.py


Open browser:

http://127.0.0.1:5000

🔐 Demo Login

Username:

admin

Password:

1234
📈 Future Improvements

Database Integration (MySQL / PostgreSQL)

Role-based Authentication (Doctor / Admin)

Model Performance Metrics (ROC Curve, Confusion Matrix)

Deployment to Cloud (Render / Railway / AWS)

REST API Documentation

💼 Project Purpose

This project demonstrates:

End-to-end Machine Learning integration

Full-stack development with Flask

Secure session handling

Data visualization implementation

Real-world healthcare analytics workflow

👩‍💻 Author

Thinzar Aung
Data Analytics & Machine Learning

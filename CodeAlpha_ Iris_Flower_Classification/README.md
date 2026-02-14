# 🌸 Iris Flower Classification using Machine Learning

This project is part of **Task 1** of my **Data Science Internship at CodeAlpha**.  
It demonstrates a complete **end-to-end machine learning classification workflow** using the classic Iris dataset.

---

## 📌 Project Overview

The objective of this project is to build a **machine learning model** that can accurately classify the species of an Iris flower based on its physical measurements.

### 🌼 Iris Flower Species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

### 📐 Input Features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## 🧠 Problem Statement

Given the measurements of an Iris flower, predict its species using a supervised machine learning classification model.

---

## 🛠️ Tech Stack Used

### 🔹 Programming Language
- Python

### 🔹 Libraries & Frameworks
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### 🔹 Machine Learning Algorithm
- K-Nearest Neighbors (KNN)

---

## 📂 Dataset

- Source: Kaggle – Iris Dataset  
- Total Samples: 150  
- Classes: 3 (50 samples per class)  
- Dataset is clean and contains no missing values

---

## 🔄 Project Workflow

1. Load and explore the dataset  
2. Perform exploratory data analysis (EDA)  
3. Visualize feature relationships using pair plots  
4. Prepare features and target variables  
5. Split the dataset into training and testing sets (80% / 20%)  
6. Train the KNN classification model  
7. Evaluate model performance  
8. Test the model on new, unseen input data  

---

## 📊 Model Performance

- **Accuracy Achieved:** `100%`
- **Evaluation Metrics Used:**
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

The confusion matrix shows zero misclassification, indicating excellent model performance on the test dataset.

---

## 🔍 Sample Prediction

The model can predict the species of a new flower using custom input values:

```python
Sample Input: [5.1, 3.5, 1.4, 0.2]
Predicted Output: Iris-setosa
````

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone <your-repository-link>
cd iris-flower-classification
```

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 4️⃣ Run the Project

```bash
python iris_classification.py
```

---

## 🌱 What I Learned

* Fundamentals of **supervised learning**
* Hands-on experience with **classification algorithms**
* Data preprocessing and feature selection
* Model evaluation using real metrics
* End-to-end ML project workflow using Scikit-learn

This project helped me strengthen my foundation in **Data Science and Machine Learning** and gain confidence in building practical ML models.

---

## 📌 Internship Information

* **Internship Domain:** Data Science
* **Organization:** CodeAlpha
* **Task:** Task 1 – Iris Flower Classification

---

## 📎 Future Improvements

* Compare multiple classification models
* Add hyperparameter tuning
* Deploy the model using Streamlit
* Convert into a web-based prediction app

---

## 📬 Contact

If you’d like to connect or discuss this project, feel free to reach out!

🔗 **LinkedIn:** 
📁 **GitHub:** 

---

⭐ If you found this project useful, consider giving it a star!


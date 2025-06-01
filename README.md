## Heart Disease Prediction using SVM and PCA ❤️
  This project predicts whether a patient is likely to suffer from heart disease using machine learning. It uses Support Vector Machine (SVM) classification, along with Principal Component Analysis (PCA) for 
  dimensionality reduction to compare model performance with and without feature reduction.

  ---

## 📂 Dataset
  The dataset used is heart.csv, a common dataset for cardiovascular disease prediction. It contains 13 features related to patient health and a binary target variable indicating presence (1) or absence (0) of    
  heart disease.

## 🛠️ Technologies Used
  - Python
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn
  
## 🚀 How it Works
  1. **Data Loading & Preprocessing**
     - Dataset is loaded using pandas.
     - Categorical features are encoded using LabelEncoder.
  2. **Train-Test Split**
     - Dataset is split into training and test sets (80/20 split).
  3. **Dimensionality Reduction**
     - PCA is applied to reduce the feature space to 5 principal components.
  4. **Model Training**
     - Two SVM models are trained:
       - One with the original features.
       - One after applying PCA.
  5. **Evaluation**
     - Both models are evaluated using accuracy score.
     - A comparison chart is shown to visualize the effect of PCA on performance.
  6. **New Prediction**
     - A new patient's data (numerical and preprocessed) is input and classified using the PCA-trained model.

## 📊 Results
  - **Accuracy without PCA:** ~X.XX (will be printed when run)
  - **Accuracy with PCA:** ~X.XX (will be printed when run)
  > The accuracy comparison is visualized using a bar chart.

## ▶️ How to Run
  1. Make sure you have Python installed.
  2. Install the required libraries:
  ```bash
  pip install numpy pandas matplotlib scikit-learn
  ```
  3. Place the heart.csv dataset in the same directory as the script.
  4. Run the script:
  ```bash
  python heart_disease_prediction.py
  ```
## 📌 Notes
  - This project uses only SVM for classification, but it can be extended to test other classifiers.
  - PCA reduces features from 13 to 5 for improved efficiency.
  - The model assumes all inputs are numerical; categorical features must be encoded before prediction.
  
  
     




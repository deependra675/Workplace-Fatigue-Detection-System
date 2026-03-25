# Office Fatigue Prediction

This project is a machine learning application that predicts an employee's fatigue level based on work patterns, sleep, and daily activity. It demonstrates data generation, model training, evaluation, and real-time prediction using a Random Forest classifier.

---

Project Structure:

fatigue_prediction_project/
│
├─ data/
│   ├─ raw/
│   │   ├─ fatigue_10k.csv        # Generated training dataset
│   │   └─ predict_input.csv      # Sample new employee data for prediction
│
├─ models/
│   ├─ random_forest_fatigue.pkl  # Trained Random Forest model
│   └─ feature_columns.pkl        # Feature columns saved during training
│
├─ src/
│   ├─ data_generation.py         # Script to generate synthetic dataset
│   ├─ train.py                   # Train the model and save it
│   ├─ evaluate.py                # Evaluate model performance with metrics and plots
│   ├─ predict.py                 # Predict fatigue levels for new employees
│   └─ preprocess.py              # Data preprocessing and feature preparation
│
└─ README.md

---

Features:

- Predicts fatigue risk levels: Low, Medium, High
- Handles both single and multiple employee predictions
- Generates and scales features for numerical and categorical data
- Visualizes model performance via classification report and confusion matrix
- Displays feature importance for insight into key fatigue factors

---

Getting Started:

1. Clone the repository:
   git clone https://github.com/yourusername/fatigue_prediction_project.git
   cd fatigue_prediction_project

2. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib seaborn

3. Generate the dataset (optional):
   python src/data_generation.py
   - Creates fatigue_10k.csv in data/raw/

4. Train the model:
   python src/train.py
   - Trains a Random Forest classifier
   - Saves the model in models/random_forest_fatigue.pkl
   - Saves feature columns in models/feature_columns.pkl

5. Evaluate the model:
   python src/evaluate.py
   - Prints classification metrics
   - Displays confusion matrix and feature importance chart

6. Make predictions:
   - Add new employee data to data/raw/predict_input.csv
   - Run:
     python src/predict.py
   - Outputs employee_id and predicted fatigue levels

---

Sample Prediction Output:

employee_id | predicted_fatigue_level
------------|-----------------------
1           | High
2           | Medium
3           | Low

---

How It Works:

1. Data Generation: Creates synthetic employee work patterns with fatigue-related features.
2. Preprocessing:
   - Scales numerical features
   - One-hot encodes categorical features (time_of_day)
3. Model Training: Random Forest classifier trained on 10,000 synthetic records.
4. Evaluation: Prints metrics and visualizes feature importance.
5. Prediction: Aligns new data with training features to avoid mismatches, then predicts fatigue risk.

---

Key Libraries Used:

- pandas / numpy — data handling
- scikit-learn — preprocessing, training, evaluation
- matplotlib / seaborn — visualization
- joblib — model and feature persistence

---

Notes:

- The dataset is synthetic for demonstration purposes.
- Feature scaling and one-hot encoding are saved to ensure predictions are consistent with the trained model.
- Project is structured to be easy to extend, e.g., integrating with a web app or Streamlit dashboard.

---

Author:

Deependra Sisodia – MCA, aspiring data scientist/machine learning engineer
GitHub: https://github.com/deependra675

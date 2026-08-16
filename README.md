# Credit Score Classifier #

## Problem Statement ##
Develop a machine learning system that classifies customer credit scores into categories (Good, Standard, Poor) using financial data, ensuring accurate prediction, efficient preprocessing, and deployment through an interactive web application.

## Data Description ##
The dataset consists of customer financial and behavioral attributes, including both numerical and categorical features used to assess creditworthiness.It is used to classify individuals into credit score categories such as Good, Standard, and Poor based on their financial profile.

## How to Run this App ##

Follow the steps below from the project folder to install the required libraries and launch the Streamlit app.

1. Open a terminal in the project root folder.

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. If Streamlit is not already included in the requirements, install it separately:

   ```bash
   pip install streamlit
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

6. After the app starts, a browser tab will open automatically with the local Streamlit dashboard (usually at http://localhost:8501).

7. Upload a CSV file from the app and click the prediction button to generate credit score results.

8. To stop the app, press Ctrl + C in the terminal.

## GitHub Repository ##
https://github.com/anilp-bits/credit-score-classifier

## Model Evaluation and Comparison Table ##

| ML Model Name | Accuracy | Precision | Recall | F1 | MCC | AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.6012 | 0.5988 | 0.6012 | 0.5821 | 0.2931 | 0.7606 |
| **Decision Tree** | 0.7040 | 0.7080 | 0.7040 | 0.7047 | 0.5095 | 0.8513 |
| **KNN** | 0.6269 | 0.6278 | 0.6269 | 0.6257 | 0.3737 | 0.7733 |
| **Naive Bayes** | 0.5710 | 0.6240 | 0.5710 | 0.5745 | 0.3688 | 0.7455 |
| **Random Forest** | 0.7904 | 0.7902 | 0.7904 | 0.7902 | 0.6514 | 0.9124 |


## Performance of Each Model on the Dataset ##

| ML Model Name | Observation about model performance |
|:---|:---:|
| **Logistic Regression** | Provides steady results but doesn’t handle complex patterns in the data very well |
| **Decision Tree** | Learns relationships clearly but can easily overfit if not controlled properly |
| **KNN** | Works based on similarity between data points but becomes slow with larger datasets |
| **Naive Bayes** | Runs very fast but may miss important relationships because of its simple assumptions |
| **Random Forest** | Produces the most reliable results by combining multiple models and reducing overfitting |
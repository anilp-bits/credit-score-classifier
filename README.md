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


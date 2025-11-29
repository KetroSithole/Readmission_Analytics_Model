import pandas as pd
import numpy as np

# Data viz
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

# Data prep
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score,
    roc_curve, auc, roc_auc_score, classification_report
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

# ------------------------------------------------------
# Function to load and return your dataframe
# ------------------------------------------------------
def load_patient_data(path='Expanded_Patient_Readmission_Data.csv'):
    df = pd.read_csv(path)
    print(df)
    return df

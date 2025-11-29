
import pandas as pd
import numpy as np
#Data viz
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

import pandas as pd
import numpy as np

#Data prep
import sklearn
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

data_path = 'Expanded_Patient_Readmission_Data.csv'
#Loading the data
df=pd.read_csv(data_path)
#display the data
print(df)


import pandas as pd
import os 
import numpy as np
import plotly.express as px

# For modeling
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

import plotly.io as pio
pio.renderers
pio.renderers.default = "vscode"


# Defining function for training model
def train_model(df):

    """ Function trains ensemble voting classifier to predict death_event
    based on input df.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing heart failure data

    Returns
    ----------
    cmatrix: list
        confusion matrix from model
    model: Classifier
        ensemble classification model from VotingClassifier
    out_predictions: DataFrame
        contains predictions for all input data assembled using
        the held out dataset from cross-validation
    importance_df: DataFrame
        contains feature importance from the rf model ensembled
        into the voting classifier
    """

    # Converting object columns to string type
    str_columns = df.select_dtypes(include="object").columns
    df[str_columns]=df[str_columns].astype("string")

    # Splitting dependent/independent var's 
    X, y = df.drop(columns = ['DEATH_EVENT']), df['DEATH_EVENT']

    # Applying standard scale to independent variables
    standard_scaler = StandardScaler()
    X_scaled = standard_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns) 

    # Test/Train split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    # Defining models and then ensembling with voting classifier
    m1 = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state = 42)
    m2 = svm.SVC(random_state = 42)
    m3 = LogisticRegression(class_weight = 'balanced', random_state = 42)
    em = VotingClassifier(
     estimators=[('rf', m1), ('svm', m2),  ('lr', m3)],
     voting='hard')
    predictions = []

    # Iterating through models
    for model in [m1, m2, m3, em]:
        # Measuring accuracy usign 5 fold cross validation
        scores = cross_val_score(model, X, y, cv=6)

        # Using 5 fold cross validation to generate predictions  +
        # Create confusion matrix for all datapoints
        y_pred_all = cross_val_predict(model, X, y, cv=6)
        cmatrix = confusion_matrix(y, y_pred_all)
        predictions.append(y_pred_all)

        # Training final model that can be used to make predictions on future data
        model.fit(X_train, y_train)

    # Determining feature importance for voting classifier
    feature_importance = em.estimators_[0].feature_importances_
    columns = X_scaled.columns.to_list()
    importance_df = pd.DataFrame(np.array([feature_importance, columns]).T, 
                                 columns = ['Importance', 'Feature Name'])
    out_predictions = pd.DataFrame(predictions[-1], columns = ['Prediction'])
    
    return cmatrix, model, out_predictions, importance_df

# loading Dataset
base_path = os.path.dirname(__file__)
file_name = 'heart_failure_clinical_records_dataset.csv'
total_path = base_path + '//Data//' + file_name
df = pd.read_csv(total_path)

# Calling function to train model
cmatrix, clf, predictions, importance_df = train_model(df)

# Outputting results
out_df = df.join(predictions)
out_df.to_csv(base_path + '//Data//' + 'heart_failure_clinical_records_dataset,predictions.csv')
importance_df.to_csv(base_path + '//Data//' + 'feature_importance.csv')

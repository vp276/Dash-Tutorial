# Importing packages
import dash
from dash import Dash, dcc, html, dcc, Input, Output, ctx, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import os
import numpy as np

#For modeling
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


#loading Dataset
base_path = os.path.dirname(__file__)
file_name = 'heart_failure_clinical_records_dataset.csv'
total_path = base_path + '\\Data\\' + file_name
df = pd.read_csv(total_path)


#Defining function for training ml model
def train_model(df):

    X, y = df.drop(columns = ['DEATH_EVENT']), df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    #Defining parameters for gridsearch
    parameters = {'max_depth':[2, 4, 6]}

    #Training
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    #Predicting and making confusion matrix
    y_pred = clf.predict(X_test)
    cmatrix = confusion_matrix(y_test, y_pred)

    return cmatrix, clf

#Training model
cmatrix, model = train_model(df)
X_cols = df.drop(columns = 'DEATH_EVENT')

#Setting theme
colors = {'background': '#A5D8DD',
        'text': '#7FDBFF'}

# Initialize the app
dash_app = Dash(__name__)
app = dash_app.server

# App layout
dash_app.layout = html.Div(style={'backgroundColor': colors['background']}, 
    children =[
        html.Div(children= [
             html.H1('Heart Failure Prediction'),
             dcc.Markdown('A comprehensive tool for examining factors impacting heart failure'),
            
            #Filters
             html.Div(children = [ 
                html.Label('Blood Pressure'),
                dcc.Dropdown(
                    id = 'BP-Filter',
                    options = [{"label": i, "value": i} for i in df['high_blood_pressure'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('Sex'),
                dcc.Dropdown(
                    id = 'Sex-Filter',
                    options = [{"label": i, "value": i} for i in df['sex'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('Anaemia'),
                dcc.Dropdown(
                    id = 'Anaemia-Filter',
                    options = [{"label": i, "value": i} for i in df['anaemia'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values")
             ],style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            #First row of graphs
            html.Div([

                #First column
                html.Div(children = [

                    dcc.Graph(id = 'Factor-Barplot',
                            figure={})
                    ],style={'width': '49%', 'display': 'inline-block'}),

                #Second column
                html.Div([
                    dcc.Graph(id = 'Age-Plot',
                            figure={})
                ],style={'width': '49%', 'display': 'inline-block','position': 'fixed'})


            ], className = 'row'),

            html.Br(),
            #Second row of graphs
            html.Div([

                #First column
                html.Div(children = [
                    dcc.Graph(id = 'confusion-matrix',
                            figure={})
                    ],style={'width': '49%', 'display': 'inline-block'}),

                #Second column
                html.Div([
                    dcc.Graph(id = 'feature-importance',
                            figure={})
                ],style={'width': '49%', 'display': 'inline-block'})


            ], className = 'row'),
            html.Br(),

            html.Div([
                html.H3('Data Sources:'),
                html.Div([
                    html.Div(children = [
                        html.Div([
                            dcc.Markdown('Dataset: ')
                        ], style={'display': 'inline-block'}),
                        html.Div([
                            html.A("Kaggle Dataset", 
                                   href='https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv', target="_blank")
                        ], style={'display': 'inline-block'})
                    ], className="row"
                )
                ])
             ])
        ])
])


#callback for top row
@callback(
    [Output(component_id='Factor-Barplot', component_property='figure'),
    Output(component_id='Age-Plot', component_property='figure')],
    [Input('BP-Filter', 'value'),
    Input('Sex-Filter', 'value'),
    Input('Anaemia-Filter', 'value')]
)
def update_output_div(bp, sex, anaemia):

    #Checking which input was fired
    ctx = dash.callback_context

    #Making copy of DF
    filtered_df = df

    bp_list, sex_list,anaemia_list  = [], [], []

    #Filtering for blood pressure
    if bp== "all_values":
        bp_list = filtered_df['high_blood_pressure'].drop_duplicates()
    else:
        bp_list = [bp]

    #Filtering for sex
    if sex== "all_values":
        sex_list = filtered_df['sex'].drop_duplicates()
    else:
        sex_list = [sex]
    
    #Filtering for Anaemia
    if anaemia== "all_values":
        anaemia_list = filtered_df['anaemia'].drop_duplicates()
    else:
        anaemia_list = [anaemia]
    
    #Applying filters to dataframe
    filtered_df = filtered_df[(filtered_df['high_blood_pressure'].isin(bp_list)) &
                              (filtered_df['sex'].isin(sex_list)) &
                               (filtered_df['anaemia'].isin(anaemia_list))]



    factor_fig = px.histogram(filtered_df, x= 'age', color = 'diabetes')
                            
    
    age_fig = px.scatter(filtered_df,
                                      x="age", y="platelets", color = "DEATH_EVENT", title = "Scatterplot")
    
    return factor_fig, age_fig


#callback for second row
@callback(
    [Output(component_id='confusion-matrix', component_property='figure'),
    Output(component_id='feature-importance', component_property='figure')],
    Input('Sex-Filter', 'value')
)
def update_model(value):

    #Making copy of df
    confusion = cmatrix
    model_copy = model
    x_copy = X_cols

    #Aggregating confusion dataframe and plotting
    y_true = confusion[:,0]
    y_pred = confusion[:,1]
    agg_confusion = confusion_matrix(y_true, y_pred)
    #print(y_true)
    confusion_fig = px.imshow(agg_confusion, 
                              labels=dict(x="Predicted Value", 
                                y="True Value", color="Prediction"),
                                aspect="auto",
                                text_auto=True,
                                title = "Confusion Matrix - Predicted vs Actual Values")
    
    #Calculating feature imporance
    importances = model_copy.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_copy.estimators_], axis=0)
    df_importance = pd.DataFrame(list(zip(x_copy, importances, std)), 
                                 columns = ['Feature Name','Importance', 'Std'])
    #importances.head
    feature_fig =  px.bar(df_importance, x='Feature Name', y='Importance',
                          title = 'Feature Importance')


    return confusion_fig, feature_fig

# Run the app
if __name__ == '__main__':
    dash_app.run_server(debug=False)
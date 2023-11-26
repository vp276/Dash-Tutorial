from dash import Dash, dcc, html, callback,Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
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


# Text field
def drawText():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Text"),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])

# Data
df = px.data.iris()
df1 = pd.read_csv(total_path)

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
cmatrix, model = train_model(df1)
X_cols = df1.drop(columns = 'DEATH_EVENT')

# Build App
app = Dash(external_stylesheets=[dbc.themes.SLATE])
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}
sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE
)
filters = dbc.Row([
                html.Div(children= [
                html.H1('Heart Failure Prediction'),
                dcc.Markdown('A comprehensive tool for examining factors impacting heart failure'),

                html.Label('Blood Pressure'),
                dcc.Dropdown(
                    id = 'BP-Filter',
                    options = [{"label": i, "value": i} for i in df1['high_blood_pressure'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('Sex'),
                dcc.Dropdown(
                    id = 'Sex-Filter',
                    options = [{"label": i, "value": i} for i in df1['sex'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('Anaemia'),
                dcc.Dropdown(
                    id = 'Anaemia-Filter',
                    options = [{"label": i, "value": i} for i in df1['anaemia'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values")],
                    style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'})
             ])

app.layout = html.Div([
    filters,

    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText()
                ], width=3),
                dbc.Col([
                    drawText()
                ], width=3),
                dbc.Col([
                    drawText()
                ], width=3),
                dbc.Col([
                    drawText()
                ], width=3),
            ], align='center'), 
            html.Br(),
            dbc.Row(id = 'EDA-Row'),

            html.Br(),
            dbc.Row(id = 'ML-Row'),      
        ]), color = 'dark'
    )
])
#callback for top row
@callback(
    Output(component_id='EDA-Row', component_property='children'),
    [Input('BP-Filter', 'value'),
     Input('Sex-Filter', 'value'),
     Input('Anaemia-Filter', 'value')]
)
def update_output_div(bp, sex, anaemia):

    #Making copy of DF
    filtered_df = df1
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

    return dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Graph(figure=factor_fig.update_layout(
                                        template='plotly_dark',
                                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                    )
                                ) 
                            ])
                        ),  
                    ])
                ], width={"size": 3, "offset": 3}),
                dbc.Col([
                    html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Graph(figure=age_fig.update_layout(
                                        template='plotly_dark',
                                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                    )
                                )
                            ])
                        ),  
                    ])
                ])
            ], align='center')

#callback for second row
@callback(
    Output(component_id='ML-Row', component_property='children'),
    Input('Sex-Filter', 'value')
)
def update_model(value):

    #Making copy of df
    confusion = cmatrix
    model_copy = model
    x_copy = X_cols

    #Aggregating confusion dataframe and plotting
    confusion_fig = px.imshow(confusion, 
                              labels=dict(x="Predicted Value", 
                                y="True Value", color="Prediction"),
                                aspect="auto",
                                text_auto=True,
                                title = "Confusion Matrix - Predicted vs Actual Values, Train set")
    
    #Calculating feature imporance
    importances = model_copy.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_copy.estimators_], axis=0)
    df_importance = pd.DataFrame(list(zip(x_copy, importances, std)), 
                                 columns = ['Feature Name','Importance', 'Std'])
    #importances.head
    feature_fig =  px.bar(df_importance, x='Feature Name', y='Importance',
                          title = 'Feature Importance')

    return dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Graph(figure=feature_fig.update_layout(
                                        template='plotly_dark',
                                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                    )
                                ) 
                            ])
                        ),  
                    ])
                ], width={"size": 3, "offset": 3}),
                dbc.Col([
                    html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Graph(figure=confusion_fig.update_layout(
                                        template='plotly_dark',
                                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                    )
                                )
                            ])
                        ),  
                    ])
                ])
            ], align='center')

# Run app and display result inline in the notebook
app.run_server()
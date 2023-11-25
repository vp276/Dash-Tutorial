# Importing packages
import dash
from dash import Dash, dcc, html, dcc, Input, Output, ctx, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import os
import numpy as np
from sklearn.metrics import confusion_matrix


#loading Dataset
base_path = os.path.dirname(__file__)
file_name = 'heart_failure_clinical_records_dataset.csv'
total_path = base_path + '\\Data\\' + file_name
df = pd.read_csv(total_path)


#Calculating aggregated dataset


#Loading training data

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
                                      x="age", y="platelets", color = "sex", title = "Scatterplot")
    
    return factor_fig, age_fig



# Run the app
if __name__ == '__main__':
    dash_app.run_server(debug=True)
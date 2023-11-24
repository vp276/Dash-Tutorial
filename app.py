import os

# Importing packages
import dash
from dash import Dash, dcc, html, dcc, Input, Output, ctx, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import os
import numpy as np
from sklearn.metrics import confusion_matrix


#loading graffiti data
base_path = os.path.dirname(os.path.dirname(__file__))
mapkey = open(base_path + "\\mapbox_token.txt", "r").read()
file_name = 'predictions.csv'
total_path = base_path + '\\Data\\Image_Predictions\\' + file_name
df = pd.read_csv(total_path)
px.set_mapbox_access_token(mapkey)

#Calculating aggregated dataset
agg_predictions = df[['City', 'Predicted_val']].groupby(['City']).mean().reset_index()
print(agg_predictions.head(10))
latitude = []
longitude = []
for city in agg_predictions['City']:
    latitude.append(df[df['City']==city]['Latitude'].mean())
    longitude.append(df[df['City']==city]['Longitude'].mean())
agg_predictions['Latitude'] = latitude
agg_predictions['Longitude'] = longitude
    

#Loading training data
train_history_df = pd.read_csv(base_path + '\\Development\\Model\\performance\\history.csv')
confusion_df = pd.read_csv(base_path + '\\Development\\Model\\performance\\Validation_Performance.csv')

colors = {'background': '#A5D8DD',
        'text': '#7FDBFF'}

# Initialize the app
dash_app = Dash(__name__)
app = dash_app.server

# App layout
dash_app.layout = html.Div(style={'backgroundColor': colors['background']}, 
    children =[
        html.Div(children= [
             html.H1('Graffiti Frequency in US Cities'),
             dcc.Markdown('A comprehensive tool for examining graffiti rates in US cities using a convolutional neural network'),
            
            #Filters
             html.Div(children = [ 
                html.Label('City'),
                dcc.Dropdown(
                    id = 'City-Filter',
                    options = [{"label": i, "value": i} for i in df['City'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('Prediction'),
                dcc.Dropdown(
                    id = 'Prediction-Filter',
                    options = [{"label": i, "value": i} for i in df['Predicted_val'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values")
             ],style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            #First row of graphs
            html.Div([

                #First column
                html.Div(children = [
                    #Back button for if we've zoomed in
                    dbc.Button('🡠', id='back-button', outline=True, size="sm",
                        className='mt-2 ml-2 col-1', style={'display': 'none'}),

                    dcc.Graph(id = 'City-Barplot',
                            figure={})
                    ],style={'width': '49%', 'display': 'inline-block'}),

                #Second column
                html.Div([
                    dcc.Graph(id = 'Geographic-Plot',
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
                    dcc.Graph(id = 'history-plot',
                            figure={})
                ],style={'width': '49%', 'display': 'inline-block'})


            ], className = 'row'),
            html.Br(),


            html.Div([
                html.H3('Data Sources:'),
                html.Div([
                    html.Div(children = [
                        html.Div([
                            dcc.Markdown('Streetview Data: ')
                        ], style={'display': 'inline-block'}),
                        html.Div([
                            html.A("Google Streetview API", href='https://developers.google.com/maps/documentation/streetview/overview', target="_blank")
                        ], style={'display': 'inline-block'})
                    ], className="row"
                )
                ])
             ])
        ])
])


#callback for top row
@callback(
    [Output(component_id='Geographic-Plot', component_property='figure'),
    Output(component_id='City-Barplot', component_property='figure'),
    Output('back-button', 'style')],
    [Input('City-Filter', 'value'),
    Input('Prediction-Filter', 'value'),
    Input('back-button', 'n_clicks'),
    Input('City-Barplot', 'clickData')]#Backbutton for returning
)
def update_output_div(cities, predictions, n_clicks, clickData):

    #Checking which input was fired
    ctx = dash.callback_context

    #Making copy of DF
    filtered_df = df
    agged_df = agg_predictions

    city_list = []
    predicted_list = []
    #Filtering for cities
    if cities== "all_values":
        city_list = filtered_df['City'].drop_duplicates()
    else:
        city_list = [cities]

    #Filtering for prediction outcomes
    if predictions== "all_values":
        predicted_list = filtered_df['Predicted_val'].drop_duplicates()
    else:
        predicted_list = [predictions]
    
    #Applying filters to dataframe
    filtered_df = filtered_df[(filtered_df['City'].isin(city_list)) &
                              (filtered_df['Predicted_val'].isin(predicted_list))]

    #Checking which input was fired for graph drilldown
    trigger_id = ctx.triggered[0]['prop_id'].split(".")[0]
    back_return = {'display':'none'}

    #If barplot has been triggered
    if trigger_id == 'City-Barplot':
        back_return = {'display':'block'}
        selected_city = clickData['points'][0]['x']
        filtered_df = filtered_df[filtered_df['City']==selected_city]
        map_fig = px.scatter_mapbox(filtered_df, 
                            lat="Latitude", lon="Longitude", color="Predicted_val",
                            hover_data=["City"],
                            zoom = 2)
    else:
        map_fig = px.scatter_mapbox(agged_df, 
                            lat="Latitude", lon="Longitude", color="Predicted_val", size = 'Predicted_val',
                            hover_data=["City", "Predicted_val"],
                            zoom = 2)
    
    barplot_fig = px.bar(filtered_df[['City', 'Predicted_val']].groupby(['City']).sum().reset_index(),
                                      x="City", y="Predicted_val", title = "Graffiti Occurences by City")
    
    return map_fig, barplot_fig, back_return



#callback for second row
@callback(
    [Output(component_id='confusion-matrix', component_property='figure'),
    Output(component_id='history-plot', component_property='figure')],
    Input('City-Filter', 'value')
)
def update_model(value):

    #Making copy of df
    confusion = confusion_df[['y_true', 'y_pred']]
    history_df = train_history_df 

    #Aggregating confusion dataframe and plotting
    agg_confusion = confusion_matrix(confusion['y_true'], confusion['y_pred'])
    confusion_fig = px.imshow(agg_confusion, 
                              labels=dict(x="Predicted Value", 
                                y="True Value", color="Prediction"),
                                x=['Graffiti', 'No Graffiti'],
                                y=['Graffiti', 'No Graffiti'], 
                                aspect="auto",
                                text_auto=True,
                                title = "Confusion Matrix - Predicted vs Actual Values")

    #Plotting history data
    history_df['Epoch'] = history_df.index
    history_df = pd.melt(history_df, id_vars='Epoch', value_vars=['accuracy', 'val_accuracy'])
    history_fig = px.line(history_df, x="Epoch", y="value", color='variable', 
                          title="Test Versus Train Accuracy across Epochs")
    
    return confusion_fig, history_fig



# Run the app
if __name__ == '__main__':
    dash_app.run_server(debug=False)
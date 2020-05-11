import dash
import re
import time
import base64
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import random
import dash_table
from urllib.parse import quote as urlquote
import json
from control import WELL_COLOR_new
import dash_table
import os
import flask
from flask_caching import Cache
from flask import Flask, send_from_directory
from clustering_correlation import compute_serial_matrix
import numpy as np
from process_data import  process_data,clean_folder
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
import copy
from dash.dependencies import Input, Output, State, ClientsideFunction
from flask_caching import Cache
import pathlib

APP_PATH = str(pathlib.Path(__file__).parent.resolve())   #include download
UPLOAD_DIRECTORY = APP_PATH+"/data/app_uploaded_files/"
PROCESSED_DIRECTORY=APP_PATH + "/data/processed/"
DEFAULT_DATA=APP_PATH + "/data/default_data/"
CACHE_DIRECTORY=APP_PATH+"/data/cache/"
FILE_LIST=""
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    print("created")
if not os.path.exists(PROCESSED_DIRECTORY):
    os.makedirs(PROCESSED_DIRECTORY)
    print("created")
if not os.path.exists(CACHE_DIRECTORY):
    os.makedirs(CACHE_DIRECTORY)
    print("created")

app = dash.Dash(meta_tags=[{"name": "viewport", "content": "width=device-width"}])
server = app.server
cache_config = {
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR":CACHE_DIRECTORY,
}

# Empty cache directory before running the app
#if
clean_folder(CACHE_DIRECTORY)
# folder = os.path.join(APP_PATH, "data/cache/")
# for the_file in os.listdir(folder):
#     file_path = os.path.join(folder, the_file)
#     try:
#         if os.path.isfile(file_path):
#             os.unlink(file_path)
#     except Exception as e:
#         print(e)

@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    if os.path.exists(UPLOAD_DIRECTORY+path):
        return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)
    elif os.path.exists(PROCESSED_DIRECTORY+path):
        return send_from_directory(PROCESSED_DIRECTORY, path, as_attachment=True)
    else:
        return send_from_directory(DEFAULT_DATA, path, as_attachment=True)

app.config.suppress_callback_exceptions = True
mapbox_access_token = "pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w"
layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=50, t=50),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    #title="",
    titlefont= {"size": 32},
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=8,
    ),
    xaxis=dict(
            autorange=True,
            showgrid=False,
            ticks='',
            showticklabels=False
        ),
    yaxis=dict(
            autorange=True,
            showgrid=False,
            ticks='',
            showticklabels=False
        )
)


#dataset=pd.read_pickle('new_test.pkl')
# with open(UPLOAD_DIRECTORY+'long-island-railroad_20090825_0512.gml.20.apx', 'r') as file:
#     test = file.read()
# print(test)


dataset_all=pd.read_pickle('long-island-railroad_argument_frequency.pkl')
df = pd.read_pickle('long-island-railroad_tsne_epts=2.4_minp=10_cluster=4.pkl')
report_cluster_km=pd.read_pickle("km_long-island-railroad_cluster_report.pkl")
report_cluster_db=pd.read_pickle("db_long-island-railroad_cluster_report.pkl")

report_groups=pd.read_pickle("long-island-railroad_groups_report.pkl")

correlation_matrix=pd.read_pickle("answer2_correlation_matrix.pkl")

cache = Cache()
cache.init_app(app.server, config=cache_config)

TIMEOUT = 120

def uploaded_files( directory ):
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


def get_file_name(dir):
    files = uploaded_files(dir)
    if len(files) == 0:
        return [html.Li("")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]


argument_analysis=html.Div([

    dcc.Link(html.Button('back'), href='/'),
    html.Div(
        [
            html.Div(
                [
                    # html.H4(
                    #     "select range in histogram:",
                    #     className="control_label",
                    # ),
                    # dcc.RangeSlider(
                    #     id='my-range-slider',
                    #     min=0,
                    #     max=len(dataset_all),
                    #     step=1,
                    #     value=[5, int(0.5*len(dataset_all))]
                    # ),
                    html.P("Presented data:", style={"font-weight": "bold"},className="control_label"),
                    dcc.RadioItems(
                        id="data_present_selector",
                        options=[
                            {"label": "All ", "value": "all"},
                            {"label": "Interesting", "value": "interesting"},
                        ],
                        value="all",
                        labelStyle={"display": "inline-block"},
                        className="dcc_control",
                    ),

                    dcc.Checklist(
                        id="sort_selector",
                        options=[{"label": "descending order", "value": "decreased"}],
                        className="dcc_control",
                        value=["decreased"],
                    ),


                    html.P("Cluster Algorithm:", style={"font-weight": "bold"}, className="control_label"),
                    dcc.RadioItems(
                        id="clustering-method",
                        options=[
                            {"label": "DBscan ", "value": "db"},
                            {"label": "Kmeans", "value": "km"},
                        ],
                        labelStyle={"display": "inline-block"},
                        value="db",
                        className="dcc_control",
                    ),


                    html.Div(
                        [html.H5(id="selected_cluster")],
                        id="selected argument",
                        className="dcc_control",
                        # className="mini_container",
                    ),
                    # html.Div(
                    #     id="card-1",
                    #     children=[
                    #
                    #         daq.LEDDisplay(
                    #             id="stable",
                    #             value="04",
                    #             color="#92e0d3",
                    #             backgroundColor="#FFFF",
                    #             size=50,
                    #         ),
                    #        daq.LEDDisplay(
                    #             id="prefer",
                    #             value="17",
                    #             color="#92e0d3",
                    #             backgroundColor="#FFFF",
                    #             size=50,
                    #         ),
                    #     ],
                    #     className="row container-display",
                    # ),



                    html.Div(
                        [
                            html.Div(
                                [html.H6(id="stable"), html.P("Stable")],
                                id="stable_block",
                                className="mini_container",
                            ),
                            html.Div(
                                [html.H6(id="prefer"), html.P("Prefer")],
                                id="prefer_block",
                                className="mini_container",
                            ),
                            html.Div(
                                [html.H6(id="complete"), html.P("Complete")],
                                id="complete_block",
                                className="mini_container",
                            ),

                        ],
                        id="info-container",
                        className="row container-display"
                    )

                ],
                className="pretty_container four columns",
                id="cross-filter-options",
            ),
            html.Div([dcc.Graph(id="bar_chart"),
                      dcc.RangeSlider(
                          id='my-range-slider',
                          min=0,
                          max=len(dataset_all),
                          step=1,
                          value=[5, int(0.5 * len(dataset_all))]
                      ),
                      ],
                     className="pretty_container seven columns",
                     style={'width': '64%'}),

        ],

        className="row flex-display",
    ),
    html.Div([
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="basic-interactions"),
        )
    ],
    className="row",
    ),
    html.Div([
        html.Div([
        dcc.Graph(
            id='basic-interactions'),
        dcc.RadioItems(
                id="argument-dimensional-reduction",
                options=[
                    {"label": "Tsne ", "value": "tsne"},
                    {"label": "SVD", "value": "svd"},
                ],
                value="tsne",
                labelStyle={"display": "inline-block"},
                className="dcc_control",
            ),
        ],
        className="pretty_container seven columns"
        ),

        html.Div(
            [dcc.Graph(id="pie_graph")],
            className="pretty_container five columns",
        ),



    ],
    className = "row flex-display",
    ),
    # html.Div(
    #     [
    #         html.Br(),

    #     className="row flex-display",
    # ),
],

)

correlation_page= html.Div([
                dcc.Link(html.Button('back'), href='/'),
                html.Div([

                dcc.Graph(
                    id="correlation_hm"
                   ),
                    html.Button('Cluster Correlation Matrix',style={'font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}, id='btn-nclicks-1', n_clicks=0)
                    ],
            className="pretty_container")
    ])

main_page =     html.Div([
    dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
    html.Div(id="output-clientside"),
    html.Div(
        [
                html.Div(
                    [
                        html.A(
                            html.Button("Get Default Data", id="get_default_data",
                                        style={"color":"#FFFF","backgroundColor":"#2F8FD2"}),
                            href="https://github.com/Lexise/ASP-Analysis/tree/master/data/default_data",
                        )
                    ],
                    #className="one-fifth column",
                    style={'width': '15%', 'textAlign': 'left'},
                    id="download_default",
                ),
                html.Div(
                    [
                        html.H2(
                            "NEVA",
                            style={"margin-bottom": "10px",'fontWeight': 'bold'},
                        ),
                        html.H5(
                            "Extension Visualization for Argumentation Frameworks", style={"margin-top": "0px"}
                        ),



                    ],

                    #className="one-half column",
                    id="title",
                    style={'width': '100%', 'textAlign': 'center'}
                ),
                html.Div(
                    [

                        dcc.Upload([html.Button("View Processed Data" ,id="view_button",style={"color":"#FFFF","backgroundColor":"#2F8FD2"})],id="view-processed-button"),

                        dcc.ConfirmDialog(
                            id='confirm',
                            message='finish uploading?',
                        ),
                        html.Div(id='hidden-div', style={'display':'none'})
                        #html.A('Refresh', href='/')
                    ],
                    #className="one-fifth column",
                    style={'width': '15%', 'textAlign': 'right'},
                    id="button",
                ),



        ],
        id="header",
        className="row flex-display",
        style={"margin-bottom": "25px"},
        ),





    #     # dcc.Graph(
    #     #     id='compare_groups',
    #     #     figure={
    #     #         'data': [
    #     #             {
    #     #                 'x': prefer.position_x,
    #     #                 'y': prefer.position_y,
    #     #                 'text': ["clusters: {}".format(x) for x in prefer['cluster_label']],
    #     #                 'name': 'prefer-',
    #     #                 'mode': 'markers',
    #     #                 'marker': {'size': 12}
    #     #             },
    #     #             {
    #     #                 'x': stable.position_x,
    #     #                 'y': stable.position_y,
    #     #                 'text': ["clusters: {}".format(x) for x in stable['cluster_label']],
    #     #                 'name': 'stable',
    #     #                 'mode': 'markers',
    #     #                 'marker': {'size': 12}
    #     #             }
    #     #         ],
    #     #         'layout': {
    #     #             'clickmode': 'event'
    #     #         }
    #     #     }
    #     # ),
    #     #
    #
    #
    html.Div([
        dcc.Tabs([
            dcc.Tab(label='Scatter with Cluster', style={ 'fontWeight': 'bold'}, children=[
                dcc.Graph(
                    id="scatter_cluster",

                ),
                html.Div([
                    html.Div(children=[
                        html.Span("Dimensional Reduction:", style={"font-weight": "bold"}),
                        dcc.RadioItems(
                            id="dimensional-reduction1",
                            options=[
                                {"label": "Tsne ", "value": "tsne"},
                                {"label": "SVD", "value": "svd"},
                            ],
                            labelStyle={"display": "inline-block"},
                            value="tsne",
                        ),
                            ],
                        style = {'width': '30%'},

                    ),
                # html.Div([dcc.RadioItems(
                #     id="dimensional-reduction2",
                #     options=[
                #         {"label": "Tsne ", "value": "tsne"},
                #         {"label": "SVD", "value": "svd"},
                #     ],
                #     value="tsne",
                #     labelStyle={"display": "inline-block"},
                #     className="dcc_control",
                #     )],
                #     className="mini_container",
                # ),
                    html.Div([
                    html.Span("Cluster Algorithm:", style={"font-weight": "bold"}),
                    dcc.RadioItems(
                        id="clustering-method",
                        options=[
                                    {"label": "DBscan ", "value": "db"},
                                    {"label": "Kmeans", "value": "km"},
                                ],
                        labelStyle={"display": "inline-block"},
                        value="db",
                    )],
                    style = {'width': '30%'},
                    ),

                ],
                className = "row flex-display"
                )
            ]),


            dcc.Tab(label='Scatter with Groups', style={ 'fontWeight': 'bold'}, children=[
                dcc.Graph(
                    id="scatter_groups",

                ),
                html.Div(children=[
                    html.Span("Dimensional Reduction:", style={"font-weight": "bold"}),
                    dcc.RadioItems(
                        id="dimensional-reduction2",
                        options=[
                            {"label": "Tsne ", "value": "tsne"},
                            {"label": "SVD", "value": "svd"},
                        ],
                        labelStyle={"display": "inline-block"},
                        value="tsne",
                    ),
                ],
                    style={'width': '30%'},

                ),
            ]),
            dcc.Tab(label='Feature Report',style={ 'fontWeight': 'bold'}, children=[
                html.Div([
                    html.Div(id="table1",
                    className="pretty_container six columns"
                    ),
                    html.Div(
                            id='table2',
                            className="pretty_container seven columns"
                    )
                #html.Div(id="orders_table", className="row table-orders"),
                ],
                 className="row flex-display"
                ),
                html.Div([
                    html.Span("Cluster Algorithm:", style={"font-weight": "bold"}),
                    dcc.RadioItems(
                        id="clustering-method-table",
                        options=[
                            {"label": "DBscan ", "value": "db"},
                            {"label": "Kmeans", "value": "km"},
                        ],
                        labelStyle={"display": "inline-block"},
                        value="db",
                    )],
                  ),
                ]),
            ]),


        ],
        className = "pretty_container"),



        dcc.Link(html.Button('Argument Analysis',style={'width': '49%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-argument'),
        dcc.Link(html.Button('Correlation Matrix',style={'marginLeft': '2%', 'width': '49%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-correlation'),
        html.Hr(),
        html.Div([

            dcc.Upload([html.Button(children='Upload File',id="upload_button")], id="upload-data",style={'width': '13%','marginRight': '2.5%'} ),
            dcc.Input(id='eps', type='text', value='Eps',style={'width': '10%','marginRight': '0.5%','marginLeft': '4%'}),
            dbc.Tooltip("DBscan  parameter, specifies the distance between two points to be considered within one cluster.suggested a decimal in range[1,3]", target="eps"),
            dcc.Input(id='minpts', type='text', value='MinPts',style={'width': '10%','marginRight': '0.5%'}),
            dbc.Tooltip("DBscan parameter, the minimum number of points to form a cluster. suggested an integer in range[3,15]", target="minpts"),
            dcc.Input(id='cluster_num', type='text', value='Cluster Num',style={'width': '11%','marginRight': '0.5%'}),
            dbc.Tooltip("Kmeans parameter, number of clusters, suggested an integer in range[2,15]", target="cluster_num"),
            html.Button(id='submit-button-state', n_clicks=0, children='Submit',style={'width': '9%','font-size':'13px',"color":"#FFFF","backgroundColor":"#2F8FD2"}),
        ],
        id="upload_block",
        className="row flex-display"),

        html.Br(),
        html.Div([
            html.Div([
                html.P("Upload"),
                html.Ul(id="file-list", children=get_file_name(UPLOAD_DIRECTORY))
            ],
                className="pretty_container seven columns"),

            html.Div([
                html.P("Processed"),
                html.Ul(id="processed-list", children=get_file_name(PROCESSED_DIRECTORY))
            ],
                className="pretty_container seven columns")
        ],
            className="row flex-display")

    ])


app.layout = html.Div([


    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    # hidden signal value
    #html.Div(id='signal', style={'display': 'none'}),
    dcc.Loading(
        id="loading-2",
        type="default",
        children=html.Div(id="signal")
    ),

],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

def global_store(eps, minpts, n_cluster):
    # simulate expensive query
    files = uploaded_files(UPLOAD_DIRECTORY)

    if len(files)>1:
        for filename in files:
            try:
                if 'apx' in filename:
                    # Assume that the user uploaded a CSV file
                    question = filename
                elif 'EE-PR' in filename:
                    # Assume that the user uploaded an excel file
                    answer = filename
            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])
        start_time = time.time()
        print("start process")
        processed_data, bar_data, correlation_matrix,cluster_feature_db,cluster_feature_km,group_feature = process_data(UPLOAD_DIRECTORY+question, UPLOAD_DIRECTORY+answer,eps, minpts, n_cluster)
        group_feature.to_pickle(PROCESSED_DIRECTORY + "group_feature.pkl")
        cluster_feature_db.to_pickle(PROCESSED_DIRECTORY + "db_cluster_feature.pkl")
        cluster_feature_km.to_pickle(PROCESSED_DIRECTORY + "km_cluster_feature.pkl")

        processed_data.to_pickle(PROCESSED_DIRECTORY+"processed_data.pkl")
        bar_data.to_pickle(PROCESSED_DIRECTORY + "bar_data.pkl")
        correlation_matrix.to_pickle(PROCESSED_DIRECTORY + "correlation_matrix.pkl")
        print("get processed data", time.time() - start_time)
        return processed_data.to_dict()
    else:
        return ""


# @app.callback(
#     Output("popover", "is_open"),
#     [Input("eps", "clickData")],
#     [State("popover", "is_open")],
# )
# def toggle_popover(n, is_open):
#     if n:
#         return not is_open
#     return is_open

@app.callback(Output('signal', 'children'), [Input('submit-button-state', 'n_clicks')],
              [State('eps', 'value'), State('minpts', 'value'), State('cluster_num', 'value')])
def compute_value( n_clicks, eps, minpts, n_cluster):
    # compute value and send a signal when done
    #print("file-list signal",content)
    if len(os.listdir(UPLOAD_DIRECTORY)) == 0:
        print("return no content")
        return ""   #haven't upload data
    else:
            if int(n_clicks)>0:
                # if len(os.listdir(PROCESSED_DIRECTORY)) != 0:
                #     clean_folder(PROCESSED_DIRECTORY)
                processed_data = global_store(eps, minpts, n_cluster)
                return "finish"
            else:
                return ""
       #already  process, no need to pass data again
    # global_store(value)
    # return value






def save_file(name, content, dir):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    #content_type, content_string = content.split(',')
    with open(os.path.join(dir, name), "wb") as fp:
        fp.write(base64.decodebytes(data))



@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents"), Input("upload_button","n_clicks"),Input('signal', 'children')],
)
def update_output(uploaded_filenames, uploaded_file_contents, n_click,children):
    """Save uploaded files and regenerate the file list."""
    if n_click is not None:
        if len(os.listdir(UPLOAD_DIRECTORY)) != 0 and uploaded_filenames is not None and n_click % 2 == 1:
            clean_folder(UPLOAD_DIRECTORY)
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip([uploaded_filenames], [uploaded_file_contents]):
            save_file(name, data, UPLOAD_DIRECTORY)

    files = uploaded_files(UPLOAD_DIRECTORY)
    if len(files) == 0:
        return ""
    else:
        FILE_LIST=[html.Li(file_download_link(filename)) for filename in files]
        return FILE_LIST

# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("bar_chart", "figure")],
)


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-argument':
        return argument_analysis
    elif pathname == '/page-correlation':
        return correlation_page
    else:
        return main_page


@app.callback([Output('scatter_cluster', 'figure'), Output('scatter_groups', 'figure'), Output('table1', 'children'),Output('table2', 'children')],
              [ Input('signal', 'children'),Input("dimensional-reduction1", "value"),Input("dimensional-reduction2", "value"),
                Input("clustering-method", "value"),Input("clustering-method-table", "value"), Input('confirm', 'submit_n_clicks')])
#@cache.memoize(TIMEOUT)
def generate_tabs( content, reduction1, reduction2, method, table_method, n_click):#processed_data, table1_data,table2_data ):
    if  os.listdir(PROCESSED_DIRECTORY) or n_click:#load and processed
        processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "processed_data.pkl")
        group_table = pd.read_pickle(PROCESSED_DIRECTORY + "group_feature.pkl")
        cluster_table = pd.read_pickle(PROCESSED_DIRECTORY + table_method+"_cluster_feature.pkl")
    else:
            processed_data=df
            group_table = report_groups
            if table_method=="km":
                cluster_table = report_cluster_km
            else:
                cluster_table = report_cluster_db
            # processed_data=pd.DataFrame.from_dict(content)
            # group_table = pd.read_pickle(PROCESSED_DIRECTORY + "group_feature.pkl")
            # cluster_table = pd.read_pickle(PROCESSED_DIRECTORY + table_method+ "_cluster_feature.pkl")
    if reduction1=="svd":
        x_axe="svd_position_x"
        y_axe="svd_position_y"
    else:
        x_axe = "tsne_position_x"
        y_axe = "tsne_position_y"

    cluster_label = method + "_cluster_label"
    cluster_set=processed_data[cluster_label].unique()

    if len(cluster_set)>25:
        r = lambda: random.randint(0, 255)
        figure1={
            'data': [
                {
                    'x': processed_data[processed_data[cluster_label] == cls][x_axe],
                    'y': processed_data[processed_data[cluster_label] == cls][y_axe],
                    'text': ["groups: {}".format(x) for x in processed_data[processed_data[cluster_label] == cls]['groups']],
                    'name': cls,
                    'mode': 'markers',
                    'marker': {'size': 12,
                               "color": '#%02X%02X%02X' % (r(),r(),r()),
                               'line': {'width': 0.5, 'color': 'white'}

                               }
                } for cls in cluster_set

            ],
            'layout': layout
        }
    else:
        figure1 = {
            'data': [
                {
                    'x': processed_data[processed_data[cluster_label] == cls][x_axe],
                    'y': processed_data[processed_data[cluster_label] == cls][y_axe],
                    'text': ["groups: {}".format(x) for x in processed_data[processed_data[cluster_label] == cls]['groups']],
                    'name': cls,
                    'mode': 'markers',
                    'marker': {'size': 12,
                               "color": WELL_COLOR_new[cls],
                               'line': {'width': 0.5, 'color': 'white'}

                               }
                } for cls in cluster_set
            ],
            'layout': layout
        }
    if reduction2=="svd":
        x_axe="svd_position_x"
        y_axe="svd_position_y"
    else:
        x_axe = "tsne_position_x"
        y_axe = "tsne_position_y"
    figure2 = {
        'data': [
            {
                'x': processed_data[processed_data.groups == cls][x_axe],
                'y': processed_data[processed_data.groups == cls][y_axe],
                'text': ["Clusters: {}".format(x) for x in processed_data[processed_data.groups == cls][cluster_label]],
                'name': cls,
                'mode': 'markers',
                'marker': {'size': 12,
                           'line': {'width': 0.5, 'color': 'white'}}
            } for cls in processed_data['groups'].unique()
        ],
        'layout': layout
    }


    #table
    if len(group_table)==0:
        table1=html.H5("No group feature")
    else:
        table1 = dash_table.DataTable(
            data=group_table.to_dict('records'),
            columns=[{"name": i, "id": i} for i in group_table.columns],
            style_table = {
                              'maxHeight': '300px',
                              'overflowY': 'scroll'
                          },
            style_header = {
                               'fontWeight': 'bold'
                           },
            style_cell = {
                             'font_size': '20px',
                             'text_align': 'center'
                         },
        )

    if len(cluster_table) == 0:
        table2=html.H1("No cluster Feature")
    else:
        table2=dash_table.DataTable(
            data=cluster_table.to_dict('records'),
            columns=[{"name": i, "id": i} for i in cluster_table.columns],
            style_table = {
                              'maxHeight': '300px',
                              'overflowY': 'scroll'
                          },
            style_header = {
                               'fontWeight': 'bold'
                           },
            style_cell = {
                             'font_size': '20px',
                             'text_align': 'center'
                         },
        )
    return figure1,figure2,table1,table2




@app.callback(
    [Output("bar_chart", "figure"),Output("my-range-slider","figure")],
    [Input("data_present_selector", "value"),Input("my-range-slider", "value"),Input("sort_selector", "value")],
    )
@cache.memoize(TIMEOUT)
def make_bar_figure(present_data, valuelist,sort_state):
    if len(os.listdir(PROCESSED_DIRECTORY))!=0:
        dataset_bar=pd.read_pickle(PROCESSED_DIRECTORY+"bar_data.pkl")
    else:
        dataset_bar=dataset_all
    figure=dict()
    slider=dict(
        min = 0,
        max = len(dataset_bar),
        step = 1,
        value = [1, int(0.5*len(dataset_bar))]
    )
    if present_data == "all":
       if sort_state == ["decreased"]:
           temp=dataset_bar.sort_values(by=['frequency'],ascending=False, inplace=False)
           figure= set_bar_figure(temp, valuelist)
       else:
           figure= set_bar_figure(dataset_bar, valuelist)
    else:
       dataset=dataset_bar[~dataset_bar.rate.isin([0,100])]
       min = int(valuelist[0] * len(dataset) / len(dataset_bar))
       max= int(valuelist[1] * len(dataset) / len(dataset_bar))
       if sort_state == ["decreased"]:
           temp=dataset.sort_values(by=['frequency'],ascending=False, inplace=False)
           figure = set_bar_figure(temp, [min,max])
       else:
           figure = set_bar_figure(dataset, [min,max])
    return figure,slider


def set_bar_figure(argument_data, valuelist):
    layout_count = copy.deepcopy(layout)
    select_idx=range(valuelist[0],valuelist[1])
    selected=argument_data.iloc[select_idx]
    selected["order"]=range(len(selected))

    data = [dict(
            type="bar",
            x=list(selected["argument"]),
            y=list(selected["rate"]),
            hovertext={"fontsize":20},
            #hovertext=["attribute:{arg},rate:{percent}".format(arg=row.attribute,percent=row.rate) for index,row in selected.iterrows()],
            name="All Wells"
        )]


    layout_count["title"] = "Rate/Argument"

    layout_count["dragmode"] = "select"
    layout_count["showlegend"] = False
    layout_count["autosize"] = True,
    layout_count["titlefont"] = {"size": 28}
    layout_count["marker"] = {"fontsize": 20}
    if 'xaxis' in layout_count:
        del layout_count['xaxis']
        del layout_count['yaxis']
    figure = dict(data=data, layout=layout_count)
    return figure

@app.callback(
    [Output("confirm", "displayed")],
    [Input("view-processed-button", "filename"), Input("view-processed-button", "contents"),Input("view_button","n_clicks")]
    )
def run_processed_data(uploaded_filenames, uploaded_file_contents,n_click ):
    if n_click is None:
        return [False]
    if len(os.listdir(PROCESSED_DIRECTORY))!=0 and uploaded_filenames is not None and n_click%6==1:
        clean_folder(PROCESSED_DIRECTORY)
    print("click number: ",n_click)
    print("upload_name:",uploaded_filenames)
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip([uploaded_filenames], [uploaded_file_contents]):
            save_file(name, data, PROCESSED_DIRECTORY)
        if len(os.listdir(PROCESSED_DIRECTORY)) >=6:
            return [True]
    return [False]


@app.callback(Output('processed-list', 'children'),
              [ Input('signal', 'children'),Input('confirm', 'submit_n_clicks')])
def update_output(children, submit_n_clicks):
    #if submit_n_clicks or children:
        return  get_file_name(PROCESSED_DIRECTORY)



@app.callback(
    [Output("selected_cluster","children"), Output('stable', 'children'), Output('prefer', 'children'),
     Output('complete', 'children'),Output("pie_graph", "figure")],
    [Input('bar_chart', 'clickData') ,Input("clustering-method","value")])

def update_cluster_rate(clickData, cluster_method):
    if len(os.listdir(PROCESSED_DIRECTORY))!=0:
        process_data=pd.read_pickle(PROCESSED_DIRECTORY+"processed_data.pkl")
    else:
        process_data=df
    layout_pie = copy.deepcopy(layout)
    layout_pie["title"] = "Cluster Summary"
    if clickData is None:
        return "Selected Argument: None","","","",dict(data=None, layout=layout_pie),
    temp=clickData["points"][0]
    arguments=int(re.search(r'\d+', temp["x"]).group())
    selected=[]
    result0= "Selected Argument:{}  \n".format(arguments)
    for index, row in process_data.iterrows():
        if arguments in row.arg:
            selected.append(index)
    if len(selected) == 0:
        return "No data has this argument","","","",dict(data=None, layout=layout_pie),
    data = process_data.loc[selected]
    result=""
    cluster_label=cluster_method+"_cluster_label"
    clusters=set(data[cluster_label])
    for cluster in clusters:
        num=len(data[data[cluster_label]==cluster])
        result=result+"{} % belong to cluster {} . ".format(num/len(data)*100,cluster)
    stable_value=len(data[data.groups == "stable"])/ len(data) * 100
    prefer_value=len(data[data.groups == "prefer-"])/ len(data) * 100
    stable= "{:.2f}".format(stable_value)+"%"
    prefer="{:.2f}".format(prefer_value)+"%"
    complete= "{:.2f}".format(100-stable_value-prefer_value)+"%"

    result = dict({
        "cluster": [],
        "num": []
    })
    for cluster in clusters:
        result["cluster"].append(str(cluster) + " cluster")
        num = len(data[data[cluster_label] == cluster])
        result["num"].append(num)
        # result["rate"].append(num/len(data))
    if len(clusters) > 25:
        r = lambda: random.randint(0, 255)
        data_bar = [
            dict(
                type="pie",
                labels=result["cluster"],
                values=result["num"],
                name="Production Breakdown",
                text=[
                    "Data Num in cluster {}".format(a) for a in result["cluster"]
                ],
                hoverinfo="text+value+percent",
                textinfo='none',
                hole=0.5,
                marker=dict(colors=['#%02X%02X%02X' % (r(), r(), r()) for i in clusters])

            )
        ]
    else:
        data_bar = [
            dict(
                type="pie",
                labels=result["cluster"],
                values=result["num"],
                name="Production Breakdown",
                text=[
                    "Data Num in cluster {}".format(a) for a in result["cluster"]
                ],
                hoverinfo="text+value+percent",
                textinfo="label+percent+name",
                hole=0.5,
                marker=dict(colors=[WELL_COLOR_new[i] for i in clusters]),

            )
        ]

    layout_pie["title"] = "Cluster Summary"
    layout_pie["legend"] = dict(
        font=dict(color="#CCCCCC", size="10"), orientation="h", bgcolor="rgba(0,0,0,0)"
    )

    figure = dict(data=data_bar, layout=layout_pie)

    result2=""
    for group in set(data.groups):
        num = len(data[data.groups == group])
        result2 = result2 + "{} % belong to group {}. ".format(num / len(data) * 100, group)

    return result0,stable,prefer,complete,figure


@app.callback(
    Output('basic-interactions', 'figure'),
    [Input('bar_chart', 'clickData'), Input("argument-dimensional-reduction","value"), Input("clustering-method","value")])

def update_graph(clickData, dimensional_reduction, cluster_method="db"):
    if len(os.listdir(PROCESSED_DIRECTORY))!=0:
        process_data=pd.read_pickle(PROCESSED_DIRECTORY+"processed_data.pkl")
    else:
        process_data=df
    layout_scatter = copy.deepcopy(layout)
    layout_scatter["title"]="Distribution of Selected Argument"
    layout_scatter["clickmode"]= 'event+select'
    if clickData is None:
        return {
            'data':[],
            'layout': layout_scatter
        }
    temp=clickData["points"][0]
    cluster_label=cluster_method +"_cluster_label"
    arguments=int(re.search(r'\d+', temp["x"]).group())
    selected=[]
    for index, row in process_data.iterrows():
        if arguments in row.arg:
            selected.append(index)
    data=process_data.loc[selected]
    unselected_data=process_data[~process_data.index.isin(selected)]
    if dimensional_reduction=="svd":
        x_axe="svd_position_x"
        y_axe="svd_position_y"
    else:
        x_axe = "tsne_position_x"
        y_axe = "tsne_position_y"
    return {
        'data': [
            dict(
            x=data[x_axe],
            y=data[y_axe],
            text=["clusters: {}".format(x) for x in data[cluster_label]],
            name="selected",
            mode='markers',
            marker={
                'size': 12,
                'opacity': 1.0,
                'line': {'width': 0.5, 'color': 'white'}
            }),
            dict(
                x = unselected_data[x_axe],
                y = unselected_data[y_axe],
                text=["clusters: {}".format(x) for x in unselected_data[cluster_label]],
                name = "unselected",
                mode = 'markers',
                marker= { 'size': 12,
                "color":"LightSkyBlue",
                'opacity': 0.3,
                'line': {'width': 0.5, 'color': 'white'}
                # make text transparent when not selected
                #'textfont': {'color': 'rgba(0, 0, 0, 0)'}
            }
            )
        ],
        'layout': layout_scatter
    }



@app.callback(Output('correlation_hm', 'figure'),
              [Input('btn-nclicks-1', 'n_clicks')])
def displayClick(btn1):
    if len(os.listdir(PROCESSED_DIRECTORY))!=0:
        data_correlation=pd.read_pickle(PROCESSED_DIRECTORY+"correlation_matrix.pkl")
    else:
        data_correlation=correlation_matrix
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    layout_matrix =  {
                            'height': 750,
                            "title": {
                                "text":"Correlation Coefficient Matrix",
                                "font":dict(family="Open Sans, sans-serif", size=30, color="#515151"),
                        },
                        "font":dict(family="Open Sans, sans-serif", size=13),
                        "automargin":True,
                        }

    if btn1%2:
        distances = np.sqrt((1 - data_correlation) / 2)
        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(distances.values, method='single')
        new_order = [data_correlation.index[i] for i in res_order]

        ordered_correlation_matrix = data_correlation.reindex(index=new_order, columns=new_order)

        data = [{
                    "type" : "heatmap",
                    "z" : ordered_correlation_matrix.to_numpy(),
                    "x" : [str(x)+"arg" for x in new_order],
                    "y" : [str(x)+"arg" for x in new_order],
                    "colorscale" : [[0, "#2F8FD2"], [1, "#ecae50"]], #[[ 0, WELL_COLOR_new[0]], [ 1, WELL_COLOR_new[3]]],
                    "reversescale" : True,
                    "showscale" : True,
                    "xgap" : 2,
                    "ygap" : 2,
                    "colorbar": {
                        "len":0.6,
                        "ticks":"",
                        "title":"Correlation",
                        "titlefont":{
                            "family":"Gravitas One",
                            "color":"#515151"
                        },
                        "thickness":30,
                        "tickcolor":"#515151",
                        "tickfont":{
                            "family":"Open Sans, sans serif", "color":"#515151"},
                        "tickvals":[-1, 1],
                    },
                }]
        figure = dict(data=data, layout=layout_matrix)
        return figure
    else:
        return {
                        "data":[
                            {
                            "type" : "heatmap",
                            "z" : data_correlation.to_numpy(),
                            "x" : [str(x)+"arg" for x in data_correlation.columns],
                            "y" : [str(x)+"arg" for x in data_correlation.index],
                            "colorscale" : [[0, "#2F8FD2"], [1, "#ecae50"]],
                            #"reversescale" : True,
                            "showscale" : True,
                            "xgap" : 2,
                            "ygap" : 2,
                            "colorbar": {
                                "len":0.6,
                                "ticks":"",
                                "title":"Correlation",
                                "titlefont":{
                                    "family":"Gravitas One",
                                    "color":"#515151"
                                },
                                "thickness":30,
                                "tickcolor":"#515151",
                                "tickfont":{
                                    "family":"Open Sans, sans serif", "color":"#515151"},
                                "tickvals":[-1, 1],
                            },
                            }
                        ],
                        "layout":layout_matrix
                    }


            #
            # {
            #     'data':[],
            #     'layout': layout_matrix
            # }




if __name__ == '__main__':
    app.run_server(debug=True)
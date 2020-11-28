import dash
import re
import time
import base64
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import random
import io
import plotly.express as px
from zipfile import ZipFile
from urllib.parse import quote as urlquote
import json
from control import WELL_COLOR_new
import dash_table
import os
import plotly.graph_objects as go
from flask_caching import Cache
from flask import Flask, send_from_directory
from clustering_correlation import compute_serial_matrix,innovative_correlation_clustering,my_optimal_leaf_ordering,abs_optimal_leaf_ordering
import numpy as np
from process_data import  process_data,clean_folder,f_comma
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


ZIP_DIRECTORY=APP_PATH + "/data/processed_zip/"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    print("created")
if not os.path.exists(PROCESSED_DIRECTORY):
    os.makedirs(PROCESSED_DIRECTORY)
    print("created")
if not os.path.exists(CACHE_DIRECTORY):
    os.makedirs(CACHE_DIRECTORY)
    print("created")
if not os.path.exists(ZIP_DIRECTORY):
    os.makedirs(ZIP_DIRECTORY)
    print("created")
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
app.scripts.config.serve_locally=True
app.css.config.serve_locally=True
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
    elif os.path.exists(ZIP_DIRECTORY + path):
        return send_from_directory(ZIP_DIRECTORY, path, as_attachment=True)
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
    showlegend=True,
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
if  len(os.listdir(PROCESSED_DIRECTORY))==12:
    loaded_processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_processed_data.pkl")
else:
    loaded_processed_data=pd.read_pickle(DEFAULT_DATA + 'prefer_bar_data.pkl')
if len(os.listdir(DEFAULT_DATA))==12:
    dataset_all=pd.read_pickle(DEFAULT_DATA+'prefer_bar_data.pkl')
    stage_dataset_all=pd.read_pickle(DEFAULT_DATA+'stage_bar_data.pkl')
    df = pd.read_pickle(DEFAULT_DATA+'prefer_processed_data.pkl')
    stage_df=pd.read_pickle(DEFAULT_DATA+'stage_processed_data.pkl')
    report_cluster_km=pd.read_pickle(DEFAULT_DATA+"prefer_db_cluster_feature.pkl")
    stage_report_cluster_km=pd.read_pickle(DEFAULT_DATA+"stage_km_cluster_feature.pkl")
    report_cluster_db=pd.read_pickle(DEFAULT_DATA+"prefer_db_cluster_feature.pkl")
    stage_report_cluster_db=pd.read_pickle(DEFAULT_DATA+"stage_db_cluster_feature.pkl")

    report_groups=pd.read_pickle(DEFAULT_DATA+"prefer_group_feature.pkl")
    stage_report_groups=pd.read_pickle(DEFAULT_DATA+"stage_group_feature.pkl")
    correlation_matrix=pd.read_pickle(DEFAULT_DATA+"prefer_correlation_matrix.pkl")

    stage_correlation_matrix=pd.read_pickle(DEFAULT_DATA+"stage_correlation_matrix.pkl")
else:
    print("wrong default data")
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
                    html.P("Presented semantic extension:", style={"font-weight": "bold"},className="control_label"),
                    dcc.RadioItems(
                        id="data_semantic_selector",
                        options=[
                            {"label": "Preferred ", "value": "pr"},
                            {"label": "Stage", "value": "stg"},
                        ],
                        value="pr",
                        labelStyle={"display": "inline-block"},
                        className="dcc_control",
                    ),

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
                                [html.H6(id="stage"), html.P("Stage")],
                                id="stage_block",
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
                          value=[int(0.2 * len(dataset_all)), int(0.5 * len(dataset_all))]
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

#html.Button('Correlation Matrix',style={'marginLeft': '2%', 'width': '49%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}),
                html.Div([
                    html.Button('HRP',
                                style={'font-size': '14px','marginLeft': '2%', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                                id='btn-nclicks-1', n_clicks=0),
                    html.Button('Revised HRP',
                                style={'font-size': '14px', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                                id='btn-nclicks-2', n_clicks=0),
                    html.Button('OLO',
                                style={'font-size': '14px', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                                id='btn-nclicks-3', n_clicks=0),
                    html.Button('Revised OLO',
                                style={'font-size': '14px', 'marginRight': '2%',"color": "#FFFF", "backgroundColor": "#2F8FD2"},
                                id='btn-nclicks-4', n_clicks=0),
                    html.P("Presented semantic extension:", style={"font-weight": "bold"}, className="dcc_control"),

                    dcc.RadioItems(
                        id="data_semantic_correlation",
                        loading_state={"is_loading":True},
                        options=[
                            {"label": "Preferred ", "value": "pr"},
                            {"label": "Stage", "value": "stg"},
                        ],
                        value="pr",
                        labelStyle={"display": "inline-block"},
                        className="dcc_control",
                    ),
                    ],
                className="row flex-display"
                )

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
                            href="https://github.com/Lexise/ASP-Analysis/tree/master/data/default_raw_data",
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

                        dcc.Upload([html.Button("View Processed Data" ,id="view_button",style={"color":"#FFFF","backgroundColor":"#2F8FD2"})],id="view-processed-button",multiple=True),

                        dcc.ConfirmDialog(
                            id='confirm',
                            message='Do you want to visualize uploaded data?',
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
            ]),


            dcc.Tab(label='Scatter with Groups', style={ 'fontWeight': 'bold'}, children=[
                dcc.Graph(
                    id="scatter_groups",)


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

            ]),
        ]),
        html.Div([
            html.Div([
                html.Span("Semantics:", style={"font-weight": "bold"}),
                dcc.RadioItems(
                    id="semantic-method",
                    options=[
                        {"label": "PREFER", "value": "pr"},
                        {"label": "STAGE", "value": "stg"},
                    ],
                    labelStyle={"display": "inline-block"},
                    value="pr",
                )],
                style={'marginLeft': '2%', 'width': '28%'},
            ),
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
                style={'marginLeft': '2%', 'width': '28%'},

            ),

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
                style={'marginLeft': '2%', 'width': '28%'},
            ),

        ],
            className="row flex-display"
        ),



        ],
        className = "pretty_container"),



        html.Div(id='click-data'),
        html.Br(),
        dcc.Link(html.Button('Argument Analysis',style={'width': '32%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-argument'),
        dcc.Link(html.Button('3D Analysis',style={'marginLeft': '2%','width': '32%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-3d'),
        dcc.Link(html.Button('Correlation Matrix',style={'marginLeft': '2%', 'width': '32%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-correlation'),
        html.Hr(),
        html.Div([

            dcc.Upload([html.Button(children='Upload File',id="upload_button")], id="upload-data",style={'width': '13%','marginRight': '2.5%'},multiple=True ),
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

                html.Div([
                    html.P("Upload",className="dcc_control",style={"width":"91%",'textAlign': 'left'}),
                    html.Button(id='clear-upload', n_clicks=0, children='Clear',style={'width': '12%','font-size':'11px','textAlign': 'right'}),

                ],
                    className="row flex-display",
                    #style={'textAlign': 'right'}
                ),
                html.Ul(id="file-list", children=get_file_name(UPLOAD_DIRECTORY))
            ],
                className="pretty_container seven columns"),

            html.Div([
                html.P("Processed"),
                html.Ul(id="processed-list", children=get_file_name(ZIP_DIRECTORY))
            ],
                className="pretty_container seven columns")
        ],
            className="row flex-display")

    ])






ThreeD_analysis=html.Div([
                dcc.Link(html.Button('back'), href='/'),
                html.Div([
                    dcc.Graph(id="3d_scatter_cluster",className="row flex-display"),
                    dcc.Graph(id="3d_scatter_group", className="row flex-display"),
                    ],
                    className="row flex-display"
                ),
                    html.Div([
                        html.Div([
                                    html.P("Presented semantic extension:", style={"font-weight": "bold"}, className="dcc_control"),

                                    dcc.RadioItems(
                                        id="data_semantic_correlation_3d",
                                        loading_state={"is_loading":True},
                                        options=[
                                            {"label": "Preferred ", "value": "pr"},
                                            {"label": "Stage", "value": "stg"},
                                        ],
                                        value="pr",
                                        labelStyle={"display": "inline-block"},
                                        className="dcc_control",
                                    ),
                                    ],
                                style={'marginRight': '2%', 'width': '30%'},
                                className="row flex-display"
                                ),
                        html.Div([
                                    html.P("Dimensional Reduction Method:", style={"font-weight": "bold"},className="dcc_control"),
                                    dcc.RadioItems(
                                        id="reduction_method",
                                        options=[
                                            {"label": "Tsne ", "value": "tsne"},
                                            {"label": "SVD", "value": "svd"},
                                        ],
                                        labelStyle={"display": "inline-block"},
                                        value="tsne",
                                        className="dcc_control",
                                )],
                                style={'marginRight': '2%', 'width': '30%'},
                                className="row flex-display"
                        ),
                        html.Div([
                                    html.P("Cluster Algorithm:", style={"font-weight": "bold"}, className="dcc_control"),
                                    dcc.RadioItems(
                                        id="clustering-method",
                                        options=[
                                            {"label": "DBscan ", "value": "db"},
                                            {"label": "Kmeans", "value": "km"},
                                        ],
                                        labelStyle={"display": "inline-block"},
                                        value="db",
                                        className="dcc_control",
                                    )],

                            style={ 'width': '30%'},
                            className="row flex-display"
                            )
                        ],
                        className = "row flex-display"),
                    ],
                className="pretty_container")

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
    question=""
    pr_answer=""
    stg_answer=""
    if len(files)>1:
        for filename in files:
            try:
                if filename.endswith('.apx'):
                    # Assume that the user uploaded a CSV file
                    question = filename
                elif 'EE-PR' in filename:
                    # Assume that the user uploaded an excel file
                    pr_answer = filename
                elif "EE-STG" in filename:
                    stg_answer = filename
            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])
        zipname=files[0].strip("apx")+"zip"
        if question!="" and pr_answer!="" and stg_answer!="":
            start_time = time.time()
            print("start process")
            process_data(PROCESSED_DIRECTORY,UPLOAD_DIRECTORY+question, UPLOAD_DIRECTORY+pr_answer,eps, minpts, n_cluster)
            process_data(PROCESSED_DIRECTORY, UPLOAD_DIRECTORY + question, UPLOAD_DIRECTORY + stg_answer, eps, minpts, n_cluster)

            print("(whole)get processed data", time.time() - start_time)
        else:
            print("the input file is not correct.")


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
                global_store(eps, minpts, n_cluster)
                return "finish"
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
    [Input("clear-upload","n_clicks"), Input('signal', 'children'),Input('upload-data', 'filename'),],
    [State('upload-data', 'contents')]
)
def update_output(clear_click,children,uploaded_filenames, uploaded_file_contents ):
    """Save uploaded files and regenerate the file list."""
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if len(os.listdir(UPLOAD_DIRECTORY)) != 0 and 'clear-upload' in changed_id:#and n_click==None:
            clean_folder(UPLOAD_DIRECTORY)
            return ""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
            for name, data in zip(uploaded_filenames, uploaded_file_contents):#[],[]
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
    elif pathname=="/page-3d":
        return ThreeD_analysis
    else:
        return main_page


@app.callback(
    Output('click-data', 'children'),
    [Input('scatter_cluster', 'clickData'),Input('scatter_groups', 'clickData'),Input("semantic-method","value"),Input("clustering-method", "value"),Input('confirm', 'submit_n_clicks')])
def display_click_data(clickData1,clickData2, semantic1,method,n_click):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if  len(os.listdir(PROCESSED_DIRECTORY))==12 or n_click:#load and processed
        if semantic1 == "pr":
            processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_processed_data.pkl")
        else:
            processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "stage_processed_data.pkl")

    else:
        if semantic1 =="pr":
            processed_data = df
        else:
            processed_data = stage_df
    cluster_label = method + "_cluster_label"

    print("changed_id:",[p['prop_id'] for p in dash.callback_context.triggered])
    if clickData1 or clickData2:
        if "scatter_cluster" in changed_id:
            clickData=clickData1
            print("clickData1",clickData1)
        elif "scatter_groups" in changed_id:
            clickData = clickData2
        else:
            return None
        print("clickData2", clickData2)
        selected_point_id=clickData["points"][0]["customdata"]
        print("clickData:",clickData)
        selected_point=processed_data[processed_data.id==selected_point_id]
        selected_point_arg=','.join(str(x) for x in selected_point.arg)

        selected_point_cluster=selected_point[cluster_label]
        return html.Div([
                html.Table([
                    html.Tr([html.Th('Id'),
                             html.Th('Cluster'),
                             html.Th('Groups'),
                             html.Th('Arguments'),
                             # html.Th('Most Recent Click')
                             ]),
                             html.Tr([html.Td(selected_point_id),
                                      html.Td(selected_point_cluster),
                                      html.Td(selected_point.groups),
                                      html.Td(selected_point_arg.strip('[]')),
                                      # html.Td(button_id)
                                      ])
                             ],
                )
                ],
        className = "pretty_container")
    return None



@app.callback(Output('scatter_cluster', 'figure'),
              [ Input('signal', 'children'),Input("dimensional-reduction1", "value"), Input("semantic-method","value"),
                Input("clustering-method", "value"),  Input('confirm', 'submit_n_clicks')])

def generate_tabs1( content, reduction1, semantic1,  method, n_click):#processed_data, table1_data,table2_data ):
    if  len(os.listdir(PROCESSED_DIRECTORY))==12 or n_click:#load and processed
        if semantic1 == "pr":
            processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_processed_data.pkl")
        else:
            processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "stage_processed_data.pkl")

    else:
        if semantic1 =="pr":
            processed_data = df
        else:
            processed_data = stage_df

    if reduction1=="svd":
        x_axe="svd_position_x"
        y_axe="svd_position_y"
    else:
        x_axe = "tsne_position_x"
        y_axe = "tsne_position_y"

    cluster_label = method + "_cluster_label"

    inputdata=processed_data.copy()
    inputdata[cluster_label]=["Cluster "+str(a) for a in processed_data[cluster_label]]
    cluster_set = list(processed_data[cluster_label].unique())
    if len(cluster_set)<=52:
        clusters_symbol=cluster_set
    else:
        clusters_symbol=[None]*len(cluster_set)

    #let hover on the plot directly
    # figure1 = px.scatter(inputdata, x=x_axe, y=y_axe,  color=cluster_label, symbol=clusters_symbol,#symbol_sequence =[102]*len(inputdata),
    #                      hover_name=cluster_label,  hover_data={
    #                                                              x_axe:False, # remove species from hover data
    #                                                               y_axe: False,
    #                                                               cluster_label:False,
    #                                                              'id':True,
    #                                                              'arg':True, # add other column, default formatting
    #                                                              'groups':True, # add other column, customized formatting
    #                                                             }
    #                      )
    # #figure1.update_traces() #remove hover  hovertemplate=None,
    # figure1.update_xaxes(showgrid=False,visible=False,zerolinecolor="Black")
    # figure1.update_yaxes(showgrid=False,zeroline=True,visible=False,zerolinecolor="black")
    # figure1.update_layout(  clickmode='event+select', plot_bgcolor='rgba(0,0,0,0)',legend_title_text='',autosize=True)
    # figure1 = go.Figure(figure1)

    #remove hover, provide an extra table to show the arg information of selected data point

    fig = go.Figure()
    for x in cluster_set:
        fig.add_trace(go.Scatter(
            x=processed_data[processed_data[cluster_label]==x][x_axe],
            y=processed_data[processed_data[cluster_label]==x][y_axe],
            customdata=processed_data[processed_data[cluster_label]==x]["id"],
            mode='markers',
            name=str(x)+" cluster",
            #hovertemplate='Id:%{customdata} ',
            # hovertext=processed_data[processed_data[cluster_label]==x].arg,
            hoverinfo="none",#'none'
            marker=dict(
                symbol=clusters_symbol[x]
            ),
            showlegend=True
        ))
    fig.update_layout(xaxis={'showgrid': False,'visible': False,},
                      yaxis={'showgrid': False, 'visible': False, },
                      plot_bgcolor='rgba(0,0,0,0)',
                      clickmode='event+select')
    return fig






@app.callback(Output('scatter_groups', 'figure'),
              [Input('signal', 'children'),
               Input("dimensional-reduction1", "value"),
               Input("semantic-method", "value"),
               #Input("clustering-method", "value"),
               Input('confirm', 'submit_n_clicks')])
# @cache.memoize(TIMEOUT)
def generate_tabs2(content, reduction2, semantic2, n_click):  # method):
    if len(os.listdir(PROCESSED_DIRECTORY)) == 12 or n_click:  # load and processed
        if semantic2 == "pr":
            processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_processed_data.pkl")
        else:
            processed_data = pd.read_pickle(PROCESSED_DIRECTORY + "stage_processed_data.pkl")
    else:
        if semantic2 == "pr":
            processed_data = df
        else:
            processed_data = stage_df

    if reduction2 == "svd":
        x_axe = "svd_position_x"
        y_axe = "svd_position_y"
    else:
        x_axe = "tsne_position_x"
        y_axe = "tsne_position_y"
    #cluster_label = method + "_cluster_label"


    # let hover on the plot directly
    # figure2 = px.scatter(processed_data, x=x_axe, y=y_axe, color="groups",
    #                      hover_name="groups", hover_data={x_axe: False,  # remove species from hover data
    #                                                       y_axe: False,
    #                                                       cluster_label: True,
    #                                                       'id': True,
    #                                                       'arg': True,  # add other column, default formatting
    #                                                       'groups': False,  # add other column, customized formatting
    #
    #                                                       })
    # figure2.update_xaxes(showgrid=False, visible=False)
    # figure2.update_yaxes(showgrid=False, zeroline=True, visible=False, zerolinecolor="black")
    # figure2.update_layout(plot_bgcolor='rgba(0,0,0,0)', legend_title_text='Groups', autosize=True)
    # figure2 = go.Figure(figure2)

    # remove hover, provide an extra table to show the arg information of selected data point
    groups_set = processed_data["groups"].unique()
    fig = go.Figure()
    for x in groups_set:
        fig.add_trace(go.Scatter(
            x=processed_data[processed_data["groups"] == x][x_axe],
            y=processed_data[processed_data["groups"] == x][y_axe],
            customdata=processed_data[processed_data["groups"] == x]["id"],
            mode='markers',
            name=str(x),
            hoverinfo="none",
            # marker=dict(
            #     symbol=x
            # ),
            showlegend=True
        ))
    fig.update_layout(xaxis={'showgrid': False, 'visible': False, },
                      yaxis={'showgrid': False, 'visible': False, },
                      plot_bgcolor='rgba(0,0,0,0)',
                      clickmode='event+select')
    return fig


@app.callback([ Output('table1', 'children'),
               Output('table2', 'children')],
              [Input('signal', 'children'), Input("semantic-method", "value"),
               Input("clustering-method", "value"),
               Input('confirm', 'submit_n_clicks')])
# @cache.memoize(TIMEOUT)
def generate_tabs3(content, semantic3,  table_method,
                   n_click):  # processed_data, table1_data,table2_data ):
    if len(os.listdir(PROCESSED_DIRECTORY)) == 12 or n_click:  # load and processed
        if semantic3 == "pr":
            group_table = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_group_feature.pkl")
            cluster_table = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_" + table_method + "_cluster_feature.pkl")
        else:
            group_table = pd.read_pickle(PROCESSED_DIRECTORY + "stage_group_feature.pkl")
            cluster_table = pd.read_pickle(PROCESSED_DIRECTORY + "stage_" + table_method + "_cluster_feature.pkl")

    else:
        if semantic3 == "pr":
            group_table = report_groups
            if table_method == "km":
                cluster_table = report_cluster_km
            else:
                cluster_table = report_cluster_db
        else:
            group_table = stage_report_groups
            if table_method == "km":
                cluster_table = stage_report_cluster_km
            else:
                cluster_table = stage_report_cluster_db

    # table
    if len(group_table) == 0:
        table1 = html.H5("No group feature")
    else:
        table1 = dash_table.DataTable(
            data=group_table.to_dict('records'),
            columns=[{"name": i, "id": i} for i in group_table.columns],
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll'
            },
            style_header={
                'fontWeight': 'bold'
            },
            style_cell={
                'font_size': '20px',
                'text_align': 'center'
            },
        )

    if not len(cluster_table):
        table2 = html.H5("No cluster Feature")
    else:
        table2 = dash_table.DataTable(
            data=cluster_table.to_dict('records'),
            columns=[{"name": i, "id": i} for i in cluster_table.columns],

            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll'
            },
            style_header={
                'fontWeight': 'bold'
            },
            style_cell={
                'font_size': '20px',
                'text_align': 'center'
            },
        )
    return  table1, table2


@app.callback(
    [Output("bar_chart", "figure"),Output("my-range-slider","figure")],
    [Input("data_present_selector", "value"),Input("my-range-slider", "value"),Input("sort_selector", "value"),Input("data_semantic_selector", "value")],
    )
@cache.memoize(TIMEOUT)
def make_bar_figure(present_data, valuelist,sort_state,semantic):


    if os.listdir(PROCESSED_DIRECTORY):  # load and processed
        if semantic == "pr":
            dataset_bar = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_bar_data.pkl")
        else:
            dataset_bar = pd.read_pickle(PROCESSED_DIRECTORY + "stage_bar_data.pkl")
    else:
        if semantic == "pr":
            dataset_bar = dataset_all
        else:
            dataset_bar = stage_dataset_all
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

# @app.callback(
#     [Output("confirm", "displayed")],
#     [Input("view-processed-button", "filename"), Input("view-processed-button", "contents"),Input("view_button","n_clicks")]
#     )
# def run_processed_data(uploaded_filenames, uploaded_file_contents,n_click ):
#     if n_click is None:
#         return [False]
#     if len(os.listdir(PROCESSED_DIRECTORY))!=0 and uploaded_filenames is not None and n_click%6==1:
#         clean_folder(PROCESSED_DIRECTORY)
#     print("click number: ",n_click)
#     print("upload_name:",uploaded_filenames)
#     if uploaded_filenames is not None and uploaded_file_contents is not None:
#         for name, data in zip([uploaded_filenames], [uploaded_file_contents]):
#             save_file(name, data, PROCESSED_DIRECTORY)
#         if len(os.listdir(PROCESSED_DIRECTORY)) >=6:
#             return [True]
#     return [False]

#
# @app.callback([Output("confirm", "displayed")],
#               [Input('view-processed-button', 'contents')],
#               [State('view-processed-button', 'filename')])
# def update_output(content, name):
#     if content is None:
#         return [False]
#     #for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
#         # the content needs to be split. It contains the type and the real content
#     content_type, content_string = content.split(',')
#     # Decode the base64 string
#     content_decoded = base64.b64decode(content_string)
#     # Use BytesIO to handle the decoded content
#     zip_str = io.BytesIO(content_decoded)
#     # Now you can use ZipFile to take the BytesIO output
#     zip_obj = ZipFile(zip_str, 'r')
#     zip_obj.extractall(PROCESSED_DIRECTORY)
#     return [True]


@app.callback(Output("confirm", "displayed"),
       [Input("hidden-div","figure")])
def show_confirm(value):
    if value :
        #print(n_click)
        if len(os.listdir(PROCESSED_DIRECTORY))==12:
            print("test:", value)
            return True
    return False


@app.callback(Output('hidden-div', 'figure'),
              [Input('view-processed-button', 'contents'),Input("view_button","n_clicks")],
              [State('view-processed-button', 'filename')])
def update_output(contents, n_click, name):
    if contents is None:
        return None
    #for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
        # the content needs to be split. It contains the type and the real content
    for content in contents:
        content_type, content_string = content.split(',')
        # Decode the base64 string
        content_decoded = base64.b64decode(content_string)
        # Use BytesIO to handle the decoded content
        zip_str = io.BytesIO(content_decoded)
        # Now you can use ZipFile to take the BytesIO output
        zip_obj = ZipFile(zip_str, 'r')
        zip_obj.extractall(PROCESSED_DIRECTORY)

    if  len(os.listdir(PROCESSED_DIRECTORY))==12: #n_click>=2 and n_click%2==0 and
        files = zip_obj.namelist()
        return files
    else:
        return None

@app.callback(Output('processed-list', 'children'),
              [ Input('signal', 'children'),Input('hidden-div', 'figure'), Input('confirm', 'submit_n_clicks')])#
def update_output(children1, children2, n_click):
    if  children2:
        return [html.Li(file_download_link(filename)) for filename in children2]

    return  get_file_name(ZIP_DIRECTORY)



@app.callback(
    [Output("selected_cluster","children"),
        Output('prefer_block', 'style'), Output('stage_block', 'style'), Output('stable', 'children'),  Output('prefer', 'children'),Output('stage', 'children')
        ,Output("pie_graph", "figure")],
    [Input('bar_chart', 'clickData') ,Input("clustering-method","value"), Input("data_semantic_selector", "value")])






def update_cluster_rate(clickData, cluster_method, semantic):
    if os.listdir(PROCESSED_DIRECTORY):  # load and processed
        if semantic == "pr":
            process_data = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_processed_data.pkl")
        else:
            process_data = pd.read_pickle(PROCESSED_DIRECTORY + "stage_processed_data.pkl")
    else:
        if semantic == "pr":
            process_data = df
        else:
            process_data = stage_df

    layout_pie = copy.deepcopy(layout)
    layout_pie["title"] = "Cluster Summary"
    if clickData is None:
        return "Selected Argument: None",{'display':'block'},{'display':'none'},"","","",dict(data=None, layout=layout_pie),
    temp=clickData["points"][0]
    arguments=int(re.search(r'\d+', temp["x"]).group())
    selected=[]
    result0= "Selected Argument:{}  \n".format(arguments)
    for index, row in process_data.iterrows():
        if arguments in row.arg:
            selected.append(index)
    if len(selected) == 0:
        return "No data has this argument","","","","","",dict(data=None, layout=layout_pie),
    data = process_data.loc[selected]
    result=""
    cluster_label=cluster_method+"_cluster_label"
    clusters=set(data[cluster_label])
    for cluster in clusters:
        num=len(data[data[cluster_label]==cluster])
        result=result+"{} % belong to cluster {} . ".format(num/len(data)*100,cluster)
    stable_value=len(data[data.groups == "stable"])/ len(data) * 100
    stable = "{:.2f}".format(stable_value) + "%"
    if semantic == "pr":
        prefer_value=len(data[data.groups == "prefer-"])/ len(data) * 100
        other="{:.2f}".format(prefer_value)+"%"
        pr_display={'display':'block'}
        stg_display={'display':'none'}
    else:
        stage_value = len(data[data.groups == "stage"]) / len(data) * 100
        other = "{:.2f}".format(stage_value) + "%"
        pr_display = {'display': 'none'}
        stg_display = {'display': 'block'}


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

    pie_figure = dict(data=data_bar, layout=layout_pie)

    result2=""
    for group in set(data.groups):
        num = len(data[data.groups == group])
        result2 = result2 + "{} % belong to group {}. ".format(num / len(data) * 100, group)

    return result0, pr_display,stg_display, stable,other,other,pie_figure


@app.callback(
    Output('basic-interactions', 'figure'),
    [Input('bar_chart', 'clickData'), Input("argument-dimensional-reduction","value"), Input("clustering-method","value"), Input("data_semantic_selector", "value")])

def update_graph(clickData, dimensional_reduction, cluster_method, semantic):

    if os.listdir(PROCESSED_DIRECTORY):  # load and processed
        if semantic == "pr":
            process_data = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_processed_data.pkl")
        else:
            process_data = pd.read_pickle(PROCESSED_DIRECTORY + "stage_processed_data.pkl")
    else:
        if semantic == "pr":
            process_data = df
        else:
            process_data = stage_df

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

@app.callback([Output('3d_scatter_cluster', 'figure'),Output('3d_scatter_group', 'figure')],
            [Input("data_semantic_correlation_3d", "value"), Input("reduction_method","value"), Input("clustering-method","value"),]
              )
def displayClick(semantic, reduction_method, cluster_method):
    if os.listdir(PROCESSED_DIRECTORY):  # load and processed
        if semantic == "pr":
            data = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_processed_data.pkl")
        else:
            data = pd.read_pickle(PROCESSED_DIRECTORY + "stage_processed_data.pkl")
    else:
        if semantic == "pr":
            data = df
        else:
            data = stage_df

    if reduction_method=="svd":
        x_axe="svd_position_x"
        y_axe="svd_position_y"
    else:
        x_axe = "tsne_position_x"
        y_axe = "tsne_position_y"
    cluster_label = cluster_method + "_cluster_label"
    cluster_set = data[cluster_label].unique()
    # fig = go.Figure(go.Scatter3d(
    #     x=data[x_axe],
    #     y=data[y_axe],
    #     z=[len(set(s)) for s in data["arg"]],
    #     text=["clusters: {}".format(x) for x in data[cluster_label]],
    #     name=data[cluster_label].tolist(),
    #     mode='markers',
    #     marker=dict(
    #         size=12,
    #         color=data[cluster_label],  # set color to an array/list of desired values
    #         colorscale='Viridis',  # choose a colorscale
    #         opacity=0.8
    #     )
    # ),
    #
    #
    # )
    #processed_data[processed_data[cluster_label] == cls][x_axe]
    text_list=[]
    inputdata=data.copy()
    for index, row in inputdata.iterrows():
        arg_list=list(row.arg)
        inserted_arg=[x for y in (arg_list[i:i + 6] + ["<br>"] * (i < len(arg_list) - 5) for
                     i in range(0, len(arg_list), 6)) for x in y]
        one_arg = str(inserted_arg).strip('[]')
        #input_arg=f_comma(one_arg, group=20, char='<br>')
        input_arg=one_arg.strip('"') #没用 ，去吃str里的‘
        text_list.append(input_arg)

    inputdata["arguments"]=text_list
    symbols=['circle', 'square', 'cross', 'square-open', 'x','diamond', 'diamond-open', 'circle-open']
    if len(cluster_set)<=len(symbols):
        cluster_symbol=symbols
    else:
        cluster_symbol=[None]*len(cluster_set)
    fig = go.Figure(data=[go.Scatter3d(
            x=data[data[cluster_label]==cls][x_axe],
            y=data[data[cluster_label]==cls][y_axe],
            z=[len(set(s)) for s in data[data[cluster_label]==cls]["arg"]],
            customdata=['<br>'+s for s in inputdata[inputdata[cluster_label]==cls]["arguments"]],
            hovertemplate='Id:%{text}</b><br>length:%{z} <br></b>Arguments:%{customdata} ',
            text=data[data[cluster_label]==cls].id,
            #text=["cluster: {}".format(x) for x in data[data[cluster_label]==cls][cluster_label]],
            mode='markers',
            name="cluster"+str(cls),
            marker=dict(
                size=3,
                color=cls,  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8,
                symbol=cluster_symbol[cls]
            ),

        ) for cls in cluster_set
        ],
        layout=go.Layout(title=dict(
            text='Clusters Distribution',
            xref="paper",
            yref="paper",
            x=0.5,
            font=dict(
                size=20,
            )
        ),
            autosize=False,
            width=850,
            height=850,
            showlegend=True,
        )
    )

    group_set = data["groups"].unique()
    fig2 = go.Figure(data=[go.Scatter3d(
        x=data[data["groups"] == cls][x_axe],
        y=data[data["groups"] == cls][y_axe],
        z=[len(set(s)) for s in data[data["groups"] == cls]["arg"]],
        # text=["groups: {}".format(x) for x in data[data["groups"] == cls]["groups"]],

        customdata=['<br>' + s for s in inputdata[inputdata["groups"] == cls]["arguments"]],
        hovertemplate='Id:%{text}</b><br>length:%{z} <br></b>Arguments:%{customdata} ',
        text=data[data["groups"] == cls].id,
        mode='markers',
        name="group " + str(cls),
        marker=dict(
            size=3,
            #color=cls,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    ) for cls in group_set
    ],
    layout=go.Layout( title=dict(
                                text='Groups Distribution',
                                xref="paper",
                                yref="paper",
                                x=0.5,
                                font=dict(
                                    size=20,
                                )
                        ),
                        autosize=False,
                        width=850,
                        height=850,
                        showlegend=True,
    ))
    # fig2.update_layout(
    #
    #     autosize=False,
    #     width=850,
    #     height=850,
    # )
    return fig,fig2


@app.callback(Output('correlation_hm', 'figure'),
              [Input('btn-nclicks-1', 'n_clicks'),Input('btn-nclicks-2', 'n_clicks'),Input('btn-nclicks-3', 'n_clicks'),Input('btn-nclicks-4', 'n_clicks'),Input("data_semantic_correlation", "value")])
def displayClick(btn1, btn2 , btn3, btn4, semantic):

    if os.listdir(PROCESSED_DIRECTORY):  # load and processed
        if semantic == "pr":
            data_correlation = pd.read_pickle(PROCESSED_DIRECTORY + "prefer_correlation_matrix.pkl")
            processed_data=pd.read_pickle(PROCESSED_DIRECTORY + "prefer_processed_data.pkl")
        else:
            data_correlation = pd.read_pickle(PROCESSED_DIRECTORY + "stage_correlation_matrix.pkl")
            processed_data=pd.read_pickle(PROCESSED_DIRECTORY + "stage_processed_data.pkl")
    else:
        if semantic == "pr":
            data_correlation = correlation_matrix
            processed_data=df
        else:
            data_correlation = stage_correlation_matrix
            processed_data = stage_df

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    abs_correlation = data_correlation.copy()
    for idx, raw in abs_correlation.iterrows():
        for x in raw.index:
            abs_correlation.loc[idx, x] = abs(raw[x])
    round_correlation = data_correlation.copy()
    threshold=1/(2*len(processed_data))
    for idx, raw in round_correlation.iterrows():
        for x in raw.index:
            to_round_value=raw[x]
            if -threshold<=to_round_value<=threshold:
                round_correlation.loc[idx, x]=0
            else:
                round_correlation.loc[idx, x] = round(to_round_value, 10)


    if btn1%2:
        temp_round_correlation=round_correlation.copy()
        distances = np.sqrt((1 - abs_correlation) / 2)
        res_order = compute_serial_matrix(distances.values, method='single')
        new_order = [abs_correlation.index[i] for i in res_order]

        ordered_correlation_matrix = round_correlation.reindex(index=new_order, columns=new_order) #data_correlation.reindex
        z_value=ordered_correlation_matrix.to_numpy()
        #original_z = z_value.copy()
        #a = pd.DataFrame(data=original_z, index=new_order, columns=new_order)
        #a.to_pickle("method1.pkl")
        #z_value[z_value==0]=np.nan
        x_value=[str(x)+"arg" for x in new_order]
        y_value=[str(x)+"arg" for x in new_order]

    elif btn2%2:
        all_new_order=innovative_correlation_clustering(round_correlation)
        new_test = round_correlation.reindex(index=all_new_order, columns=all_new_order)#data_correlation.reindex
        z_value=new_test.to_numpy()
        #original_z = z_value.copy()
        #a = pd.DataFrame(data=original_z, index=all_new_order, columns=all_new_order)
        #a.to_pickle("method2.pkl")
        #z_value[z_value == 0] = np.nan
        x_value=[str(x) + "arg" for x in new_test.columns]
        y_value=[str(x) + "arg" for x in new_test.index]

    elif btn3 % 2:
        new_order=abs_optimal_leaf_ordering(abs_correlation)
        ordered_correlation_matrix = round_correlation.reindex(index=new_order, columns=new_order)#data_correlation.reindex
        z_value = ordered_correlation_matrix.to_numpy()
        #original_z = z_value.copy()
        #a = pd.DataFrame(data=original_z, index=new_order, columns=new_order)
        #a.to_pickle("method3.pkl")
        #z_value[z_value == 0] = np.nan
        x_value = [str(x) + "arg" for x in new_order]
        y_value = [str(x) + "arg" for x in new_order]
    elif btn4 % 2:
        all_new_order=my_optimal_leaf_ordering(round_correlation)
        new_test = round_correlation.reindex(index=all_new_order, columns=all_new_order)#data_correlation.reindex
        z_value = new_test.to_numpy()
        #original_z = z_value.copy()
        #a=pd.DataFrame(data=original_z, index=all_new_order, columns=all_new_order)
        #a.to_pickle("method4.pkl")
        #z_value[z_value == 0] = np.nan
        x_value = [str(x) + "arg" for x in new_test.columns]
        y_value = [str(x) + "arg" for x in new_test.index]
    else:
        z_value=round_correlation.to_numpy()#data_correlation.reindex
        #original_z=z_value.copy()
        #z_value[z_value == 0] = np.nan
        x_value=[str(x)+"arg" for x in round_correlation.columns]
        y_value=[str(x)+"arg" for x in round_correlation.index]



    fig = go.Figure(go.Heatmap(
        z=z_value,
        x=x_value,
        y=y_value,
        #customdata =original_z,
        hovertemplate='value:%{z} <br><b>x:%{x}</b><br>y: %{y} ',
        name='',
        colorscale='RdBu',
       ))
    fig.update_layout(
        autosize=False,
        plot_bgcolor='rgba(0,0,0,0)',

        height=750)
    return fig




if __name__ == '__main__':
    app.run_server(debug=True)
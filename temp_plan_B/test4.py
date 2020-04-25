import dash
import re
import base64
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table
from urllib.parse import quote as urlquote
import json
from control import WELL_COLOR_new
import dash_table
import os
from flask_caching import Cache
from flask import Flask, send_from_directory
from clustering_correlation import compute_serial_matrix
import numpy as np
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
import copy
from dash.dependencies import Input, Output, State, ClientsideFunction
from flask_caching import Cache

UPLOAD_DIRECTORY = os.getcwd()+"/data/app_uploaded_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
    print("created")

server = Flask(__name__)
app = dash.Dash(server=server,meta_tags=[{"name": "viewport", "content": "width=device-width"}])
@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


#app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
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
)
#dataset=pd.read_pickle('new_test.pkl')
# with open(UPLOAD_DIRECTORY+'long-island-railroad_20090825_0512.gml.20.apx', 'r') as file:
#     test = file.read()
# print(test)


dataset_all=pd.read_pickle('long-island-railroad_attribute_frequency.pkl')

df = pd.read_pickle('long-island-railroad_tsne_epts=2.4_minp=10.pkl')
report_cluster=pd.read_pickle("long-island-railroad_cluster_report.pkl")
report_cluster.rename(columns={"character":"attribute_combination_feature"},inplace=True)

report_groups=pd.read_pickle("long-island-railroad_groups_report.pkl")

correlation_matrix=pd.read_pickle("answer2_correlation_matrix.pkl")

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

TIMEOUT = 60

attribute_analysis=html.Div([

    dcc.Link(html.Button('back'), href='/'),
    html.Div(
        [
            html.Div(
                [
                    html.H4(
                        "select range in histogram:",
                        className="control_label",
                    ),
                    dcc.RangeSlider(
                        id='my-range-slider',
                        min=0,
                        max=len(dataset_all),
                        step=1,
                        value=[5, 25]
                    ),
                    html.P("Presented data:", className="control_label"),
                    dcc.RadioItems(
                        id="data_present_selector",
                        options=[
                            {"label": "All ", "value": "all"},
                            {"label": "Interested", "value": "interested"},
                        ],
                        value="interested",
                        labelStyle={"display": "inline-block"},
                        className="dcc_control",
                    ),

                    dcc.Checklist(
                        id="sort_selector",
                        options=[{"label": "descending sort", "value": "decreased"}],
                        className="dcc_control",
                        value=[],
                    ),

                    html.Div(
                        [html.H5(id="selected_cluster")],
                        id="selected attribute",
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
                        className="row container-display",
                    ),

                ],
                className="pretty_container four columns",
                id="cross-filter-options",
            ),
            html.Div([dcc.Graph(id="bar_chart")],
                     className="pretty_container seven columns",
                     style={'width': '64%'}),

        ],

        className="row flex-display",
    ),
    html.Div([
        html.Div([
        dcc.Graph(
            id='basic-interactions'),
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
    dcc.Upload(html.Button('Upload File'),id="upload-data",),
    html.Ul(id="file-list"),
    html.Div(
        [

                html.Div(
                    [
                        html.H3(
                            "Answer Sets Analysis",
                            style={"margin-bottom": "10px"},
                        ),
                        html.H5(
                            "Feature Overview", style={"margin-top": "0px"}
                        ),
                    ],
                    className="one-half column",
                    id="title",
                    style={'width': '100%', 'textAlign': 'center'}
                )

        ],
        id="header",
        className="row flex-display",
        style={"margin-bottom": "35px"},
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
                    figure={
                        'data': [
                            {
                                'x': df[df.cluster_label == cls].position_x,
                                'y': df[df.cluster_label == cls].position_y,
                                'text': ["groups: {}".format(x) for x in df[df.cluster_label == cls]['groups']],
                                'name': cls,
                                'mode': 'markers',
                                'marker': {'size': 12,
                                            "color": WELL_COLOR_new[cls],
                                             'line': {'width': 0.5, 'color': 'white'}

                                }
                            } for cls in df['cluster_label'].unique()

                        ],
                        'layout': layout
                    }
                )
            ]),


            dcc.Tab(label='Scatter with Groups', style={ 'fontWeight': 'bold'}, children=[
                dcc.Graph(
                    id="scatter_groups",
                    figure={
                        'data': [
                            {
                                'x': df[df.groups == cls].position_x,
                                'y': df[df.groups == cls].position_y,
                                'text': ["Clusters: {}".format(x) for x in df[df.groups == cls]['cluster_label']],
                                'name': cls,
                                'mode': 'markers',
                                'marker': {'size': 12,
                                             'line': {'width': 0.5, 'color': 'white'}}
                            } for cls in df['groups'].unique()
                        ],
                        'layout': layout
                    }
                )
            ]),
            dcc.Tab(label='Feature Report',style={ 'fontWeight': 'bold'}, children=[
                html.Div([
                    html.Div([
                        dash_table.DataTable(
                            id='table1',
                            columns=[{"name": i, "id": i} for i in report_groups.columns],
                            style_table={
                                'maxHeight': '300px',
                                'overflowY': 'scroll'
                            },
                            style_header={
                                    'fontWeight': 'bold'
                                },
                            style_cell = {
                                            'font_family': 'cursive',
                                            'font_size': '20px',
                                            'text_align': 'center'
                                        },
                            data=report_groups.to_dict('records'),
                        ),
                    ],
                    className="pretty_container six columns"
                    ),
                    html.Div([
                        dash_table.DataTable(
                            id='table2',
                            columns=[{"name": i, "id": i} for i in report_cluster.columns],
                            style_table={
                                'maxHeight': '300px',
                                'overflowY': 'scroll'
                            },
                            style_header={
                                'fontWeight': 'bold'
                            },
                            style_cell={
                                'font_family': 'cursive',
                                'font_size': '20px',
                                'text_align': 'center'
                            },
                            data=report_cluster.to_dict('records'),
                        )
                    ],
                    className="pretty_container seven columns",
                    )
                #html.Div(id="orders_table", className="row table-orders"),
                ],
                 className="row flex-display"
                )
                ]),
            ])
        ],
        className = "pretty_container"
    ),



        dcc.Link(html.Button('Attribute Analysis',style={'width': '49%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-attribute'),
        dcc.Link(html.Button('Correlation Matrix',style={'marginLeft': '2%', 'width': '49%','font-size':'14px',"color":"#FFFF","backgroundColor":"#2F8FD2"}), href='/page-correlation'),

    ]),


app.layout = html.Div([


    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)



def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    print("save",data)
    #content_type, content_string = content.split(',')
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

@cache.memoize(timeout=TIMEOUT)
def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""
    print(uploaded_file_contents)
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        save_file(uploaded_filenames, uploaded_file_contents)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]


# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("bar_chart", "figure")],
)


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-attribute':
        return attribute_analysis
    elif pathname == '/page-correlation':
        return correlation_page
    else:
        return main_page





@app.callback(
    Output("bar_chart", "figure"),
    [Input("data_present_selector", "value"),Input("my-range-slider", "value"),Input("sort_selector", "value")],
    )
@cache.memoize(timeout=TIMEOUT)
def make_bar_figure(present_data, valuelist,sort_state):
   if present_data == "all":
       if sort_state == ["decreased"]:
           temp=dataset_all.sort_values(by=['frequency'],ascending=False, inplace=False)
           return set_bar_figure(temp, valuelist)
       else:
           print("no reaction")
           return set_bar_figure(dataset_all, valuelist)
   else:
       dataset=dataset_all[~dataset_all.rate.isin([0,100])]
       min = int(valuelist[0] * len(dataset) / len(dataset_all))
       max= int(valuelist[1] * len(dataset) / len(dataset_all))
       if sort_state == ["decreased"]:
           temp=dataset.sort_values(by=['frequency'],ascending=False, inplace=False)
           return set_bar_figure(temp, [min,max])
       else:
           return set_bar_figure(dataset, [min,max])

def set_bar_figure(attribute_data, valuelist):
    layout_count = copy.deepcopy(layout)
    select_idx=range(valuelist[0],valuelist[1])
    selected=attribute_data.iloc[select_idx]
    selected["order"]=range(len(selected))

    data = [dict(
            type="bar",
            x=list(selected["attribute"]),
            y=list(selected["rate"]),
            hovertext={"fontsize":20},
            #hovertext=["attribute:{arg},rate:{percent}".format(arg=row.attribute,percent=row.rate) for index,row in selected.iterrows()],
            name="All Wells"
        )]


    layout_count["title"] = "Rate/Attribute"

    layout_count["dragmode"] = "select"
    layout_count["showlegend"] = False
    layout_count["autosize"] = True,
    layout_count["titlefont"] = {"size": 28}
    layout_count["marker"] = {"fontsize": 20}
    figure = dict(data=data, layout=layout_count)
    return figure





@app.callback(
    [Output("selected_cluster","children"), Output('stable', 'children'), Output('prefer', 'children'),Output('complete', 'children'),Output("pie_graph", "figure")],
    [Input('bar_chart', 'clickData')])

def update_cluster_rate(clickData):
    layout_pie = copy.deepcopy(layout)
    layout_pie["title"] = "Cluster Summery"
    if clickData is None:
        return "Selected Attribute: None","","","",dict(data=None, layout=layout_pie),
    temp=clickData["points"][0]
    attributes=int(re.search(r'\d+', temp["x"]).group())
    selected=[]
    result0= "Selected Attribute:{}  \n".format(attributes)
    for index, row in df.iterrows():
        if attributes in row.arg:
            selected.append(index)
    if len(selected) == 0:
        return "No data has this attribute","","","",dict(data=None, layout=layout_pie),
    data = df.loc[selected]
    result=""
    for cluster in set(data.cluster_label):
        num=len(data[data.cluster_label==cluster])
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
    for cluster in set(data.cluster_label):
        result["cluster"].append(str(cluster) + " cluster")
        num = len(data[data.cluster_label == cluster])
        result["num"].append(num)
        # result["rate"].append(num/len(data))

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
            marker=dict(colors=[WELL_COLOR_new[i] for i in set(data.cluster_label)]),
            # dict(colors=["#fac1b7", "#a9bb95", "#92d8d8"]),
            # domain={"x": [0, 0.45], "y": [0.2, 0.8]},
        )
    ]
    layout_pie["title"] = "Cluster Summery"
    layout_pie["font"] = dict(color="#777777")
    layout_pie["legend"] = dict(
        font=dict(color="#CCCCCC", size="10"), orientation="h", bgcolor="rgba(0,0,0,0)"
    )

    figure = dict(data=data_bar, layout=layout_pie)

    result2=""
    for group in set(data.groups):
        num = len(data[data.groups == group])
        result2 = result2 + "{} % belong to group {}. ".format(num / len(data) * 100, group)
    print(result2)

    return result0,stable,prefer,complete,figure


@app.callback(
    Output('basic-interactions', 'figure'),
    [Input('bar_chart', 'clickData')])

def update_graph(clickData):

    layout_scatter = copy.deepcopy(layout)
    layout_scatter["title"]="Distribution of Select Attribute"
    layout_scatter["clickmode"]= 'event+select'
    if clickData is None:
        return {
            'data':[],
            'layout': layout_scatter
        }
    temp=clickData["points"][0]

    attributes=int(re.search(r'\d+', temp["x"]).group())

    selected=[]
    for index, row in df.iterrows():
        if attributes in row.arg:
            selected.append(index)

    data=df.loc[selected]
    unselected_data=df[~df.index.isin(selected)]
    return {
        'data': [
            dict(
            x=data['position_x'],
            y=data['position_y'],
            text=["clusters: {}".format(x) for x in data['cluster_label']],
            name="selected",
            mode='markers',
            marker={
                'size': 12,
                'opacity': 1.0,
                'line': {'width': 0.5, 'color': 'white'}
            }),
            dict(
                x = unselected_data['position_x'],
                y = unselected_data['position_y'],
                text=["clusters: {}".format(x) for x in unselected_data['cluster_label']],
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

    if btn1 > 0:
        distances = np.sqrt((1 - correlation_matrix) / 2)
        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(distances.values, method='single')
        data = [{
                    "type" : "heatmap",
                    "z" : ordered_dist_mat,
                    "x" : [str(x)+"arg" for x in res_order],
                    "y" : [str(x)+"arg" for x in res_order],
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
                            "z" : correlation_matrix.to_numpy(),
                            "x" : [str(x)+"arg" for x in correlation_matrix.columns],
                            "y" : [str(x)+"arg" for x in correlation_matrix.index],
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
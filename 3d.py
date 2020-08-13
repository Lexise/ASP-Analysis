import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
import pandas as pd
import json

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})



import pandas as pd
df = pd.DataFrame(columns=['x_val','z_val','y_val'])
df.loc['a'] = [11.313449728149418,0.13039110880256777,0.5386387766748618]
df.loc['b'] = [11.321463427315383,0.2360697833061771,1.32441455152796]
df.loc['c'] = [10.127132005050942,0.23085014016641864,1.0961116175427044]
df.loc['d'] = [11.639819269465233,0.0958798324712593,0.6506370305953094]
df.loc['e'] = [8.892696370438149,0.08223988244819926,0.6440321391968353]
df.loc['f'] = [6.711586646011124,0.3657515974938044,0]
df.loc['g'] = [7.095030650760687,0,0.5723062047617504]
df.loc['h'] = [6.4523124528415,0,1.293852184258803]
df.loc['i'] = [7.165105300812886,0.4151365420301895,-0.5920674079031845]
df.loc['j'] = [7.480703395137295,0.14284429977557123,1.0600936940126982]
df.loc['k'] = [5.570775744372319,0,0]
df.loc['l'] = [4.358946555449826,0,0]

def create_figure(skip_points=[]):
    dfs = df.drop(skip_points)
    return px.scatter_3d(dfs, x = 'x_val', y = 'y_val', z = 'z_val')
f= create_figure()

app.layout = html.Div([html.Button('Delete', id='delete'),
                    html.Button('Clear Selection', id='clear'),
                    dcc.Graph(id = '3d_scat', figure=f),
                    html.Div('selected:'),
                    html.Div(id='selected_points'), #, style={'display': 'none'})),
                    html.Div('deleted:'),
                    html.Div(id='deleted_points') #, style={'display': 'none'}))
])

@app.callback(Output('deleted_points', 'children'),
            [Input('delete', 'n_clicks')],
            [State('selected_points', 'children'),
            State('deleted_points', 'children')])
def delete_points(n_clicks, selected_points, delete_points):
    print('n_clicks:',n_clicks)
    if selected_points:
        selected_points = json.loads(selected_points)
    else:
        selected_points = []

    if delete_points:
        deleted_points = json.loads(delete_points)
    else:
        deleted_points = []
    ns = [p['pointNumber'] for p in selected_points]
    new_indices = [df.index[n] for n in ns if df.index[n] not in deleted_points]
    print('new',new_indices)
    deleted_points.extend(new_indices)
    return json.dumps(deleted_points)



@app.callback(Output('selected_points', 'children'),
            [Input('3d_scat', 'clickData'),
                Input('deleted_points', 'children'),
                Input('clear', 'n_clicks')],
            [State('selected_points', 'children')])
def select_point(clickData, deleted_points, clear_clicked, selected_points):
    ctx = dash.callback_context
    ids = [c['prop_id'] for c in ctx.triggered]

    if selected_points:
        results = json.loads(selected_points)
    else:
        results = []


    if '3d_scat.clickData' in ids:
        if clickData:
            for p in clickData['points']:
                if p not in results:
                    results.append(p)
    if 'deleted_points.children' in ids or  'clear.n_clicks' in ids:
        results = []
    results = json.dumps(results)
    return results

@app.callback(Output('3d_scat', 'figure'),
            [Input('selected_points', 'children'),
            Input('deleted_points', 'children')],
            [State('deleted_points', 'children')])
def chart_3d( selected_points, deleted_points_input, deleted_points_state):
    global f
    deleted_points = json.loads(deleted_points_state) if deleted_points_state else []
    f = create_figure(deleted_points)

    selected_points = json.loads(selected_points) if selected_points else []
    if selected_points:
        f.add_trace(
            go.Scatter3d(
                mode='markers',
                x=[p['x'] for p in selected_points],
                y=[p['y'] for p in selected_points],
                z=[p['z'] for p in selected_points],
                marker=dict(
                    color='red',
                    size=5,
                    line=dict(
                        color='red',
                        width=2
                    )
                ),
                showlegend=False
            )
        )

    return f

if __name__ == '__main__':
    app.run_server(debug=True)
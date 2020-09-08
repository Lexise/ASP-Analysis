
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash(name=__name__)

app.layout = html.Div(
    children=[
        html.Button(
            'Button',
            id='upload_button'),

        dcc.ConfirmDialog(
            id='upload_success_1',
            message="I want to say this...",
        ),

        dcc.ConfirmDialog(
            id='upload_success_2',
            message="... and I want to say that.",
        ),

        html.Div(id='output-confirm_1', style={'display': 'none'}),
        html.Div(id='output-confirm_2', style={'display': 'none'}),
    ]
)


@app.callback(Output('upload_success_2', 'displayed'),
              [Input('upload_success_1', 'submit_n_clicks')])
def second_confirm_dialog(n_clicks):
    print("second_confirm_dialog", n_clicks)
    if n_clicks:
        return True


@app.callback(Output('upload_success_1', 'displayed'),
              [Input('upload_button', 'n_clicks')])
def first_confirm_dialog(n_clicks):
    print("first_confirm_dialog", n_clicks)
    if n_clicks:
        # do something
        return True
    return False


@app.callback(
    Output('output-confirm_1', 'children'),
    [Input('upload_success_1', 'submit_n_clicks')]
)
def update_output_1(submit_n_clicks):
    print("first dummy", submit_n_clicks)
    if submit_n_clicks:
        return "dummy"


@app.callback(
    Output('output-confirm_2', 'children'),
    [Input('upload_success_2', 'submit_n_clicks')]
)
def update_output_2(submit_n_clicks):
    print("second dummy", submit_n_clicks)
    if submit_n_clicks:
        return "dummy"


if __name__ == '__main__':
    app.run_server(debug=False)
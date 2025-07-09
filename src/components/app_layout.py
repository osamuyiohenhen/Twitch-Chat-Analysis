from dash import Dash, html, dcc
from src.components.dropdown import create_dropdown
from . import ids


def create_layout(app: Dash) -> html.Div:
    return html.Div(className="app-div", children=[
        html.H1(app.title, style={'textAlign': 'center', 'fontSize': 40, 'paddingTop': '20px', 'paddingBottom': '15px'}), html.Hr(),
        html.Div(className=ids.DROPDOWN_CONTAINER, children=[create_dropdown(app)]),  
        html.Button('Select All', id=ids.SELECT_BUTTON, style={'marginBottom': '10px'}),
        html.Br(),
        # html.Div([
        #     dcc.Input(id=ids.MY_INPUT, placeholder='Enter anything here', type='text')]),
        # html.Br(),
        # html.Div(id=ids.MY_OUTPUT, children=''), # Pastes anything from id
        html.Div(id=ids.GRAPH_CONTAINER, children=[]),
        ])


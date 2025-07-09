from dash import Dash, html, dcc
from . import ids

def create_dropdown(app: Dash) -> html.Div:
    return html.Div(
        children=[
            html.H6("Select Bars to Display", style={}),
            dcc.Dropdown(
                options=[{"label": bar, "value": bar} for bar in ids.BAR_OPTIONS],
                placeholder="Select a sentiment bar",
                value=ids.BAR_OPTIONS,
                multi=True,
                id=ids.BAR_SELECTION,
            )
            ]
    )

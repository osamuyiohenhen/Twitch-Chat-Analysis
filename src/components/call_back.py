from dash import html, Input, Output, callback, dcc, State, no_update

from . import ids

# Test callback (input bar)
# @callback(
#     Output(component_id=ids.MY_OUTPUT, component_property='children'), # Fill  output info
#     Input(component_id=ids.MY_INPUT, component_property='value') # With input info
# )

# def update_output_div(input_value) -> str:
#     if input_value == None:
#         input_value = ''
#     return f'Output: {input_value}'

# Select all button callback
@callback(
    Output(component_id=ids.BAR_SELECTION, component_property='value'), # Fill output info
    Output(component_id=ids.SELECT_BUTTON, component_property='children'),
    Input(component_id=ids.SELECT_BUTTON, component_property='n_clicks'), # With input info
    State(component_id=ids.BAR_SELECTION, component_property='value'),
    prevent_initial_call=True
)
def select_all_button(current_dropdown_value): # Toggles dropdown value between all options and none, while changing text
    all_options = set(ids.BAR_OPTIONS)
    current_value_set = set(current_dropdown_value if current_dropdown_value is not None else [])

    if current_value_set == all_options:
        new_dropdown_value = []
        new_button_text = "Select All"
        print("Toggle button: Deselecting all.")
    else:
        new_dropdown_value = ids.BAR_OPTIONS
        new_button_text = "Deselect All"
        print("Toggle button: Selecting all.")

    # Return the new values for BOTH outputs
    return new_dropdown_value, new_button_text

# Updating bars
import plotly.express as px
import pandas as pd

def create_positive_graph_figure():
    print("Creating Positive Graph...")
    df = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Value': [10, 15, 5]})
    fig = px.bar(df, x='Category', y='Value', title="Positive Sentiment Analysis")
    return fig

def create_neutral_graph_figure():
    print("Creating Neutral Graph...")
    df = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Value': [10, 15, 5]})
    fig = px.bar(df, x='Category', y='Value', title="Neutral Sentiment Analysis")
    return fig

def create_negative_graph_figure():
    print("Creating Negative Graph...")
    df = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Value': [10, 15, 5]})
    fig = px.bar(df, x='Category', y='Value', title="Negative Sentiment Analysis")
    return fig

@callback (
    Output(component_id=ids.GRAPH_CONTAINER, component_property='children'),
    Input(component_id=ids.BAR_SELECTION, component_property='value')
)
def create_graphs(selected_graphs):
    
    graphs_to_display = []

    if selected_graphs is None:
        print("No sentiments selected.")
    
    if "Positive" in selected_graphs:
        graphs_to_display.append(
            dcc.Graph(
                figure=create_positive_graph_figure(),
                style={'margin-bottom': '20px'}
            )
        )


    if "Neutral" in selected_graphs:
        graphs_to_display.append(
            dcc.Graph(
                create_neutral_graph_figure(),
                style={'margin-bottom': '20px'}
            )
        )

    if "Negative" in selected_graphs:
        graphs_to_display.append(
            dcc.Graph(
                figure=create_negative_graph_figure(),
                style={'margin-bottom': '20px'}
            )
        )

    print(f"Returning {len(graphs_to_display)} graphs.")
    
    return graphs_to_display
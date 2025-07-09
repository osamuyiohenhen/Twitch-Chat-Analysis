from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP
from config import channel

from src.components.app_layout import create_layout
import src.components.call_back

def main() -> None:
    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = f"{channel}'s Twitch Chat Analysis"
    app.layout = create_layout(app)     

    app.run(debug=True)


if __name__ == "__main__":
    main()
"""
Hugging Face Space App for Steel Client Purchase Prediction

Before running:
1. Put your Excel dataset inside data/
2. Run the training file once:
   python steel_clients_pipeline.py --data data/STEELMANUF_CLIENTS_SV.xlsx --sheet DB

Then launch:
   python app.py
"""

from pathlib import Path
import subprocess
import sys

import gradio as gr
import joblib
import pandas as pd


DATA_PATH = "data/STEELMANUF_CLIENTS_SV.xlsx"
SHEET_NAME = "DB"
MODEL_PATH = "models/best_model.pkl"


def train_if_needed():
    if not Path(MODEL_PATH).exists():
        if not Path(DATA_PATH).exists():
            raise FileNotFoundError(
                "Model file was not found and dataset is missing. "
                "Upload STEELMANUF_CLIENTS_SV.xlsx inside the data/ folder."
            )

        subprocess.run(
            [
                sys.executable,
                "steel_clients_pipeline.py",
                "--data",
                DATA_PATH,
                "--sheet",
                SHEET_NAME,
                "--no-images",
            ],
            check=True,
        )


train_if_needed()
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]
best_model_name = artifact["best_model_name"]


def predict_purchases(
    number_of_negotiations,
    sessions_year,
    tons_confirmed,
    sessions_onwebapp,
    change_distcenter,
    delivery_or_pickup,
    sessions_attended_byexecutive,
    average_actions,
    maxnumber_pages_visited,
    avgnumber_pages_visited,
    use_of_excel_tool,
    use_of_client_catalogue,
    classification,
):
    row = {col: 0 for col in feature_columns}

    base_values = {
        "NUMBER_OF_NEGOTIATIONS": number_of_negotiations,
        "SESSIONS_YEAR": sessions_year,
        "TONS_CONFIRMED": tons_confirmed,
        "SESSIONS_ONWEBAPP": sessions_onwebapp,
        "CHANGE_DISTCENTER": change_distcenter,
        "DELIVERY_OR_PICKUP": delivery_or_pickup,
        "SESSIONS_ATTENDED_BYEXECUTIVE": sessions_attended_byexecutive,
        "AVERAGE_ACTIONS": average_actions,
        "MAXNUMBER_PAGES_VISITED": maxnumber_pages_visited,
        "AVGNUMBER_PAGES_VISITED": avgnumber_pages_visited,
        "USE_OF_EXCEL_TOOL": use_of_excel_tool,
        "USE_OF_CLIENT_CATALOGUE": use_of_client_catalogue,
    }

    for key, value in base_values.items():
        if key in row:
            row[key] = value

    classification_col = f"CLASSIFICATION_{classification}"
    if classification_col in row:
        row[classification_col] = 1

    input_df = pd.DataFrame([row], columns=feature_columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    prediction = max(0, round(float(prediction), 2))

    return {
        "Predicted Number of Purchases": prediction,
        "Best Model Used": best_model_name,
    }


with gr.Blocks(title="Steel Client Purchase Prediction") as demo:
    gr.Markdown(
        """
        # Steel Client Purchase Prediction

        This app predicts the expected number of purchases for a steel client based on web activity, negotiations,
        tonnage behavior, and platform usage.

        **Model target:** `NUMBER_OF_PURCHASES`
        """
    )

    with gr.Row():
        with gr.Column():
            number_of_negotiations = gr.Number(label="Number of Negotiations", value=2)
            sessions_year = gr.Number(label="Sessions per Year", value=20)
            tons_confirmed = gr.Number(label="Tons Confirmed", value=50)
            sessions_onwebapp = gr.Number(label="Sessions on Web App", value=10)
            average_actions = gr.Number(label="Average Actions", value=15)
            maxnumber_pages_visited = gr.Number(label="Max Number of Pages Visited", value=8)

        with gr.Column():
            avgnumber_pages_visited = gr.Number(label="Average Number of Pages Visited", value=4)
            change_distcenter = gr.Radio([0, 1], label="Changed Distribution Center", value=0)
            delivery_or_pickup = gr.Radio([0, 1], label="Delivery or Pickup", value=1)
            sessions_attended_byexecutive = gr.Number(label="Sessions Attended by Executive", value=2)
            use_of_excel_tool = gr.Radio([0, 1], label="Use of Excel Tool", value=0)
            use_of_client_catalogue = gr.Radio([0, 1], label="Use of Client Catalogue", value=1)
            classification = gr.Dropdown(
                choices=["A", "B", "C", "D", "E"],
                label="Client Classification",
                value="A",
            )

    predict_button = gr.Button("Predict Purchases")
    output = gr.JSON(label="Prediction Output")

    predict_button.click(
        fn=predict_purchases,
        inputs=[
            number_of_negotiations,
            sessions_year,
            tons_confirmed,
            sessions_onwebapp,
            change_distcenter,
            delivery_or_pickup,
            sessions_attended_byexecutive,
            average_actions,
            maxnumber_pages_visited,
            avgnumber_pages_visited,
            use_of_excel_tool,
            use_of_client_catalogue,
            classification,
        ],
        outputs=output,
    )

demo.launch()

import os
import pandas as pd
import logging
from datetime import datetime
from google.oauth2.service_account import Credentials
import gspread
import joblib
import numpy as np
from fan_state_detector_clustering import FanStateUnsupervisedDetector
from influxdb_client import InfluxDBClient, Point, WriteOptions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Predictor")

# InfluxDB setup (read from env)
INFLUX_URL = os.environ["INFLUX_URL"]
INFLUX_TOKEN = os.environ["INFLUX_TOKEN"]
INFLUX_ORG = os.environ["INFLUX_ORG"]
INFLUX_BUCKET = os.environ["INFLUX_BUCKET"]

# Google Sheets auth
google_creds = os.environ["GCP_CREDENTIALS_JSON"]
with open("temp_google_creds.json", "w") as f:
    f.write(google_creds)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("temp_google_creds.json", scopes=scope)
client = gspread.authorize(creds)

# Load model artifacts
model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Track last seen timestamp
last_seen_time_file = "last_seen.txt"
try:
    with open(last_seen_time_file, "r") as f:
        last_seen = pd.to_datetime(f.read().strip())
except FileNotFoundError:
    last_seen = pd.Timestamp("1970-01-01")

# Fetch data from Google Sheets
sheet = client.open("EX301 Vibration Sensor").worksheet("Data")
data = sheet.get_all_records()
df = pd.DataFrame(data)
df['time stamp'] = pd.to_datetime(df['time stamp'])

# Filter only new data
new_data = df[df['time stamp'] > last_seen].copy()
if new_data.empty:
    logger.info("No new data found.")
    exit(0)

# Preprocess and predict
X = new_data[["x_acc", "x_displacement", "x_frq", "x_speed",
              "y_acc", "y_displacement", "y_frq", "y_speed",
              "z_acc", "z_displacement", "z_frq", "z_speed"]]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)

# Attach predictions to new_data DataFrame
new_data["predicted_label"] = predictions

# Write predictions to InfluxDB
influx = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx.write_api(write_options=WriteOptions(batch_size=1))

for _, row in new_data.iterrows():
    point = Point("fan_predictions") \
        .tag("source", "gh-action") \
        .field("predicted_label", int(row["predicted_label"])) \
        .time(row["time stamp"])
    write_api.write(bucket=INFLUX_BUCKET, record=point)

logger.info(f"Wrote {len(new_data)} predictions to InfluxDB.")

# Save latest timestamp
latest = new_data['time stamp'].max()
with open(last_seen_time_file, "w") as f:
    f.write(str(latest))

# Check if 'predicted_label' column exists in Google Sheets, if not, add it
headers = sheet.row_values(1)
if 'predicted_label' not in headers:
    # Add the 'predicted_label' column to the header row
    sheet.add_cols(1)  # Adds one new column at the end
    sheet.update_cell(1, len(headers) + 1, 'predicted_label')  # Update the header

# Batch Update Google Sheet with predicted labels
def batch_update_predictions(df_new, df_pred, worksheet, col_indices):
    """Batch update predictions for large datasets."""
    cells = []
    
    for i, row in df_new.iterrows():
        try:
            row_index = row.name + 2  # Google Sheets rows are 1-indexed, +2 to skip header
            
            # Add all prediction columns to batch update
            for col_name, col_idx in col_indices.items():
                if col_name in df_pred.columns:
                    value = df_pred.loc[i, col_name]
                    # Convert to JSON-serializable type
                    serializable_value = convert_to_serializable(value)
                    cells.append({
                        'range': f'{gspread.utils.rowcol_to_a1(row_index, col_idx)}',
                        'values': [[serializable_value]]
                    })
        except Exception as e:
            logger.error(f"Error preparing batch update for row {i}: {str(e)}")
    
    if cells:
        try:
            worksheet.batch_update(cells)
            logger.info(f"Batch update completed for {len(cells)//len(col_indices)} predictions.")
        except Exception as e:
            logger.error(f"Error in batch update: {str(e)}")

def convert_to_serializable(value):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif pd.isna(value):
        return ""
    else:
        return str(value) if value is not None else ""

# Update Google Sheet with predictions
batch_update_predictions(new_data, new_data, sheet, {'predicted_label': 1})


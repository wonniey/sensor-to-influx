import os
import logging
import pandas as pd
import time
import threading
from datetime import datetime, timedelta
from fan_state_detector_clustering import EnhancedFanStateDetector
from google.oauth2.service_account import Credentials
import gspread
import gspread.utils
import joblib
import numpy as np

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths to save/load models
MODEL_PATH = "enhanced_model.joblib"
SCALER_PATH = "enhanced_scaler.joblib"
DETECTORS_PATH = "enhanced_detectors.joblib"
CONFIG_PATH = "enhanced_config.joblib"

# Google Sheets API info
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = '/path/to/your/credentials.json'
SPREADSHEET_ID = 'your_spreadsheet_id'
WORKSHEET_NAME = 'Data'

def load_gsheet_client():
    """Load Google Sheets client."""
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def preprocess_gsheet_data(df):
    """Preprocess data from Google Sheets to handle formatting issues."""
    df.columns = df.columns.str.strip()  # Handle potential column name issues
    # Handle timestamp column names
    timestamp_cols = ['time stamp', 'timestamp', 'Time Stamp']
    for col in timestamp_cols:
        if col in df.columns:
            df['time stamp'] = df[col]
            break
    # Convert numeric columns
    numeric_columns = ['battery_value', 'temperature_value', 'x_acc', 'x_displacement', 
                       'x_frq', 'x_speed', 'y_acc', 'y_displacement', 'y_frq', 'y_speed',
                       'z_acc', 'z_displacement', 'z_frq', 'z_speed', 'rssi', 'snr', '']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def run_on_new_data(detector):
    """Run prediction on new data from Google Sheets."""
    try:
        client = load_gsheet_client()
        worksheet = client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
        records = worksheet.get_all_records()
        df_live = pd.DataFrame.from_records(records)
        
        if df_live.empty:
            logger.info("No data found in Google Sheet.")
            return
        
        # Preprocess the data
        df_live = preprocess_gsheet_data(df_live)
        
        # Get current headers
        header = worksheet.row_values(1)
        
        # Add prediction columns if they don't exist
        prediction_cols = ['predicted_label', 'degree_of_anomaly', 'confidence', 'cluster']
        col_indices = {}
        
        for col in prediction_cols:
            if col not in header:
                worksheet.update_cell(1, len(header) + 1, col)
                header.append(col)
            col_indices[col] = header.index(col) + 1
        
        # Filter data that hasn't been processed yet
        if 'predicted_label' in df_live.columns:
            df_new = df_live[df_live['predicted_label'].isna() | (df_live['predicted_label'] == '')]
        else:
            df_new = df_live.copy()
        
        if len(df_new) > 0:
            logger.info(f"Processing {len(df_new)} new records.")
            
            # Run prediction
            df_pred = detector.predict(df_new, timestamp_col='time stamp')
            
            # Update Google Sheet
            if len(df_new) > 10:
                batch_update_predictions(df_new, df_pred, worksheet, col_indices)
            else:
                individual_update_predictions(df_new, df_pred, worksheet, col_indices)
            
            logger.info(f"Predictions updated to Google Sheet at {datetime.now()}")
            
            # Log summary statistics
            log_prediction_summary(df_pred)
            
        else:
            logger.info("No new data to predict on.")
            
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")

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

def individual_update_predictions(df_new, df_pred, worksheet, col_indices):
    """Update predictions individually for small batches."""
    for i, row in df_new.iterrows():
        try:
            row_index = row.name + 2  # Google Sheets rows are 1-indexed, +2 to skip header
            
            # Update each prediction column
            for col_name, col_idx in col_indices.items():
                if col_name in df_pred.columns:
                    value = df_pred.loc[i, col_name]
                    # Convert to JSON-serializable type
                    serializable_value = convert_to_serializable(value)
                    worksheet.update_cell(row_index, col_idx, serializable_value)
                    
        except Exception as e:
            logger.error(f"Error updating row {i}: {str(e)}")

def batch_update_predictions(df_new, df_pred, worksheet, col_indices):
    """Batch update predictions for large datasets."""
    cells = []
    
    for i, row in df_new.iterrows():
        try:
            row_index = row.name + 2
            
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

def log_prediction_summary(df_pred):
    """Log summary statistics of predictions."""
    try:
        summary = df_pred['predicted_label'].value_counts()
        logger.info("Prediction Summary:")
        for state, count in summary.items():
            logger.info(f"  {state}: {count}")
        
        # Log anomaly statistics
        anomalies = df_pred[df_pred['predicted_label'] == 'Anomaly']
        if len(anomalies) > 0:
            logger.warning(f"Found {len(anomalies)} anomalies!")
            avg_anomaly_score = anomalies['degree_of_anomaly'].mean()
            logger.warning(f"Average anomaly score: {avg_anomaly_score:.3f}")
    
    except Exception as e:
        logger.error(f"Error creating prediction summary: {str(e)}")

def auto_predict(detector, interval_minutes):
    """Continuously run predictions at specified intervals."""
    while True:
        logger.info(f"Running prediction task at {datetime.now()}")
        run_on_new_data(detector)
        
        next_run_time = datetime.now() + timedelta(minutes=interval_minutes)
        logger.info(f"Next prediction task will run at {next_run_time}")
        
        time.sleep(interval_minutes * 60)

def train_new_model():
    """Train a new model interactively."""
    print("Training new model...")
    
    # Get training data from Google Sheets
    try:
        client = load_gsheet_client()
        worksheet = client.open_by_key(SPREADSHEET_ID).worksheet(WORKSHEET_NAME)
        records = worksheet.get_all_records()
        df_train = pd.DataFrame.from_records(records)
        
        if df_train.empty:
            print("No training data found in Google Sheet.")
            return None
        
        df_train = preprocess_gsheet_data(df_train)
        
        # Filter for normal operation data (exclude anomalies if labeled)
        if 'predicted_label' in df_train.columns:
            df_normal = df_train[~df_train['predicted_label'].str.contains('Anomaly', na=False)]
        else:
            df_normal = df_train.copy()
        
        print(f"Training on {len(df_normal)} records...")
        
        # Initialize enhanced detector
        detector = EnhancedFanStateDetector(
            n_clusters=3,
            scaler_type='robust',  # More robust to outliers
            anomaly_method='ocsvm'
        )
        
        # Train the model
        detector.fit(df_normal)
        
        # Save the model
        detector.save_model(MODEL_PATH, SCALER_PATH, DETECTORS_PATH, CONFIG_PATH)
        
        # Display model summary
        summary = detector.get_model_summary()
        print("Model Summary:")
        print(f"  Features used: {len(summary['features_used'])}")
        print(f"  Operational modes: {summary['mode_mapping']}")
        print(f"  Anomaly detection method: {summary['anomaly_method']}")
        
        print("Model training completed and saved!")
        return detector
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None

def main():
    """Main function with enhanced options."""
    print("Enhanced Fan State Predictive Maintenance System")
    print("=" * 50)
    
    # Check if model exists
    model_exists = all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, DETECTORS_PATH, CONFIG_PATH])
    
    if model_exists:
        print("Existing model found.")
        detector = EnhancedFanStateDetector()
        detector.load_model(MODEL_PATH, SCALER_PATH, DETECTORS_PATH, CONFIG_PATH)
        print("Model loaded successfully!")
        
    else:
        # Train new model
        detector = train_new_model()
        if detector is None:
            return
    
    # Start auto prediction every 10 minutes
    interval_minutes = 10  # You can change this value
    print(f"Starting auto-prediction every {interval_minutes} minutes.")
    
    try:
        # Run prediction task in background
        prediction_thread = threading.Thread(
            target=auto_predict, 
            args=(detector, interval_minutes), 
            daemon=True
        )
        prediction_thread.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping prediction system...")
        print("Program terminated.")

if __name__ == "__main__":
    main()

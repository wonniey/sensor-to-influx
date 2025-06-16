import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import joblib
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import logging
from collections import deque

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class EnhancedFanStateDetector:
    def __init__(self, vibration_features=None, n_clusters=3, 
                 scaler_type='standard', anomaly_method='ocsvm',
                 off_state_tolerance=0.2, use_moving_average=True,
                 moving_window_size=5, anomaly_confirmation_count=3):
        """
        Enhanced detector with robust Off state detection.
        
        Args:
            vibration_features: List of vibration-related features
            n_clusters: Number of clusters for operational modes
            scaler_type: 'standard', 'robust', or 'minmax'
            anomaly_method: 'ocsvm' or 'isolation_forest'
            off_state_tolerance: Tolerance for Off state variations (default: 0.2)
            use_moving_average: Whether to apply moving average smoothing
            moving_window_size: Size of moving average window
            anomaly_confirmation_count: Number of consecutive anomalies needed for confirmation
        """
        # Only vibration features
        self.vibration_features = vibration_features or [
            'x_acc', 'x_displacement', 'x_frq', 'x_speed',
            'y_acc', 'y_displacement', 'y_frq', 'y_speed',
            'z_acc', 'z_displacement', 'z_frq', 'z_speed'
        ]
        
        self.all_features = self.vibration_features
        self.n_clusters = n_clusters
        self.anomaly_method = anomaly_method
        
        # New robust detection parameters
        self.off_state_tolerance = off_state_tolerance
        self.use_moving_average = use_moving_average
        self.moving_window_size = moving_window_size
        self.anomaly_confirmation_count = anomaly_confirmation_count
        
        # Initialize scalers
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
            
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.anomaly_detectors = {}
        self.thresholds = {}
        self.fitted = False
        self.feature_importance = {}
        
        # Off state baseline values
        self.off_state_baseline = {}
        self.off_cluster_id = None
        
        # Moving average buffers for real-time processing
        self.moving_buffers = {}
        
        # Anomaly confirmation tracking
        self.anomaly_counters = {}
        
    def _detect_available_features(self, df):
        """Automatically detect which vibration features are available in the dataset."""
        available_vibration = [f for f in self.vibration_features if f in df.columns]
        
        logger.info(f"Available vibration features: {available_vibration}")
        
        return available_vibration
    
    def _apply_moving_average(self, data, feature_name):
        """Apply moving average smoothing to reduce noise."""
        if not self.use_moving_average:
            return data
        
        # Initialize buffer if not exists
        if feature_name not in self.moving_buffers:
            self.moving_buffers[feature_name] = deque(maxlen=self.moving_window_size)
        
        smoothed_data = []
        buffer = self.moving_buffers[feature_name]
        
        for value in data:
            buffer.append(value)
            # Calculate moving average
            smoothed_value = sum(buffer) / len(buffer)
            smoothed_data.append(smoothed_value)
        
        return np.array(smoothed_data)
    
    def _preprocess_data(self, df):
        """Enhanced data preprocessing with robust NaN handling and noise reduction."""
        df_processed = df.copy()
        
        # Get available features first
        available_features = [f for f in self.vibration_features if f in df_processed.columns]
        
        if not available_features:
            logger.warning("No vibration features found in data")
            return df_processed
        
        # Convert to numeric and handle missing values for vibration features only
        for feature in available_features:
            df_processed[feature] = pd.to_numeric(df_processed[feature], errors='coerce')
        
        # Forward and backward fill
        df_processed[available_features] = df_processed[available_features].ffill().bfill()
        
        # Handle remaining NaN values with median (more robust approach)
        for feature in available_features:
            if df_processed[feature].isna().any():
                median_val = df_processed[feature].median()
                if pd.isna(median_val):
                    median_val = 0.0
                    logger.warning(f"Feature {feature} has all NaN values, using 0 as fallback")
                df_processed[feature] = df_processed[feature].fillna(median_val)
        
        # Apply moving average smoothing if enabled
        if self.use_moving_average:
            for feature in available_features:
                df_processed[feature] = self._apply_moving_average(
                    df_processed[feature].values, feature
                )
        
        # Final check - ensure no NaN values remain
        for feature in available_features:
            if df_processed[feature].isna().any():
                logger.warning(f"Still found NaN in {feature}, filling with 0")
                df_processed[feature] = df_processed[feature].fillna(0.0)
        
        # Convert to float to ensure consistency
        df_processed[available_features] = df_processed[available_features].astype(float)
        
        logger.info(f"Preprocessed {len(available_features)} vibration features")
        return df_processed
    
    def _identify_operational_modes(self, X_scaled, cluster_labels):
        """Enhanced mode identification with Off state baseline calculation."""
        cluster_centers = self.kmeans.cluster_centers_
        mode_mapping = {}
        
        # Calculate vibration intensity for each cluster using all available features
        vibration_intensity = np.linalg.norm(cluster_centers, axis=1)
        
        # Off mode: lowest vibration
        off_cluster = np.argmin(vibration_intensity)
        self.off_cluster_id = off_cluster
        
        # Calculate baseline values for Off state
        off_state_data = X_scaled[cluster_labels == off_cluster]
        self.off_state_baseline = {
            'mean': np.mean(off_state_data, axis=0),
            'std': np.std(off_state_data, axis=0),
            'median': np.median(off_state_data, axis=0)
        }
        
        # Sort remaining clusters by intensity
        remaining_clusters = [i for i in range(self.n_clusters) if i != off_cluster]
        remaining_intensities = [(i, vibration_intensity[i]) for i in remaining_clusters]
        remaining_intensities.sort(key=lambda x: x[1])
        
        mode_mapping[off_cluster] = "Off"
        
        for idx, (cluster_id, intensity) in enumerate(remaining_intensities):
            if idx == 0:
                mode_mapping[cluster_id] = "Low"
            elif idx == len(remaining_intensities) - 1:
                mode_mapping[cluster_id] = "High"
            else:
                mode_mapping[cluster_id] = f"Medium_{idx}"
        
        logger.info(f"Identified operational modes: {mode_mapping}")
        logger.info(f"Off state baseline calculated for cluster {off_cluster}")
        
        return mode_mapping
    
    def _is_within_off_state_tolerance(self, sample_scaled):
        """Check if a sample is within acceptable Off state range."""
        if self.off_state_baseline is None or len(self.off_state_baseline) == 0:
            return False
        
        # Calculate distance from Off state baseline
        baseline_mean = self.off_state_baseline['mean']
        
        # Use L2 norm distance
        distance = np.linalg.norm(sample_scaled - baseline_mean)
        
        # Use adaptive tolerance based on Off state standard deviation
        baseline_std = np.mean(self.off_state_baseline['std'])
        adaptive_tolerance = max(self.off_state_tolerance, 2 * baseline_std)
        
        return distance <= adaptive_tolerance
    
    def _confirm_anomaly(self, sample_id, is_anomalous):
        """Confirm anomaly only after consecutive detections."""
        if sample_id not in self.anomaly_counters:
            self.anomaly_counters[sample_id] = 0
        
        if is_anomalous:
            self.anomaly_counters[sample_id] += 1
            return self.anomaly_counters[sample_id] >= self.anomaly_confirmation_count
        else:
            self.anomaly_counters[sample_id] = 0
            return False
    
    def fit(self, df_normal):
        """Fit the model with enhanced preprocessing and mode detection."""
        df_processed = self._preprocess_data(df_normal)
        
        # Use only available vibration features
        available_features = self._detect_available_features(df_processed)
        self.features_used = available_features
        
        if not available_features:
            raise ValueError("No valid vibration features found in the dataset")
        
        X = df_processed[available_features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Cluster the data
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Identify operational modes (this also calculates Off state baseline)
        self.mode_mapping = self._identify_operational_modes(X_scaled, cluster_labels)
        
        # Train anomaly detectors per cluster with adjusted parameters
        for cluster in range(self.n_clusters):
            cluster_data = X_scaled[cluster_labels == cluster]
            if len(cluster_data) == 0:
                logger.warning(f"No data for cluster {cluster}")
                continue
            
            # Adjust contamination and nu parameters for Off state
            if cluster == self.off_cluster_id:
                contamination = 0.02  # Less sensitive for Off state
                nu = 0.02
            else:
                contamination = 0.05  # Normal sensitivity for other states
                nu = 0.05
                
            # Choose anomaly detection method
            if self.anomaly_method == 'isolation_forest':
                detector = IsolationForest(contamination=contamination, random_state=42)
                detector.fit(cluster_data)
                scores = detector.decision_function(cluster_data)
                threshold = np.percentile(scores, contamination * 100)
            else:
                detector = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
                detector.fit(cluster_data)
                scores = detector.decision_function(cluster_data)
                threshold = np.percentile(scores, nu * 100)
            
            self.anomaly_detectors[cluster] = detector
            self.thresholds[cluster] = threshold
            
            logger.info(f"Cluster {cluster} ({self.mode_mapping.get(cluster, 'Unknown')}): "
                       f"threshold={threshold:.3f}, samples={len(cluster_data)}")
        
        self.fitted = True
        logger.info("Model training completed successfully with enhanced Off state detection")
        
    def predict(self, df_test, timestamp_col='time stamp', confidence_threshold=0.5):
        """Enhanced prediction with robust Off state handling."""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        df_test = self._preprocess_data(df_test)
        df_test['timestamp'] = pd.to_datetime(df_test[timestamp_col], errors='coerce')
        
        # Double-check for NaN values before prediction
        X_test = df_test[self.features_used]
        
        # Log info about data quality
        nan_counts = X_test.isna().sum()
        if nan_counts.any():
            logger.warning(f"NaN values found after preprocessing: {nan_counts[nan_counts > 0].to_dict()}")
            X_test = X_test.fillna(0.0)
        
        # Ensure all data is numeric
        X_test = X_test.astype(float)
        
        # Final NaN check
        if X_test.isna().any().any():
            logger.error("Critical: NaN values still present after all preprocessing")
            X_test = X_test.fillna(0.0)
        
        try:
            X_scaled = self.scaler.transform(X_test)
            
            # Predict clusters
            cluster_preds = self.kmeans.predict(X_scaled)
            cluster_distances = self.kmeans.transform(X_scaled)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            n_samples = len(X_test)
            return self._create_default_predictions(df_test, n_samples)
        
        predictions = []
        anomaly_scores = []
        confidence_scores = []
        off_state_flags = []
        
        for i, cluster in enumerate(cluster_preds):
            detector = self.anomaly_detectors.get(cluster)
            threshold = self.thresholds.get(cluster)
            
            if detector is None or threshold is None:
                predictions.append(f"Unknown (Cluster {cluster})")
                anomaly_scores.append(0)
                confidence_scores.append(0)
                off_state_flags.append(False)
                continue
            
            try:
                # Get anomaly score
                score = detector.decision_function([X_scaled[i]])[0]
                
                # Calculate confidence based on distance to cluster center
                min_distance = np.min(cluster_distances[i])
                confidence = 1 / (1 + min_distance)
                
                # Get mode name
                mode_name = self.mode_mapping.get(cluster, f"Mode_{cluster}")
                
                # Enhanced Off state handling
                is_off_state = (cluster == self.off_cluster_id)
                within_off_tolerance = False
                
                if is_off_state:
                    within_off_tolerance = self._is_within_off_state_tolerance(X_scaled[i])
                    off_state_flags.append(True)
                else:
                    off_state_flags.append(False)
                
                # Determine if anomalous
                is_anomalous = score < threshold
                
                # Apply Off state tolerance override
                if is_off_state and within_off_tolerance:
                    # Force as normal if within Off state tolerance
                    predictions.append(f"Normal ({mode_name})")
                    anomaly_scores.append(score)
                elif is_anomalous:
                    # Confirm anomaly with consecutive detection
                    confirmed_anomaly = self._confirm_anomaly(i, True)
                    if confirmed_anomaly:
                        predictions.append("Anomaly")
                    else:
                        predictions.append(f"Potential Anomaly ({mode_name})")
                    anomaly_scores.append(score)
                else:
                    # Reset anomaly counter for normal detection
                    self._confirm_anomaly(i, False)
                    if confidence >= confidence_threshold:
                        predictions.append(f"Normal ({mode_name})")
                    else:
                        predictions.append(f"Normal ({mode_name})")
                    anomaly_scores.append(score)
                
                confidence_scores.append(confidence)
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {str(e)}")
                predictions.append("Error")
                anomaly_scores.append(0)
                confidence_scores.append(0)
                off_state_flags.append(False)
        
        # Add results to dataframe
        df_test['predicted_label'] = predictions
        df_test['degree_of_anomaly'] = anomaly_scores
        df_test['confidence'] = confidence_scores
        df_test['cluster'] = cluster_preds
        df_test['is_off_state'] = off_state_flags
        
        return df_test
    
    def _create_default_predictions(self, df_test, n_samples):
        """Create default predictions when main prediction fails."""
        df_test['predicted_label'] = ['Error'] * n_samples
        df_test['degree_of_anomaly'] = [0.0] * n_samples
        df_test['confidence'] = [0.0] * n_samples
        df_test['cluster'] = [0] * n_samples
        df_test['is_off_state'] = [False] * n_samples
        return df_test
    
    def analyze_feature_importance(self, df):
        """Analyze which vibration features contribute most to cluster separation."""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet.")
        
        X = df[self.features_used]
        X_scaled = self.scaler.transform(X)
        
        # Calculate feature importance based on cluster center separation
        centers = self.kmeans.cluster_centers_
        feature_importance = {}
        
        for i, feature in enumerate(self.features_used):
            # Calculate variance of this feature across cluster centers
            feature_variance = np.var(centers[:, i])
            feature_importance[feature] = feature_variance
        
        # Normalize importance scores
        max_importance = max(feature_importance.values())
        for feature in feature_importance:
            feature_importance[feature] /= max_importance
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def plot_enhanced(self, df_pred, figsize=(15, 12)):
        """Enhanced plotting with Off state analysis."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Enhanced Fan State Analysis - Robust Off State Detection', fontsize=16)
        
        # Color mapping for states
        unique_labels = df_pred['predicted_label'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))
        
        # 1. State timeline
        ax1 = axes[0, 0]
        for i, (label, color) in enumerate(color_map.items()):
            mask = df_pred['predicted_label'] == label
            ax1.scatter(df_pred[mask]['timestamp'], [i] * sum(mask), 
                       c=[color], label=label, alpha=0.7, s=10)
        ax1.set_title('State Timeline')
        ax1.set_ylabel('State')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Anomaly scores over time
        ax2 = axes[0, 1]
        ax2.plot(df_pred['timestamp'], df_pred['degree_of_anomaly'], alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        # Highlight Off state periods
        off_state_mask = df_pred['is_off_state']
        ax2.scatter(df_pred[off_state_mask]['timestamp'], 
                   df_pred[off_state_mask]['degree_of_anomaly'], 
                   c='green', alpha=0.5, s=20, label='Off State')
        ax2.set_title('Anomaly Scores Over Time')
        ax2.set_ylabel('Anomaly Score')
        ax2.legend()
        
        # 3. Off state tolerance visualization
        ax3 = axes[0, 2]
        if 'is_off_state' in df_pred.columns:
            off_data = df_pred[df_pred['is_off_state']]
            normal_off = off_data[off_data['predicted_label'].str.contains('Normal', na=False)]
            anomaly_off = off_data[off_data['predicted_label'].str.contains('Anomaly', na=False)]
            
            ax3.scatter(normal_off['timestamp'], normal_off['degree_of_anomaly'], 
                       c='green', label='Normal Off', alpha=0.7)
            ax3.scatter(anomaly_off['timestamp'], anomaly_off['degree_of_anomaly'], 
                       c='red', label='Anomalous Off', alpha=0.7)
            ax3.set_title('Off State Analysis')
            ax3.set_ylabel('Anomaly Score')
            ax3.legend()
        
        # 4. Vibration pattern (all axes)
        ax4 = axes[1, 0]
        vibration_cols = [col for col in df_pred.columns if 'acc' in col]
        for col in vibration_cols[:3]:  # Limit to first 3 for readability
            if col in df_pred.columns:
                ax4.plot(df_pred['timestamp'], df_pred[col], label=col, alpha=0.7)
        ax4.set_title('Vibration Patterns (Acceleration)')
        ax4.set_ylabel('Acceleration')
        ax4.legend()
        
        # 5. Feature importance
        ax5 = axes[1, 1]
        if self.feature_importance:
            features = list(self.feature_importance.keys())
            importance = list(self.feature_importance.values())
            bars = ax5.barh(features, importance)
            ax5.set_title('Vibration Feature Importance')
            ax5.set_xlabel('Importance Score')
            
            # Color bars by feature type
            for i, bar in enumerate(bars):
                if 'acc' in features[i]:
                    bar.set_color('red')
                elif 'speed' in features[i]:
                    bar.set_color('blue')
                elif 'frq' in features[i]:
                    bar.set_color('green')
                else:
                    bar.set_color('orange')
        
        # 6. Confidence distribution
        ax6 = axes[1, 2]
        ax6.hist(df_pred['confidence'], bins=20, alpha=0.7, edgecolor='black')
        ax6.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
        ax6.set_title('Confidence Score Distribution')
        ax6.set_xlabel('Confidence')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        
        plt.tight_layout()
        return fig
    
    def get_model_summary(self):
        """Return a summary of the model configuration and performance."""
        if not self.fitted:
            return "Model not fitted yet."
        
        summary = {
            'features_used': self.features_used,
            'n_clusters': self.n_clusters,
            'mode_mapping': self.mode_mapping,
            'anomaly_method': self.anomaly_method,
            'feature_importance': self.feature_importance,
            'total_vibration_features': len(self.features_used),
            'off_state_tolerance': self.off_state_tolerance,
            'use_moving_average': self.use_moving_average,
            'moving_window_size': self.moving_window_size,
            'anomaly_confirmation_count': self.anomaly_confirmation_count,
            'off_cluster_id': self.off_cluster_id
        }
        return summary
    
    def save_model(self, model_path='enhanced_model.joblib', scaler_path='enhanced_scaler.joblib',
                   detectors_path='enhanced_detectors.joblib', config_path='enhanced_config.joblib'):
        """Save the enhanced model."""
        joblib.dump(self.kmeans, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            'detectors': self.anomaly_detectors,
            'thresholds': self.thresholds
        }, detectors_path)
        joblib.dump({
            'features_used': self.features_used,
            'mode_mapping': self.mode_mapping,
            'feature_importance': self.feature_importance,
            'anomaly_method': self.anomaly_method,
            'off_state_baseline': self.off_state_baseline,
            'off_cluster_id': self.off_cluster_id,
            'off_state_tolerance': self.off_state_tolerance,
            'use_moving_average': self.use_moving_average,
            'moving_window_size': self.moving_window_size,
            'anomaly_confirmation_count': self.anomaly_confirmation_count
        }, config_path)
    
    def load_model(self, model_path='enhanced_model.joblib', scaler_path='enhanced_scaler.joblib',
                   detectors_path='enhanced_detectors.joblib', config_path='enhanced_config.joblib'):
        """Load the enhanced model."""
        self.kmeans = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        detector_data = joblib.load(detectors_path)
        self.anomaly_detectors = detector_data['detectors']
        self.thresholds = detector_data['thresholds']
        
        config_data = joblib.load(config_path)
        self.features_used = config_data['features_used']
        self.mode_mapping = config_data['mode_mapping']
        self.feature_importance = config_data.get('feature_importance', {})
        self.anomaly_method = config_data.get('anomaly_method', 'ocsvm')
        self.off_state_baseline = config_data.get('off_state_baseline', {})
        self.off_cluster_id = config_data.get('off_cluster_id', None)
        self.off_state_tolerance = config_data.get('off_state_tolerance', 0.2)
        self.use_moving_average = config_data.get('use_moving_average', True)
        self.moving_window_size = config_data.get('moving_window_size', 5)
        self.anomaly_confirmation_count = config_data.get('anomaly_confirmation_count', 3)
        
        self.fitted = True
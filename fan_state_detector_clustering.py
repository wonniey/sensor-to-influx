import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
import shap
import joblib
import matplotlib.pyplot as plt
import warnings

# Suppress pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class FanStateUnsupervisedDetector:
    def __init__(self, features, n_clusters=3):
        self.features = features
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.ocsvms = {}
        self.thresholds = {}
        self.fitted = False
        self.shap_explainer = None
        self.shap_values = None

    def fit(self, df_normal):
        # Only features from normal data (assumed unlabeled)
        X = df_normal[self.features]
        X_scaled = self.scaler.fit_transform(X)

        # Cluster the normal data
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # Train OC-SVM per cluster
        for cluster in range(self.n_clusters):
            cluster_data = X_scaled[cluster_labels == cluster]
            if len(cluster_data) == 0:
                print(f"Warning: No data for cluster {cluster}")
                continue
            ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
            ocsvm.fit(cluster_data)
            scores = ocsvm.decision_function(cluster_data)
            threshold = np.percentile(scores, 5)
            self.ocsvms[cluster] = ocsvm
            self.thresholds[cluster] = threshold

        self.fitted = True
        
    def predict(self, df_test, timestamp_col='time stamp', off_mode_relax_factor=2, noise_threshold=0.1):
        if not self.fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        df_test = df_test.copy()
        df_test = df_test.ffill()  # Forward fill missing values
        df_test = df_test.bfill()  # Backward fill missing values
        df_test['timestamp'] = pd.to_datetime(df_test[timestamp_col], errors='coerce')

        # Convert features to numeric and fill missing
        df_test[self.features] = df_test[self.features].apply(pd.to_numeric, errors='coerce')
        df_test[self.features] = df_test[self.features].ffill().bfill()

        X_test = df_test[self.features]
        X_scaled = self.scaler.transform(X_test)

        # Predict clusters for test data
        cluster_preds = self.kmeans.predict(X_scaled)

        final_labels = []
        anomaly_scores = []  # List to store anomaly scores

        # Map each cluster to a state (e.g., "Off", "Normal", "Anomaly")
        # We assume mode 0 is "Off", mode 1 is "Normal", etc.
        cluster_state_mapping = self.map_clusters_to_states(cluster_preds, X_scaled)

        for i, cluster in enumerate(cluster_preds):
            ocsvm = self.ocsvms.get(cluster)
            threshold = self.thresholds.get(cluster)

            # Apply relaxed threshold for "Off" mode (mode identified dynamically)
            if ocsvm is None or threshold is None:
                final_labels.append(f"Normal (Mode {cluster})")
                anomaly_scores.append(0)  # No anomaly score for normal data
            else:
                score = ocsvm.decision_function([X_scaled[i]])[0]

                # Check if current mode corresponds to "Off"
                if cluster_state_mapping[cluster] == "Off":
                    relaxed_threshold = threshold * off_mode_relax_factor
                    if score >= relaxed_threshold:
                        final_labels.append(f"Normal (Mode {cluster})")
                        anomaly_scores.append(score)
                    else:
                        # If the score is close to 0, it's likely due to noise, so we treat it as normal
                        if abs(score) < noise_threshold:
                            final_labels.append(f"Normal (Mode {cluster})")
                            anomaly_scores.append(score)
                        else:
                            final_labels.append(f"Anomaly")
                            anomaly_scores.append(score)
                else:
                    # For other modes (Normal or Anomaly), use the original threshold
                    if score >= threshold:
                        final_labels.append(f"Normal (Mode {cluster})")
                        anomaly_scores.append(score)  # Positive scores indicate normal data
                    else:
                        final_labels.append(f"Anomaly")
                        anomaly_scores.append(score)  # Negative scores indicate anomalies

        df_test['predicted_label'] = final_labels
        df_test['anomaly_score'] = anomaly_scores  # Add the anomaly score to the dataframe

        # State changes for plotting
        state_map = {'Normal': 0, 'Anomaly': 1}
        df_test['state_numeric'] = df_test['predicted_label'].map(state_map)

        # Create the state_change column (tracks transitions)
        df_test['state_change'] = df_test['state_numeric'].diff().fillna(0) != 0

        return df_test



    def map_clusters_to_states(self, cluster_preds, X_scaled):
        """
        This method dynamically maps each cluster to a state based on its feature characteristics.
        """
        cluster_state_mapping = {}

        # Calculate the mean vibration for each cluster to infer the "Off" state
        cluster_means = self.kmeans.cluster_centers_

        # For example, we can say that the "Off" state is the cluster with the lowest mean vibration across all axes
        cluster_vibrations = np.linalg.norm(cluster_means[:, :3], axis=1)  # Assuming the first 3 features are vibration-based

        # Assign the "Off" mode based on lowest vibration (you can change this logic)
        off_cluster = np.argmin(cluster_vibrations)  # Find the cluster with the lowest vibration
        print("Cluster with lowest vibration (Off state):", off_cluster)  # Debug print

        for cluster in range(self.n_clusters):
            if cluster == off_cluster:
                cluster_state_mapping[cluster] = "Off"
            else:
                cluster_state_mapping[cluster] = f"Normal (mode {cluster})"

        return cluster_state_mapping





    def plot(self, df_pred, vibration_axis='x_acc'):
        unique_labels = df_pred['predicted_label'].unique()
        available_colors = ['green', 'red', 'blue', 'purple', 'brown', 'cyan', 'magenta', 'orange', 'lime', 'teal']
        colors_map = {}

        for i, lbl in enumerate(unique_labels):
            colors_map[lbl] = available_colors[i % len(available_colors)]

        colors = df_pred['predicted_label'].map(colors_map)
        change_points = df_pred[df_pred['state_change']]

        fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})

        # Scatter plot of fan states
        axs[0].scatter(df_pred['timestamp'], [1] * len(df_pred), c=colors, alpha=0.6)
        for _, row in change_points.iterrows():
            ts = row['timestamp']
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            for ax in axs:
                ax.axvline(x=ts, color='gray', linestyle='--', alpha=0.3)
            axs[0].text(ts, 1.05, ts_str, rotation=90, fontsize=7, verticalalignment='bottom')

        axs[0].set_title('Fan State Predictions Over Time')
        axs[0].set_yticks([1])
        axs[0].set_yticklabels(['Fan State'])
        axs[0].legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', label=lbl, markerfacecolor=col, markersize=10)
            for lbl, col in colors_map.items()
        ])
        axs[0].grid(True)

        # Vibration signal line plot
        axs[1].plot(df_pred['timestamp'], df_pred[vibration_axis], label=vibration_axis, color='black')
        axs[1].set_title(f'{vibration_axis} Over Time')
        axs[1].set_ylabel(vibration_axis)
        axs[1].grid(True)
        plt.xlabel('Time')
        plt.tight_layout()

        return fig


    def shap_summary(self, df_pred=None):
        if self.shap_values is None:
            raise RuntimeError("No SHAP values found. Call predict() first.")

        X = df_pred[self.features] if df_pred is not None else None

        # shap_values can be a list (multi-class) or Explanation object
        if isinstance(self.shap_values, list) or (hasattr(self.shap_values, "__len__") and len(self.shap_values) > 1):
            for class_idx, class_name in enumerate(self.class_map):
                shap_vals = self.shap_values[class_idx]
                # sometimes shap_vals shape has an extra last col; remove if so
                if shap_vals.shape[1] == X.shape[1] + 1:
                    shap_vals = shap_vals[:, :-1]
                shap.summary_plot(
                    shap_vals,
                    X,
                    class_names=[class_name],
                    plot_type='bar',
                    show=(class_idx == len(self.class_map) - 1)  # Show on last plot
                )
        else:
            shap.summary_plot(self.shap_values, X)


    def save_model(self, model_path='kmeans_model.joblib', scaler_path='scaler.joblib',
                ocsvms_path='ocsvms.joblib', thresholds_path='thresholds.joblib'):
        import joblib
        joblib.dump(self.kmeans, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.ocsvms, ocsvms_path)
        joblib.dump(self.thresholds, thresholds_path)

    def load_model(self, model_path='kmeans_model.joblib', scaler_path='scaler.joblib',
                ocsvms_path='ocsvms.joblib', thresholds_path='thresholds.joblib'):
        import joblib
        self.kmeans = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.ocsvms = joblib.load(ocsvms_path)
        self.thresholds = joblib.load(thresholds_path)
        self.fitted = True

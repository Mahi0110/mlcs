import tensorflow as tf
from scapy.all import sniff, IP
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings

# Suppress warning messages for cleaner output
warnings.filterwarnings('ignore')

# --- Recreate the Preprocessor ---
# This part MUST match the preprocessor from your training script
print("Loading preprocessor structure from training data...")
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]
df_placeholder = pd.read_csv('KDDTest+.txt', header=None, names=columns)
features_placeholder = df_placeholder.drop('label', axis=1)
if 'num_outbound_cmds' not in features_placeholder.columns:
    features_placeholder['num_outbound_cmds'] = 0

categorical_features = ['protocol_type', 'service', 'flag']
numerical_features = features_placeholder.drop(categorical_features, axis=1).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
preprocessor.fit(features_placeholder)
print("Preprocessor is ready.")

# 1. Load the trained model
print("Loading trained autoencoder model...")
model = tf.keras.models.load_model('autoencoder_model.keras')

# 2. Set your ANOMALY_THRESHOLD
# IMPORTANT: Replace 0.001156 with the actual threshold value from your training script
ANOMALY_THRESHOLD = 0.0003234863847918426

print(f"Anomaly threshold set to: {ANOMALY_THRESHOLD}")
print("\nStarting real-time network sniffing...")
print("---------------------------------------")

def packet_callback(packet):
    if IP in packet:
        try:
            proto_map = {1: 'icmp', 6: 'tcp', 17: 'udp'}
            protocol_type = proto_map.get(packet[IP].proto, 'other')

            # This is a simplified feature extraction for demonstration
            packet_features = {
                'duration': 0, 'protocol_type': protocol_type, 'service': 'http', 'flag': 'SF',
                'src_bytes': len(packet), 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
                'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0,
                'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
                'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 'count': 1, 'srv_count': 1,
                'serror_rate': 0, 'srv_serror_rate': 0, 'rerror_rate': 0, 'srv_rerror_rate': 0,
                'same_srv_rate': 1, 'diff_srv_rate': 0, 'srv_diff_host_rate': 0, 'dst_host_count': 1,
                'dst_host_srv_count': 1, 'dst_host_same_srv_rate': 1, 'dst_host_diff_srv_rate': 0,
                'dst_host_same_src_port_rate': 1, 'dst_host_srv_diff_host_rate': 0, 'dst_host_serror_rate': 0,
                'dst_host_srv_serror_rate': 0, 'dst_host_rerror_rate': 0, 'dst_host_srv_rerror_rate': 0
            }

            live_packet_df = pd.DataFrame([packet_features], columns=features_placeholder.columns)
            live_packet_processed = preprocessor.transform(live_packet_df).toarray()
            reconstruction = model.predict(live_packet_processed, verbose=0)
            loss = np.mean(np.square(reconstruction - live_packet_processed))

            if loss > ANOMALY_THRESHOLD:
                print(f"🚨 ANOMALY DETECTED! Loss: {loss:.6f} | SRC: {packet[IP].src} -> DST: {packet[IP].dst} | PROTO: {protocol_type.upper()}")

        except Exception as e:
            pass

# Find the correct interface name (e.g., 'enp0s3' or 'eth0')
# Use 'ip a' in your terminal to find the name of your Host-Only adapter
sniff(iface='enp0s3', prn=packet_callback, store=0)

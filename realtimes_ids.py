# realtime_ids.py
from scapy.all import sniff, IP
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load artifacts
model = tf.keras.models.load_model("autoencoder_model.keras")
preprocessor = joblib.load("preprocessor.pkl")
threshold = float(np.load("anomaly_threshold.npy"))

# Feature placeholder
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
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

def packet_callback(packet):
    if IP in packet:
        proto_map = {1: 'icmp', 6: 'tcp', 17: 'udp'}
        proto = proto_map.get(packet[IP].proto, 'other')

        features = {
            'duration': 0,
            'protocol_type': proto,
            'service': 'http',
            'flag': 'SF',
            'src_bytes': len(packet),
            'dst_bytes': 0,
            'land': 0, 'wrong_fragment': 0, 'urgent': 0,
            'hot': 0, 'num_failed_logins': 0, 'logged_in': 0,
            'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
            'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
            'num_access_files': 0, 'num_outbound_cmds': 0,
            'is_host_login': 0, 'is_guest_login': 0,
            'count': 1, 'srv_count': 1,
            'serror_rate': 0, 'srv_serror_rate': 0, 'rerror_rate': 0,
            'srv_rerror_rate': 0, 'same_srv_rate': 1, 'diff_srv_rate': 0,
            'srv_diff_host_rate': 0, 'dst_host_count': 1,
            'dst_host_srv_count': 1, 'dst_host_same_srv_rate': 1,
            'dst_host_diff_srv_rate': 0,
            'dst_host_same_src_port_rate': 1, 'dst_host_srv_diff_host_rate': 0,
            'dst_host_serror_rate': 0, 'dst_host_srv_serror_rate': 0,
            'dst_host_rerror_rate': 0, 'dst_host_srv_rerror_rate': 0
        }

        df = pd.DataFrame([features])
        transformed = preprocessor.transform(df).toarray()
        reconstructed = model.predict(transformed, verbose=0)
        loss = np.mean(np.square(transformed - reconstructed))

        if loss > threshold:
            print(f"🚨 ANOMALY DETECTED: {packet[IP].src} → {packet[IP].dst} | Loss: {loss:.6f}")
        else:
            print(f"✅ Normal packet: {packet[IP].src} → {packet[IP].dst}")

print("[*] Sniffing packets...")
sniff(iface="enp0s3", prn=packet_callback, store=0)

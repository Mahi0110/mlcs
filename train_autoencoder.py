# train_autoencoder.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load KDDCup99 dataset
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

df = pd.read_csv("KDDTrain+.txt", header=None, names=columns)
df = df[df['label'] == 'normal']  # Train only on normal traffic

# Handle missing column
if 'num_outbound_cmds' not in df.columns:
    df['num_outbound_cmds'] = 0

# Preprocessing
categorical = ['protocol_type', 'service', 'flag']
numerical = df.drop(['label'] + categorical, axis=1).columns

preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

X = preprocessor.fit_transform(df.drop('label', axis=1))

# Save preprocessor
joblib.dump(preprocessor, "preprocessor.pkl")

# Autoencoder
input_dim = X.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(
    X, X,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=2)]
)

# Save model
autoencoder.save("autoencoder_model.keras")

# Save threshold
reconstruction = autoencoder.predict(X)
loss = np.mean(np.square(X - reconstruction), axis=1)
threshold = np.max(loss)
np.save("anomaly_threshold.npy", threshold)

print(f"[✔] Training complete. Anomaly threshold: {threshold}")

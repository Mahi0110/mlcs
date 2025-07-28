import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
import numpy as np

# 1. Load and Prepare Data
print("Loading data...")
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
df = pd.read_csv('KDDTest+.txt', header=None, names=columns)
# --- ADD THIS DATA CLEANING SECTION ---

# Clean 'Response Time' by removing ' ms' and converting to a number
if 'Response Time' in df.columns:
    df['Response Time'] = df['Response Time'].str.replace(' ms', '').astype(float)

# Clean 'Data Transfer Rate' by removing ' Mbps' and converting to a number
if 'Data Transfer Rate' in df.columns:
    df['Data Transfer Rate'] = df['Data Transfer Rate'].str.replace(' Mbps', '').astype(float)

# --- END OF DATA CLEANING SECTION ---


# Now, redefine your features and labels with the cleaned data
features = df.drop('label', axis=1) # Or whatever your label column is named
# ... the rest of your preprocessing script follows

if 'num_outbound_cmds' not in df.columns:
    df['num_outbound_cmds'] = 0

features = df.drop('label', axis=1)
labels = df['label']

categorical_features = ['protocol_type', 'service', 'flag']
numerical_features = features.drop(categorical_features, axis=1).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X = preprocessor.fit_transform(features).toarray()
X_normal = X[labels == 'normal']
print(f"Data prepared. Normal samples: {X_normal.shape[0]}")

# 2. Build the Autoencoder Model
print("Building autoencoder model...")
input_dim = X_normal.shape[1]
encoding_dim = 16

autoencoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(encoding_dim, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

# 3. Train the Model
print("Training model on NORMAL data only...")
autoencoder.fit(X_normal, X_normal, epochs=20, batch_size=32, shuffle=True, validation_split=0.2)

# 4. Save the Model
print("Saving the trained model...")
autoencoder.save('autoencoder_model.h5')
print("Model saved as autoencoder_model.h5")

# 5. Determine and print the Anomaly Threshold
reconstructions = autoencoder.predict(X_normal)
train_loss = tf.keras.losses.mse(reconstructions, X_normal)
threshold = np.quantile(train_loss, 0.95)
print(f"\nAnomaly detection threshold (95th percentile of normal loss): {threshold}")

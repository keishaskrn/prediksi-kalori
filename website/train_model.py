import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv('data/train.csv')

# ambil kolom penting
df = df[['Store', 'DayOfWeek', 'Promo', 'Customers', 'Sales']]

# hapus data kosong
df.dropna(inplace=True)

# =========================
# FITUR DAN TARGET
# =========================

X = df[['Store', 'DayOfWeek', 'Promo', 'Customers']]
y = df['Sales']

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# simpan scaler
joblib.dump(
    scaler,
    'models/scaler.pkl'
)

# =========================
# LINEAR REGRESSION
# =========================

model = LinearRegression()

model.fit(X_train, y_train)

# simpan model
joblib.dump(
    model,
    'models/linear_regression.pkl'
)

print("MODEL BERHASIL DISIMPAN!")
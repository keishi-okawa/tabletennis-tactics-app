import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
import os

def train_model(csv_path, player_name):
    df = pd.read_csv(csv_path)
    features = ["サーブ種類", "サーブコース", "サーブ速度", "サーブ長さ"]
    targets = ["レシーブ方法", "レシーブコース", "レシーブ長さ", "最終得点"]

    encoders = {}
    for col in features + targets:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[features]
    y = df[targets]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{player_name}_model.pkl")
    joblib.dump(encoders, f"models/{player_name}_encoders.pkl")
    print(f"✅ モデル保存完了: {player_name}")

if __name__ == "__main__":
    player_name = sys.argv[1]  # 例: katayama
    csv_path = f"data/{player_name}.csv"
    train_model(csv_path, player_name)

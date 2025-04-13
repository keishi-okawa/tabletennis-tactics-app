import streamlit as st
import joblib
import pandas as pd
import os

st.title("🏓 卓球戦術アドバイザー 🎯")

# モデルファイル一覧を取得
models = [f.replace("_model.pkl", "") for f in os.listdir("models") if f.endswith("_model.pkl")]

# モデルが見つからなければ中止
if not models:
    st.error("モデルファイルが見つかりません。`models/` フォルダに `.pkl` ファイルを追加してください。")
    st.stop()

# 対戦相手の選択
opponent = st.selectbox("対戦相手を選択", models)

# モデルとエンコーダの読み込み
model_path = f"models/{opponent}_model.pkl"
encoder_path = f"models/{opponent}_encoders.pkl"

model = joblib.load(model_path)
encoders = joblib.load(encoder_path)

# ユーザー入力フォーム
def user_input():
    serve_type = st.selectbox("サーブ種類", encoders["サーブ種類"].classes_)
    serve_course = st.selectbox("サーブコース", encoders["サーブコース"].classes_)
    serve_speed = st.selectbox("サーブ速度", encoders["サーブ速度"].classes_)
    serve_length = st.selectbox("サーブ長さ", encoders["サーブ長さ"].classes_)
    return {
        "サーブ種類": serve_type,
        "サーブコース": serve_course,
        "サーブ速度": serve_speed,
        "サーブ長さ": serve_length
    }

input_data = user_input()

# 戦術提案
if st.button("戦術を提案"):
    input_df = pd.DataFrame([[ 
        encoders["サーブ種類"].transform([input_data["サーブ種類"]])[0],
        encoders["サーブコース"].transform([input_data["サーブコース"]])[0],
        encoders["サーブ速度"].transform([input_data["サーブ速度"]])[0],
        encoders["サーブ長さ"].transform([input_data["サーブ長さ"]])[0]
    ]], columns=["サーブ種類", "サーブコース", "サーブ速度", "サーブ長さ"])

    prediction = model.predict(input_df)[0]
    result = {
        key: encoders[key].inverse_transform([prediction[i]])[0] 
        for i, key in enumerate(["レシーブ方法", "レシーブコース", "レシーブ長さ", "最終得点"])
    }

    st.success("### 推奨レシーブ戦術")
    st.table(pd.DataFrame([result]))

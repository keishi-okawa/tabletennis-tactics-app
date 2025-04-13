# 卓球戦術アドバイザー 🏓

このアプリは、過去の試合データに基づいて戦術を提案する卓球用のStreamlitアプリです。

## 機能
- 対戦相手ごとに専用の学習モデルを使ってレシーブ戦術を提案
- UIからサーブ条件を入力して即座にフィードバックを取得

## 使い方
1. `data/` にCSVデータを入れる（例: `katayama.csv`）
2. 以下でモデルを作成：
   ```bash
   python utils/train_model.py katayama
   ```
3. Streamlitでアプリを実行：
   ```bash
   streamlit run streamlit_app.py
   ```

## フォルダ構成
```
tabletennis-tactics-app/
├── streamlit_app.py
├── requirements.txt
├── models/  ← モデル出力場所
├── data/    ← 入力CSVデータ
├── utils/
│   └── train_model.py
├── .gitignore
└── README.md

# dashboard/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from io import BytesIO
from datetime import datetime, timedelta

# ⚙️ Cấu hình
st.set_page_config(layout="wide")
st.title("📊 Phân tích cảm xúc livestream theo thời gian")

DATA_PATH = "data/raw/live_sentiment.csv"
REFRESH_SEC = 5

# 🎨 Định nghĩa style cho từng cảm xúc
style_dict = {
    "negative": {"color": "orange", "linestyle": "-", "marker": "o"},
    "neutral": {"color": "darkorange", "linestyle": "--", "marker": "s"},
    "positive": {"color": "deeppink", "linestyle": ":", "marker": "D"},
}

# 📈 Hàm vẽ biểu đồ
def plot_chart(df, group_freq='1min', title="Biểu đồ cảm xúc"):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["rounded"] = df["timestamp"].dt.floor(group_freq)
    df["time_label"] = df["rounded"].dt.strftime("%H:%M")

    chart_data = df.groupby(["time_label", "sentiment"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    for sentiment in chart_data.columns:
        style = style_dict.get(sentiment, {})
        ax.plot(
            chart_data.index,
            chart_data[sentiment],
            label=sentiment,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"]
        )

    step = max(1, len(chart_data.index) // 10)
    ax.set_xticks(chart_data.index[::step])
    ax.set_xticklabels(chart_data.index[::step], rotation=45, ha='right')

    ax.set_title(title)
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Số lượng bình luận")
    ax.legend(title="Cảm xúc")
    ax.grid(True)
    return fig

# 📤 Hàm tạo link tải ảnh
def get_image_download_link(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer

# ========== Bắt đầu vòng lặp ==========

placeholder = st.empty()

while True:
    if not os.path.exists(DATA_PATH):
        st.warning("⚠️ Chưa có dữ liệu. Hãy chạy real_time_pipeline.py.")
        time.sleep(REFRESH_SEC)
        continue

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        st.warning("📭 Chưa có bình luận nào...")
        time.sleep(REFRESH_SEC)
        continue

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    with placeholder.container():
        col1, col2 = st.columns(2)

        # 📡 Biểu đồ trực tiếp: 5 phút gần nhất, mỗi phút 1 mốc
        with col1:
            recent_df = df[df["timestamp"] > datetime.utcnow() - timedelta(minutes=5)]
            st.subheader("🛰️ Biểu đồ trực tiếp (5 phút gần nhất)")
            fig1 = plot_chart(recent_df, group_freq='1min', title="Cảm xúc từng phút (Live)")
            st.pyplot(fig1)

        # 📊 Biểu đồ tổng hợp: toàn bộ dữ liệu, mỗi 5 phút 1 mốc
        with col2:
            st.subheader("📚 Biểu đồ tổng hợp (toàn bộ)")
            fig2 = plot_chart(df, group_freq='5min', title="Cảm xúc theo mốc 5 phút")
            st.pyplot(fig2)

    time.sleep(REFRESH_SEC)

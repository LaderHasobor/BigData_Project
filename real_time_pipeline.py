# real_time_pipeline.py

import os
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from googleapiclient.discovery import build

from utils.text_utils import predict_sentiment  # hàm gán nhãn

# 🔐 Load API key từ .env
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("❌ Không tìm thấy API KEY trong file .env")

# ====== Hàm lấy liveChatId từ video livestream ======
def get_live_chat_id(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    response = youtube.videos().list(
        part="liveStreamingDetails",
        id=video_id
    ).execute()

    items = response.get("items", [])
    if not items:
        raise ValueError("Không tìm thấy video livestream.")

    return items[0]["liveStreamingDetails"].get("activeLiveChatId")

# ====== Hàm đọc bình luận theo thời gian thực ======
def stream_live_comments(video_id, duration_seconds=120, output_path="data/raw/live_sentiment.csv"):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    live_chat_id = get_live_chat_id(video_id)
    print(f"🎯 LiveChat ID: {live_chat_id}")

    next_page_token = None
    start_time = time.time()

    print("📡 Bắt đầu thu thập bình luận...\n")

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Nếu chưa có file, ghi header
    if not os.path.exists(output_path):
        pd.DataFrame(columns=["timestamp", "author", "text", "sentiment"]).to_csv(output_path, index=False)

    while time.time() - start_time < duration_seconds:
        comments = []

        response = youtube.liveChatMessages().list(
            liveChatId=live_chat_id,
            part="snippet,authorDetails",
            pageToken=next_page_token,
            maxResults=200
        ).execute()

        for item in response.get("items", []):
            msg = item["snippet"]["displayMessage"]
            author = item["authorDetails"]["displayName"]
            timestamp_local = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            sentiment = predict_sentiment(msg)
            comments.append({
                "timestamp": timestamp_local,
                "author": author,
                "text": msg,
                "sentiment": sentiment
            })

            print(f"[{timestamp_local}] ({sentiment.upper()}) {author}: {msg}")

        # Ghi đợt comment này vào file
        if comments:
            df = pd.DataFrame(comments)
            df.to_csv(output_path, mode="a", index=False, header=False)

        next_page_token = response.get("nextPageToken")
        polling_interval = int(response.get("pollingIntervalMillis", 2000)) / 1000
        time.sleep(polling_interval)

    print(f"\n✅ Đã hoàn tất thu thập trong {duration_seconds} giây.")
    print(f"📁 Dữ liệu lưu tại: {output_path}")

# ========== Chạy ==========
if __name__ == "__main__":
    video_id = input("📺 Nhập VIDEO_ID của livestream (ví dụ: dQw4w9WgXcQ): ").strip()
    duration = int(input("⏱️ Thời gian thu thập (giây): ").strip())
    stream_live_comments(video_id, duration_seconds=duration)
# This script collects live comments from a YouTube livestream using the YouTube Data API.
# It retrieves comments in real-time, labels their sentiment, and saves them to a CSV file.
# The script requires a valid YouTube API key stored in a .env file.    
# Make sure to install the required packages:
# pip install google-api-python-client python-dotenv pandas
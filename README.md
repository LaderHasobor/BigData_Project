# BigData_Project
Cấu trúc thư mục
BigData_CommentsAnalysis/
├── real_time_pipeline.py       # Thu thập + gán nhãn bình luận
├── utils/
│   └── text_utils.py           # predict_sentiment() dùng mô hình mới
├── dashboard/
│   └── app.py                  # Dashboard hiển thị biểu đồ
├── models/
│   └── sentiment_model/        # Đầu ra của mô hình đã huấn luyện
│   └── train_sentiment_model.py    # Fine-tune PhoBERT (huấn luyện mô hình)
│   └── *.csv                   # File csv để train
├── data/raw/live_sentiment.csv # Dữ liệu đầu ra (auto tạo)
└── .env        


📊 YouTube Livestream Sentiment Analysis (PhoBERT-based, Real-time)
Phân tích cảm xúc bình luận trong livestream YouTube theo thời gian thực, sử dụng mô hình PhoBERT đã fine-tune trên dữ liệu thực tế.


🚀 Chức năng chính
📥 Thu thập bình luận từ livestream YouTube bằng YouTube API
🤖 Gán nhãn cảm xúc (positive, negative) bằng mô hình PhoBERT đã huấn luyện
📈 Hiển thị biểu đồ cảm xúc trực tiếp & tổng hợp
💾 Ghi dữ liệu bình luận theo thời gian vào .csv để lưu trữ hoặc phân tích thêm


🧱 Kiến trúc hệ thống
[ YouTube Livestream ] → [ real_time_pipeline.py ]
                          ↓
        [ PhoBERT sentiment model (fine-tuned) ]
                          ↓
           [ data/raw/live_sentiment.csv ]
                          ↓
           [ Streamlit dashboard (app.py) ]


📦 Huấn luyện mô hình
FIle: train_sentiment_model.py dùng để huấn luyện mô hình
Đảm bảo bạn đã chuẩn bị file train.csv với 2 cột text,label


▶️ Chạy hệ thống real-time
Thu thập & gán nhãn bình luận:
python real_time_pipeline.py

Nhập VIDEO_ID và thời gian thu thập (tính bằng giây)

Hiển thị dashboard biểu đồ cảm xúc:
streamlit run dashboard/app.py

📊 Dashboard gồm 2 biểu đồ:
🛰️ Trực tiếp: 5 phút gần nhất (hiển thị theo phút)
📚 Tổng hợp: toàn bộ dữ liệu (hiển thị theo mốc 5 phút)

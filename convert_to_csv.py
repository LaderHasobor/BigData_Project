import os
import pandas as pd
import chardet

def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read(20000)
        result = chardet.detect(raw_data)
        return result['encoding']

def load_comments_from_folder(folder_path, label):
    data = []
    total = 0
    print(f"📁 Đang xử lý thư mục: {folder_path} (label: {label})")

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            encoding = detect_encoding(file_path)
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                content = f.read().strip()
                if content:
                    data.append({"text": content, "label": label})
                    total += 1
        except Exception as e:
            print(f"⚠️ Lỗi đọc {file_path}: {e}")

    print(f"✅ Đã đọc được {total} bình luận từ {folder_path}")
    return data

def convert_to_csv(pos_dir, neg_dir, output_path):
    pos_data = load_comments_from_folder(pos_dir, "positive")
    neg_data = load_comments_from_folder(neg_dir, "negative")

    all_data = pos_data + neg_data
    df = pd.DataFrame(all_data)

    # Ghi file với encoding utf-8-sig để Excel đọc đúng tiếng Việt
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Đã lưu {len(df)} dòng dữ liệu vào file: {output_path}")

if __name__ == "__main__":
    # 👉 Thay đổi đường dẫn nếu cần
    base_dir = "C:/Users/lader/Downloads/data_test/data_test/test"
    pos_dir = os.path.join(base_dir, "pos")
    neg_dir = os.path.join(base_dir, "neg")
    output_csv = "train.csv"

    convert_to_csv(pos_dir, neg_dir, output_csv)
    print("🎉 Hoàn tất chuyển đổi!")

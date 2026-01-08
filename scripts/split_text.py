# File: scripts/split_text.py
import os
import json
import re
from tqdm import tqdm

# --- SỬA LỖI IMPORT Ở ĐÂY ---
# Thay đổi từ 'langchain.text_splitter' sang 'langchain_text_splitters'
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback cho phiên bản cũ (nếu có)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
# ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
CHUNK_DIR = os.path.join(BASE_DIR, "data", "chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

def split_by_article(text):
    """
    Chia văn bản theo cấu trúc 'Điều <số>'.
    Sử dụng Lookahead Regex để giữ lại chữ 'Điều' ở đầu mỗi chunk.
    """
    # Pattern cải tiến: Bắt "Điều" ở đầu dòng HOẶC đầu file (^)
    # Thêm flags=re.IGNORECASE để bắt cả "điều 1", "ĐIỀU 1"
    pattern = r'(?=(?:^|\n)Điều \d+)'

    chunks = re.split(pattern, text, flags=re.IGNORECASE)

    # Làm sạch
    valid_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 30: # Chỉ lấy chunk có nội dung thực sự
            valid_chunks.append(chunk)

    return valid_chunks

def main():
    print("✂️  Đang chia nhỏ văn bản (Refactored)...")

    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".txt")]

    if not files:
        print("⚠️ Không tìm thấy file .txt nào trong data/cleaned/. Hãy chạy extract_pdf.py trước.")
        return

    # Khởi tạo Splitter dự phòng một lần duy nhất
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    for filename in tqdm(files):
        file_path = os.path.join(CLEAN_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # 1. Thử chia theo Điều luật
            raw_chunks = split_by_article(text)

            # 2. Fallback: Nếu không tìm thấy "Điều" nào, chia theo ký tự
            if len(raw_chunks) < 2:
                raw_chunks = fallback_splitter.split_text(text)

            # 3. Đóng gói thành Object (Quan trọng cho RAG)
            # Giúp lưu kèm tên file nguồn vào từng chunk
            final_chunks = []
            source_name = filename.replace("_clean.txt", ".pdf") # Tên file gốc

            for content in raw_chunks:
                final_chunks.append({
                    "page_content": content,
                    "metadata": {"source": source_name}
                })

            # 4. Lưu file
            out_name = filename.replace("_clean.txt", "_chunks.json")
            out_path = os.path.join(CHUNK_DIR, out_name)

            with open(out_path, "w", encoding="utf-8") as out:
                json.dump(final_chunks, out, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý file {filename}: {e}")

    print(f"✅ Đã xử lý xong {len(files)} file văn bản.")

if __name__ == "__main__":
    main()

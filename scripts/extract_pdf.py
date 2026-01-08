# File: scripts/extract_pdf.py
import os
import re
import unicodedata
from tqdm import tqdm
import pdfplumber  # <--- Thay thế PyPDF2 để đọc tiếng Việt chuẩn hơn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

def extract_text_from_pdf(pdf_path):
    """
    Dùng pdfplumber để trích xuất văn bản, giữ bố cục tốt hơn.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # extract_text() của pdfplumber thông minh hơn PyPDF2
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        print(f"⚠️ Lỗi đọc file PDF {os.path.basename(pdf_path)}: {e}")
    return text

def clean_text(text):
    """
    Làm sạch văn bản luật chuyên sâu.
    """
    if not text: return ""

    # 1. Chuẩn hóa Unicode (Rất quan trọng với tiếng Việt)
    # Chuyển các ký tự tổ hợp về dựng sẵn (NFC)
    text = unicodedata.normalize('NFC', text)

    # 2. Xóa các dòng tiêu đề/footer rác thường gặp trong văn bản luật
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()

        # Bỏ qua dòng số trang (Ví dụ: "Trang 1", "Page 5/10")
        if re.match(r'^(Trang|Page)\s*\d+(\/\d+)?$', line, re.IGNORECASE):
            continue
        # Bỏ qua dòng chỉ có số (số trang đứng một mình)
        if re.match(r'^\d+$', line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 3. Gộp các dòng bị ngắt quãng vô lý (tùy chọn, nhưng tốt cho RAG)
    # Xử lý: "Cộng hòa xã \n hội chủ nghĩa" -> "Cộng hòa xã hội chủ nghĩa"
    # (Phần này regex phức tạp, tạm thời để đơn giản là xóa dòng thừa)
    text = re.sub(r'\n{3,}', '\n\n', text) # Tối đa 2 dòng trống liên tiếp

    return text.strip()

if __name__ == "__main__":
    os.makedirs(CLEAN_DIR, exist_ok=True)

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]

    if not files:
        print("⚠️ Không tìm thấy file PDF nào trong data/raw/")
    else:
        print(f"🚀 Đang xử lý {len(files)} file PDF với pdfplumber...")

        for filename in tqdm(files):
            pdf_path = os.path.join(RAW_DIR, filename)
            txt_name = filename.replace(".pdf", "_clean.txt")
            txt_path = os.path.join(CLEAN_DIR, txt_name)

            raw_text = extract_text_from_pdf(pdf_path)
            clean = clean_text(raw_text)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(clean)

        print("✅ Đã xử lý xong tất cả file PDF!")

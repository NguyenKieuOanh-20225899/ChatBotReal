import re

# Nếu bạn có cài pyvi hoặc underthesea thì import ở đây. 
# Ví dụ: from pyvi import ViTokenizer
# Hiện tại mình viết hàm regex cải tiến hơn split() một chút để bạn không phải cài thêm lib nặng.

def preprocess_text(text: str) -> str:
    """Chuẩn hóa văn bản cơ bản"""
    if not text: return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text) # Xóa khoảng trắng thừa
    return text

def tokenize_vn(text: str):
    """
    Tokenize cho tiếng Việt.
    Tốt nhất nên dùng: ViTokenizer.tokenize(text).split()
    Ở đây dùng simple split sau khi preprocess để demo.
    """
    text = preprocess_text(text)
    # TODO: Khuyên dùng thư viện tách từ chuyên dụng như pyvi/underthesea cho kết quả tốt nhất
    # return ViTokenizer.tokenize(text).split() 
    return text.split() 

def get_meta_id(meta: dict) -> str:
    """Lấy ID định danh cho chunk để tính toán metrics"""
    return meta.get("stable_id") or meta.get("chunk_id")
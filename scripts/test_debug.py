import re

print("=== PHẦN 1: TÁI HIỆN LỖI SẬP CHƯƠNG TRÌNH (LOGIC LOOP) ===\n")

# Giả lập danh sách file trong thư mục của bạn
# Có file .txt (hợp lệ) và file .png (gây lỗi nếu code sai)
mock_files = ["luat_so_1.txt", "image_loi.png", "luat_so_2.txt"]

print(f"Danh sách file giả lập: {mock_files}")

try:
    print("\n--> Đang chạy thử logic CŨ (bị lỗi)...")
    for filename in mock_files:
        # LOGIC CŨ CỦA BẠN (Lỗi thụt lề)
        if filename.endswith(".txt"):
            # Giả vờ đọc file và xử lý
            content = "Nội dung file..."
            parts = ["Điều 1...", "Điều 2..."]
            # Biến 'parts' chỉ được tạo ra KHI VÀ CHỈ KHI file là .txt

        # [LỖI Ở ĐÂY]: Dòng này nằm NGOÀI 'if', nên nó chạy cho CẢ file .png
        # Khi gặp 'image_loi.png', khối if ở trên bị bỏ qua -> 'parts' chưa được định nghĩa
        count = len(parts)
        print(f" - Xử lý xong {filename}: {count} phần")

except Exception as e:
    print(f"\n[BẮT ĐƯỢC LỖI]: {type(e).__name__}: {e}")
    print("=> GIẢI THÍCH: Khi vòng lặp chạy đến file 'image_loi.png', nó không chui vào 'if', nên biến 'parts' không tồn tại. Nhưng code bên dưới vẫn cố gọi 'parts' -> Gây sập.")

print("\n" + "="*50 + "\n")

print("=== PHẦN 2: KIỂM TRA LỖI CẮT DỮ LIỆU (REGEX) ===\n")

# Giả lập nội dung file luật (có xuống dòng lung tung và nhắc đến điều luật khác)
sample_text = """
Điều 1. Phạm vi điều chỉnh
Luật này quy định về ABC...

Điều 2. Đối tượng áp dụng
Theo quy định tại Điều 1 của Luật này thì... (đây là chỗ gây lỗi cũ)
1. Công dân Việt Nam.
2. Người nước ngoài.

Điều 3. Nguyên tắc
"""

print(f"--- Văn bản mẫu ---\n{sample_text}\n-------------------")

# --- HÀM CŨ (Mô phỏng) ---
print("\n--> Kết quả với Regex CŨ:")
old_pattern = r"(Điều\s+\d+[\.\:\-]?\s*.*?)(?=(Điều\s+\d+[\.\:\-]?\s)|$)"
# Regex cũ dùng lookahead (?=...) nhưng không bắt buộc xuống dòng
old_parts = re.findall(old_pattern, sample_text, flags=re.DOTALL)
print(f"Số phần tách được: {len(old_parts)}")
for i, p in enumerate(old_parts):
    # Logic cũ trả về tuple, lấy phần tử đầu
    text_content = p[0].strip() if isinstance(p, tuple) else p.strip()
    print(f"  [Phần {i+1}]: {text_content[:50]}...")
    if "Theo quy định tại" in text_content:
        print("    -> LỖI: Bị cắt cụt ngay sau chữ 'Điều 1' trong câu văn!")

# --- HÀM MỚI (Đã sửa) ---
print("\n--> Kết quả với Regex MỚI (Fixed):")
# Regex mới: Bắt buộc phải có xuống dòng (\n) trước chữ Điều
new_pattern = r"(?:^|\n)(Điều\s+\d+[\.\:\-]?\s*.*?)(?=(?:\nĐiều\s+\d+[\.\:\-]?\s)|$)"
new_parts = re.findall(new_pattern, sample_text.strip(), flags=re.DOTALL)

print(f"Số phần tách được: {len(new_parts)}")
for i, p in enumerate(new_parts):
    print(f"  [Phần {i+1}]: {p.strip()[:50]}...")

print("\n=> KẾT LUẬN: Regex mới bỏ qua được chữ 'Điều 1' nằm giữa câu, giữ nguyên vẹn nội dung Điều 2.")
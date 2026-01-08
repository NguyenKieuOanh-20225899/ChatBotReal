import os
import shutil
import subprocess
import time
import sys

# Định nghĩa các thư mục dữ liệu cần dọn dẹp
# Lưu ý: Không xóa 'data/raw' vì chứa file gốc
DIRS_TO_CLEAN = [
    "data/cleaned",
    "data/chunks",
    "data/vector_db",
    "data/artifacts"
]

FILES_TO_REMOVE = [
    "data/knowledge_graph.json"
]

def clean_data():
    """Xóa dữ liệu cũ để chạy lại từ đầu"""
    print("\n🧹 BƯỚC 1: Dọn dẹp dữ liệu cũ...")

    # 1. Xóa và tạo lại các thư mục
    for folder in DIRS_TO_CLEAN:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"   - Đã xóa: {folder}")
            except Exception as e:
                print(f"   ⚠️ Không thể xóa {folder}: {e}")

        # Tạo lại thư mục rỗng
        os.makedirs(folder, exist_ok=True)
        print(f"   - Đã tạo lại: {folder}")

    # 2. Xóa các file lẻ
    for file_path in FILES_TO_REMOVE:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"   - Đã xóa file: {file_path}")
            except Exception as e:
                print(f"   ⚠️ Không thể xóa {file_path}: {e}")

def run_step(script_name, description):
    """Chạy một script python con"""
    print(f"\n🚀 BƯỚC: {description} ({script_name})...")
    start_time = time.time()

    script_path = os.path.join("scripts", script_name)
    if not os.path.exists(script_path):
        print(f"❌ Lỗi: Không tìm thấy file {script_path}")
        sys.exit(1)

    try:
        # Gọi subprocess để chạy lệnh: python scripts/ten_file.py
        result = subprocess.run([sys.executable, script_path], check=True)

        elapsed = time.time() - start_time
        print(f"✅ Hoàn thành trong {elapsed:.2f} giây.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy {script_name}. Mã lỗi: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {e}")
        sys.exit(1)

def main():
    print("="*60)
    print("🤖  AUTO PIPELINE: RAW DATA -> KNOWLEDGE GRAPH")
    print("="*60)

    # 1. Dọn dẹp dữ liệu cũ
    clean_data()

    # 2. Chạy lần lượt các script xử lý
    # Lưu ý: Thứ tự này RẤT QUAN TRỌNG

    # B2: PDF -> Text
    run_step("extract_pdf.py", "Trích xuất văn bản từ PDF")

    # B3: Text -> Chunks (JSON)
    run_step("split_text.py", "Chia nhỏ văn bản theo Điều luật")

    # B4: Chunks -> Vector DB & BM25
    run_step("create_vector_index.py", "Tạo Vector Index & BM25")

    # B5: Chunks -> Knowledge Graph (Cần Groq API)
    run_step("build_knowledge_graph.py", "Xây dựng Knowledge Graph (có AI tóm tắt)")

    print("\n" + "="*60)
    print("🎉  XỬ LÝ HOÀN TẤT! HỆ THỐNG ĐÃ SẴN SÀNG.")
    print("👉  Bạn có thể chạy thử chatbot: python scripts/run_cli_chat.py")
    print("="*60)

if __name__ == "__main__":
    main()

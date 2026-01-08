import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Không tìm thấy API KEY trong file .env")
else:
    genai.configure(api_key=api_key)
    print(f"🔑 Đang kiểm tra với Key: {api_key[:5]}...")

    print("\n📋 DANH SÁCH MODEL CÓ THỂ DÙNG (generateContent):")
    try:
        count = 0
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
                count += 1
        if count == 0:
            print("⚠️ Không tìm thấy model nào. Hãy kiểm tra lại API Key hoặc vùng địa lý.")
    except Exception as e:
        print(f"❌ Lỗi khi lấy danh sách: {e}")

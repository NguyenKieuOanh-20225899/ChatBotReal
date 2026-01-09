import streamlit as st
import sys
import os
import time

# 1. Cấu hình đường dẫn (giống như trong run_cli_chat.py)
# Thêm root project vào sys.path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.graph_rag_service import GraphRAGService

# 2. Cấu hình trang Streamlit
st.set_page_config(
    page_title="Trợ lý Luật sư AI",
    page_icon="",
    layout="centered"
)

st.title(" Trợ lý Luật sư AI (GraphRAG)")
st.caption("Hỏi đáp pháp luật dựa trên Văn bản pháp quy & Knowledge Graph")

# 3. Khởi tạo Bot (Sử dụng cache để không phải load lại Model/Vector DB mỗi lần reload trang)
@st.cache_resource
def load_chatbot():
    # Sử dụng đúng đường dẫn như trong file CLI cũ
    return GraphRAGService(
        vector_db_path="data/artifacts",
        graph_path="data/knowledge_graph.json"
    )

try:
    with st.spinner("Đang khởi tạo hệ thống (Loading Vector DB & Graph)..."):
        bot = load_chatbot()
    st.success("Hệ thống đã sẵn sàng!", icon="✅")
except Exception as e:
    st.error(f"Lỗi khởi tạo: {e}")
    st.stop()

# 4. Quản lý lịch sử chat (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Nếu có metadata (nguồn) đi kèm trong tin nhắn cũ, hiển thị lại (nếu lưu)
        if "meta_info" in message:
            st.caption(message["meta_info"])

# 5. Xử lý input từ người dùng
if prompt := st.chat_input("Nhập câu hỏi pháp luật của bạn..."):
    # Hiển thị câu hỏi người dùng
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Xử lý trả lời
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Gọi hàm query từ service của bạn
            # Hàm trả về: answer, meta, latency (theo file run_cli_chat.py)
            answer, meta, latency = bot.query(prompt)

            message_placeholder.markdown(answer)

            # Xử lý hiển thị Metadata (Nguồn trích dẫn)
            n_graph = meta.get('graph_edges_used', 0)
            vector_sources = meta.get('vector_sources', [])
            n_vector = len(vector_sources)

            # Tạo chuỗi thông tin phụ
            meta_info = f"⏱️ Thời gian: {latency:.2f}s | 📊 Graph edges: {n_graph} | 📄 Vector docs: {n_vector}"
            if n_vector > 0:
                # Lấy tên các nguồn (loại bỏ trùng lặp)
                sources_list = list(set(vector_sources))
                meta_info += f"\n\n📚 Nguồn tham khảo: {', '.join(sources_list[:3])}"
                if len(sources_list) > 3:
                    meta_info += "..."

            st.caption(meta_info)

            # Lưu vào lịch sử chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta_info": meta_info
            })

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")

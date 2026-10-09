import streamlit as st
from rag import answer_question

st.set_page_config(page_title="AI-tourism | Penglipuran", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.stApp { background: #f6f5ef; }
[data-testid="stSidebar"] { background: #173d32; }
[data-testid="stSidebar"] * { color: #f5f1df !important; }
.hero { padding: 1.5rem 1.8rem; border-radius: 18px; color: #fff; background: linear-gradient(120deg,#173d32,#39755a); }
.hero h1 { color: #fff; margin: 0; }
.small-muted { color: #64756a; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌿 AI-tourism")
    st.caption("Belajar budaya Penglipuran")
    st.divider()
    st.markdown("**Topik yang bisa ditanyakan**")
    st.markdown("- Tata ruang desa\n- Rumah dan angkul-angkul\n- Hutan bambu\n- Tradisi dan kehidupan warga")
    st.divider()
    st.caption("Jawaban dibuat dari materi lokal yang tersedia di project. Periksa sumber untuk informasi terbaru.")

st.markdown('<section class="hero"><h1>Jelajahi Penglipuran</h1><p>Asisten belajar wisata budaya berbasis sumber.</p></section>', unsafe_allow_html=True)
st.write("")
st.markdown("Tanyakan sesuatu tentang budaya, tata ruang, atau lingkungan Desa Penglipuran.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Lihat sumber"):
                for source in message["sources"]:
                    st.markdown(f"**[{source['title']}]({source['url']})** — {source['section']}")
                    st.caption(source["excerpt"])

if not st.session_state.messages:
    st.markdown("#### Coba tanyakan")
    examples = ["Bagaimana tata ruang tradisional Penglipuran?", "Apa hubungan desa dengan hutan bambu?", "Apa itu angkul-angkul?"]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state.pending_question = example

question = st.chat_input("Tulis pertanyaan…")
question = question or st.session_state.pop("pending_question", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        result = answer_question(question)
    except Exception as error:
        st.error(str(error))
        st.stop()
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
    st.rerun()

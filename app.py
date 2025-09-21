import streamlit as st
import requests
from streamlit_extras.add_vertical_space import add_vertical_space

BACKEND_URL = "http://127.0.0.1:8010"  # FastAPI server

# --- Streamlit Page Config ---
st.set_page_config(page_title="Deep Researcher Agent", page_icon="🤖", layout="wide")

# --- Sidebar Branding ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=120)
    st.title("📑 Deep Researcher Agent")
    st.caption("Built for CodeMate Hackathon 🚀")
    add_vertical_space(2)

    # File upload
    st.subheader("📂 Upload Documents")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("📥 Ingesting document..."):
            files = {"file": uploaded_file}
            res = requests.post(f"{BACKEND_URL}/ingest/file", files=files)
        if res.status_code == 200:
            st.success(res.json()["message"])
        else:
            st.error("❌ Upload failed. Check backend logs.")

# --- Main Area ---
st.title("🔎 Ask Research Questions")

# Keep track of chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Query box
query = st.text_input("💡 Enter your research question:")
if st.button("🚀 Run Query"):
    if query.strip() == "":
        st.warning("Please enter a question first.")
    else:
        with st.spinner("🤔 Thinking... querying backend..."):
            res = requests.post(f"{BACKEND_URL}/query", data={"query": query})
        if res.status_code == 200:
            data = res.json()
            st.session_state["history"].append((query, data))

# --- Display Results ---
for q, data in reversed(st.session_state["history"]):
    st.markdown(f"### ❓ Question: {q}")

    # Synthesis
    with st.container():
        st.markdown("#### 📝 Synthesis")
        st.success(data["synthesis"])

    # Tasks
    st.markdown("#### 📌 Task Breakdown")
    for tr in data["task_results"]:
        with st.expander(f"➡️ {tr['task']}"):
            st.write(tr["answer"])
            st.markdown("**📚 Sources:**")
            for prov in tr["hits"]:
                st.markdown(
                    f"- `{prov['meta']['file']}` — score {prov['score']:.3f}"
                )
                st.caption(prov["meta"]["text"][:300])

    st.markdown("---")

# --- Export Section ---
st.subheader("📤 Export Results")
if st.button("💾 Export Report as PDF & Markdown"):
    if not query:
        st.warning("Run a query before exporting.")
    else:
        with st.spinner("🛠 Generating report..."):
            res = requests.post(f"{BACKEND_URL}/export", data={"query": query})
        if res.status_code == 200:
            data = res.json()
            st.success("✅ Report generated successfully!")
            st.download_button("⬇️ Download Markdown", data=open("report.md").read(), file_name="report.md")
            st.download_button("⬇️ Download PDF", data=open("report.pdf","rb"), file_name="report.pdf")
        else:
            st.error("❌ Failed to generate report.")

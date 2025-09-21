from fastapi import FastAPI, UploadFile, File, Form
import ollama

import pdfplumber
from sentence_transformers import SentenceTransformer
from store import DocStore


from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

import re, os

# ============ Load model first ============
model = SentenceTransformer("all-MiniLM-L6-v2")
store = DocStore(index_path="faiss.index", meta_path="meta.json", dim=model.get_sentence_embedding_dimension())

# ============ FastAPI app ============
app = FastAPI(title="Deep Researcher Agent")

# ============ Planner ============
def simple_planner(question: str):
    q = question.strip()
    if 'compare' in q.lower():
        items = re.findall(r'compare (.+?) (and|with) (.+)', q, flags=re.I)
        if items:
            left, _, right = items[0]
            return [f"Summarize information about {left.strip()}",
                    f"Summarize information about {right.strip()}",
                    f"Compare {left.strip()} and {right.strip()}"]
    if len(q) > 120:
        parts = re.split(r',|;|\band\b|\bthen\b', q)
        tasks = [p.strip() for p in parts if p.strip()]
        if len(tasks) > 1:
            return tasks
    return [q]

# ============ Executor ============
import ollama  # make sure you have this import at the top

def execute_tasks(tasks):
    task_results = []
    for t in tasks:
        # Create embedding for query
        q_emb = model.encode(t, convert_to_numpy=True)
        hits = store.search(q_emb, top_k=3)

        # Collect retrieved context
        context_text = "\n\n".join([h["meta"]["text"] for h in hits])

        # 🔹 Use Ollama for local LLM
        prompt = f"Use ONLY this context to answer:\n\nTask: {t}\n\nContext:\n{context_text}\n\nAnswer clearly:"
        response = ollama.chat(
        model="gemma:2b",   # or "llama2:7b-chat"
        messages=[{"role": "user", "content": prompt}]
      )
        ans = response["message"]["content"].strip()

        task_results.append({"task": t, "answer": ans, "hits": hits})

    # Combine answers
    synthesized = "\n\n".join([
        f"Task: {r['task']}\nAnswer: {r['answer']}" for r in task_results
    ])
    return synthesized, task_results


# ============ Export functions ============
def export_markdown_report(query, synthesis, task_results, out_path="report.md"):
    lines = [f"# Research Report\n\n**Query:** {query}\n\n## Synthesis\n\n{synthesis}\n\n"]
    for tr in task_results:
        lines.append(f"### Task: {tr['task']}\n\n**Answer:**\n{tr['answer']}\n\n**Sources:**\n")
        for h in tr['hits']:
            meta = h['meta']
            lines.append(f"- {meta.get('file')} (page: {meta.get('page')}) — score {h['score']}\n\n  > {meta['text'][:300]}...\n")
    md = "\n".join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)
    return out_path

def md_to_pdf(md_path, pdf_path="report.pdf"):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path)
    story = []

    # Convert each line into a PDF paragraph
    for line in md_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))

    doc.build(story)
    return pdf_path


# ============ Endpoints ============
@app.get("/")
def root():
    return {"message": "Deep Researcher Agent API is running. Visit /docs for Swagger UI"}

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    text_content = ""
    with pdfplumber.open(file.file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            text_content += page_text + "\n"

    # Split text into chunks
    chunks = [text_content[i:i+500] for i in range(0, len(text_content), 500)]

    # Encode embeddings
    chunk_embs = model.encode(chunks, convert_to_numpy=True)
    metas = [{"text": c, "file": file.filename, "page": None, "chunk_idx": i} for i,c in enumerate(chunks)]
    store.add(chunk_embs, metas)

    return {"message": f"Stored {len(chunks)} chunks from {file.filename}"}

@app.post("/query")
async def query_agent(query: str = Form(...)):
    tasks = simple_planner(query)
    synthesis, task_results = execute_tasks(tasks)   # <-- indented with 4 spaces

    provenance = []
    for tr in task_results:
        provs = [{"file": h["meta"]["file"],
                  "page": h["meta"].get("page"),
                  "score": h["score"],
                  "text": h["meta"]["text"][:300]} for h in tr["hits"]]
        provenance.append({"task": tr["task"], "provenance": provs})

    return {
        "query": query,
        "synthesis": synthesis,
        "task_results": task_results,
        "provenance": provenance
    }

@app.post("/export")
async def export_report(query: str = Form(...)):
    tasks = simple_planner(query)
    synthesis, task_results = execute_tasks(tasks)   # <-- also indented 4 spaces

    md_path = export_markdown_report(query, synthesis, task_results, out_path="report.md")
    pdf_path = md_to_pdf(md_path, pdf_path="report.pdf")

    return {
        "message": "Report generated",
        "markdown_file": md_path,
        "pdf_file": pdf_path
    }

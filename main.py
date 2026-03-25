from fastapi import FastAPI
from pydantic import BaseModel
import os
import re
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

app = FastAPI()

# 🔹 Use correct model type (IMPORTANT)
polisher = pipeline("text2text-generation", model="google/flan-t5-small")

# 🔹 video info
video_info = {
    "aircAruvnKk_final.txt": "3Blue1Brown — But what is a Neural Network? — https://youtube.com/watch?v=aircAruvnKk",
    "wjZofJX0v4M_final.txt": "3Blue1Brown — Transformers, the tech behind LLMs — https://youtube.com/watch?v=wjZofJX0v4M",
    "fHF22Wxuyw4_final.txt": "CampusX — What is Deep Learning (Hindi) — https://youtube.com/watch?v=fHF22Wxuyw4",
    "C6YtPJxNULA_final.txt": "CodeWithHarry — All About ML & Deep Learning (Hindi) — https://youtube.com/watch?v=C6YtPJxNULA"
}

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = []
metadata = []

# 🔹 extract timestamp
def extract_time(text):
    match = re.search(r"\[(\d+:\d+)\]", text)
    return match.group(1) if match else "0:00"

# 🔹 Load data
chunk_size = 5

for file in os.listdir("."):
    if file.endswith("_final.txt"):
        with open(file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("===")]

            for i in range(0, len(lines), chunk_size):
                chunk = lines[i:i+chunk_size]

                text = " ".join([re.sub(r"\[\d+:\d+\]\s*", "", l) for l in chunk])

                documents.append(text.lower())

                metadata.append({
                    "original": text,
                    "source": video_info.get(file, file),
                    "start": extract_time(chunk[0]),
                    "end": extract_time(chunk[-1])
                })

# 🔹 FAISS index
embeddings = model.encode(documents)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 🔹 polish answer (FINAL FIX)
def polish_answer(text):
    try:
        prompt = f"Give a one-line clear definition: {text}"

        result = polisher(
            prompt,
            max_length=50,
            do_sample=False
        )

        return result[0]['generated_text'].strip()

    except:
        # fallback
        sentences = text.split(".")
        return sentences[0].strip() + "."

# 🔹 request schema
class QueryRequest(BaseModel):
    question: str

# 🔹 API endpoint
@app.post("/ask")
def ask_question(req: QueryRequest):

    query = req.question.lower().replace("llm", "language model")

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, 5)

    best = None

    # 🔹 pick definition-like chunk
    for i in indices[0]:
        text = documents[i]

        if any(word in text for word in [" is ", " are ", " means ", " refers to "]):
            best = i
            break

    if best is None:
        best = indices[0][0]

    return {
        "question": req.question,
        "answer": polish_answer(metadata[best]["original"]),
        "source": metadata[best]["source"],
        "timestamp": f"{metadata[best]['start']} to {metadata[best]['end']}"
    }

# 🔹 optional root (avoid 404)
@app.get("/")
def home():
    return {"message": "RAG API is running 🚀"}

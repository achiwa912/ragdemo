import chromadb
import requests

def embed(text):
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "all-minilm", "prompt": text},
        timeout=60
        )
    data = r.json()
    if "embedding" not in data:
        print("Ollama returned an error:", data)
        return None

    return data["embedding"]

def chunk_text(text, size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks

def ask_llm(question, context):
    prompt = f"Answer the question using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
        )
    # print("RAW LLM OUTPUT:", r.json())
    return r.json()["response"]

def rag(question):
    qvec = embed(question)
    results = collection.query(query_embeddings= [qvec], n_results=3)
    if not results["documents"] or not results["documents"][0]:
        return "No relevant context found."
    context = "\n\n".join(results["documents"][0])
    return ask_llm(question, context)

client = chromadb.PersistentClient(path="./chroma")
collection = client.get_or_create_collection("demo")

with open("vbs_readme.org") as f:
    text = f.read()

chunks = chunk_text(text, size=80, overlap=20)

for i, chunk in enumerate(chunks):
    vec = embed(chunk)
    if vec is None:
        print(f"Skipping chunk {i} due to embedding error")
        continue
    
    collection.add(
        ids=[f"vbs_{i}"],
        embeddings=[vec],
        documents=[chunk],
        metadatas=[{"source": "vbs_readme.org",
                    "chunk_id": i,
                    }]
        )

results = rag("How does the shifting leaning window work?")

print(results)


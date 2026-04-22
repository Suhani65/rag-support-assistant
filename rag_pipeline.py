from langgraph.graph import StateGraph
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline


# =========================
# INTENT DETECTION
# =========================
def detect_intent(query):
    query = query.lower()

    if "refund" in query or "return" in query:
        return "refund_policy"
    elif "price" in query or "cost" in query:
        return "pricing"
    elif "delivery" in query or "shipping" in query:
        return "shipping"
    else:
        return "general"


# =========================
# STATE
# =========================
class State(TypedDict):
    query: str
    context: str
    answer: str
    intent: str


# =========================
# LOAD PDF
# =========================
loader = PyPDFLoader("data.pdf")
raw_docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

# =========================
# EMBEDDINGS + VECTOR DB
# =========================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever(
    search_kwargs={"k": 5},
    search_type="mmr"   
)


# =========================
# LLM
# =========================
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=256,
    repetition_penalty=1.2
)


# =========================
# RETRIEVE NODE
# =========================
def retrieve(state: State):
    docs = retriever.invoke(state["query"])

    seen = set()
    clean = []

    for d in docs:
        text = " ".join(d.page_content.split())  # removes noise
        if text not in seen:
            seen.add(text)
            clean.append(text)

    state["context"] = "\n".join(clean)
    return state


# =========================
# GENERATE NODE
# =========================
def generate(state: State):
    prompt = f"""
You are a strict customer support assistant.

Intent: {state['intent']}

RULES:
- Use ONLY the context below
- Do NOT repeat information
- Merge similar points
- Keep answer 2–3 lines max

Context:
{state['context']}

Question:
{state['query']}

Answer:
"""

    result = pipe(prompt)
    state["answer"] = result[0]["generated_text"]
    return state


# =========================
# HITL NODE
# =========================
def hitl(state: State):
    if len(state["answer"].strip()) < 20 or "I don't know" in state["answer"]:
        state["answer"] = "⚠ Escalated to Human Support (Low confidence response)"
    return state


# =========================
# BUILD GRAPH
# =========================
graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("hitl", hitl)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "hitl")
graph.set_finish_point("hitl")

app = graph.compile()


# =========================
# RUN SYSTEM
# =========================
query = input("Ask: ")

intent = detect_intent(query)
print("Detected Intent:", intent)

result = app.invoke({
    "query": query,
    "intent": intent,
    "context": "",
    "answer": ""
})

print("\nAnswer:\n", result["answer"])

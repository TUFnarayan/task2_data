import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import os

# -------------------- CONFIGURATION --------------------
st.set_page_config(page_title="Aurora Skies FAQ Bot ✈️", layout="centered")
st.title("🤖 Aurora Skies Airways FAQ Chatbot")

# ✅ Your working Gemini API key
GOOGLE_API_KEY = "AIzaSyCkwdnrSZisXj7arazOy0MBoB6uZL0_pps"
genai.configure(api_key=GOOGLE_API_KEY)

# ✅ Use latest Gemini 2.5 model (works with your key)
MODEL_NAME = "models/gemini-2.5-flash"

# ✅ Path to dataset
DATA_PATH = r"C:\Users\NARAYAN DEVGAN\OneDrive\Desktop\12MegaBlog\rag_chatbot\data\faq_clean.csv"

# -------------------- LOAD FUNCTIONS --------------------
@st.cache_data(show_spinner=False)
def load_faq():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_faq(query, embed_model, index, df, top_k=3):
    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb).astype("float32"), top_k)
    return df.iloc[I[0]]["Question"].tolist(), df.iloc[I[0]]["Answer"].tolist()

def generate_answer(context, user_query):
    prompt = f"""
You are a helpful assistant for Aurora Skies Airways.
Answer the user's query using the context below.

Context:
{context}

User Query:
{user_query}

Provide a concise, friendly, and factual answer.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text.strip()

# -------------------- LOAD MODELS + DATA --------------------
st.markdown("---")
st.subheader("📦 Loading FAQ data and embeddings...")

try:
    with st.spinner("🔄 Loading FAQ data..."):
        df = load_faq()
        st.success(f"✅ FAQ data loaded: {len(df)} entries")
except FileNotFoundError:
    st.error(f"❌ FAQ file not found at '{DATA_PATH}'. Please check the path.")
    st.stop()

with st.spinner("⚙️ Initializing embedding model..."):
    embed_model = load_embed_model()
    faq_texts = (df["Question"].astype(str) + " " + df["Answer"].astype(str)).tolist()
    embeddings = embed_model.encode(faq_texts, convert_to_numpy=True)
    index = build_faiss_index(np.array(embeddings).astype("float32"))

st.success("✅ FAISS index built and ready for queries!")
st.markdown("---")

# -------------------- USER CHAT INTERFACE --------------------
st.subheader("💬 Ask Aurora Skies Airways (type 'exit' to quit)")

user_query = st.text_input("Your question:")

if user_query:
    if user_query.lower() == "exit":
        st.info("👋 Thank you for chatting with Aurora Skies Airways!")
    else:
        with st.spinner("Thinking... ✈️"):
            retrieved_qs, retrieved_as = retrieve_faq(user_query, embed_model, index, df)
            context = "\n\n".join(
                [f"Q: {q}\nA: {a}" for q, a in zip(retrieved_qs, retrieved_as)]
            )
            answer = generate_answer(context, user_query)
        st.markdown("### 🤖 Chatbot:")
        st.write(answer)
        with st.expander("📚 Retrieved Context:"):
            st.write(context)

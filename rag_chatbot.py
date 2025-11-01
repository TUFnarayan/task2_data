import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import google.generativeai as genai

# ✅ Configure Gemini API (Hardcoded API Key)
genai.configure(api_key="AIzaSyDTD5bLTr8tMsHIVVvskCHgMSp0mu3bezk")

# ✅ Load FAQ data
faq_path = "rag_chatbot/data/faq_clean.csv"
df = pd.read_csv(faq_path)
print("✅ FAQ entries loaded:", len(df))
print(df.head())

# ✅ Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Combine Question + Answer for better retrieval
faq_texts = (df["Question"] + " " + df["Answer"]).tolist()

# Generate embeddings
print("🔍 Generating embeddings...")
embeddings = model.encode(faq_texts, convert_to_numpy=True, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("✅ FAISS index ready with", index.ntotal, "entries")

# ✅ Function to retrieve similar FAQs
def retrieve_relevant_faq(query, top_k=2):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [faq_texts[i] for i in indices[0]]
    return "\n\n".join(results)

# ✅ Function to generate Gemini answer
def generate_answer_with_gemini(question, context):
    prompt = f"""
You are a helpful Aurora Skies Airways assistant.
Use ONLY the context below to answer the user’s question.
If the answer isn’t available in the context, respond with:
"I'm sorry, I don’t have that information."

Context:
{context}

Question: {question}
"""

    try:
        # ✅ Use latest available model from your list
        model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model_gemini.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "⚠️ No valid response text received from Gemini."

    except Exception as e:
        print("❌ Error while generating Gemini response:", e)
        return "I'm sorry, I couldn’t generate a response due to a technical issue."

# ✅ Main loop
if __name__ == "__main__":
    print("\n🤖 Aurora Skies Airways Chatbot")
    print("(Type 'exit' to quit)\n")

    while True:
        user_query = input("💬 Ask Aurora Skies Airways (or type 'exit' to quit): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Step 1: Retrieve relevant FAQ context
        context = retrieve_relevant_faq(user_query)
        print("\n📚 Retrieved Context:\n", context)

        # Step 2: Generate response using Gemini
        answer = generate_answer_with_gemini(user_query, context)
        print("\n🤖 Chatbot:\n", answer)

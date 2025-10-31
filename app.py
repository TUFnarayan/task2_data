import chainlit as cl
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
model = genai.GenerativeModel('gemini-1.5-flash')

faiss_index = None
questions = []
answers = []

def build_faiss_index(csv_path='cleaned_airline_faq.csv'):
    global faiss_index, questions, answers
    df = pd.read_csv(csv_path, usecols=['Question', 'Answer'])
    questions = df['Question'].dropna().tolist()
    answers = df['Answer'].dropna().tolist()
    question_embeddings = embedding_model.encode(questions, show_progress_bar=False)
    dimension = question_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(question_embeddings).astype('float32'))

def retrieve_relevant_faqs(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding).astype('float32'), top_k)
    return [{'question': questions[i], 'answer': answers[i]} for i in indices[0]]

def generate_response(query, relevant_faqs):
    context = "\n\n".join([
        f"FAQ {i+1}:\nQ: {faq['question']}\nA: {faq['answer']}"
        for i, faq in enumerate(relevant_faqs)
    ])
    prompt = f"""You are a helpful customer service assistant for Aurora Skies Airways.
Answer ONLY using the FAQs below. If the question cannot be answered, say:
"I don't have specific information about that in our FAQ database. Please contact Aurora Skies Airways customer service for assistance."

FAQs:
{context}

Customer Question: {query}

Your Response:"""
    response = model.generate_content(prompt)
    return response.text

build_faiss_index()

@cl.on_chat_start
async def start():
    await cl.Message(
        author="Aurora Bot ✈️",
        content="""
# 👋 Welcome to **Aurora Skies Airways**
Your personal assistant for all flight-related queries.

**I can help you with:**
- 🧳 Baggage policies
- 💸 Refunds & cancellations
- 🕒 Flight status & delays
- ✍️ Booking changes

_Type your question below to get started!_
"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    thinking_msg = cl.Message(author="Aurora Bot ✈️", content="🔍 Searching FAQ database...")
    await thinking_msg.send()

    relevant_faqs = retrieve_relevant_faqs(message.content, top_k=3)
    thinking_msg.content = "💭 Generating response..."
    await thinking_msg.update()

    response = generate_response(message.content, relevant_faqs)
    await thinking_msg.remove()

    await cl.Message(author="Aurora Bot ✈️", content=response).send()

    references = "**📚 Top FAQ References:**\n"
    for i, faq in enumerate(relevant_faqs[:2], 1):
        references += f"- **Q{i}:** _{faq['question']}_\n"
    await cl.Message(author="Aurora Bot ✈️", content=references).send()

    await cl.Message(
        author="Aurora Bot ✈️",
        content="Was this answer helpful?",
        actions=[
            cl.Action(name="yes", value="yes", label="👍 Yes"),
            cl.Action(name="no", value="no", label="👎 No")
        ]
    ).send()

@cl.action_callback("yes")
async def handle_yes(action):
    await cl.Message(author="Aurora Bot ✈️", content="Glad I could help! 😊").send()

@cl.action_callback("no")
async def handle_no(action):
    await cl.Message(author="Aurora Bot ✈️", content="Sorry to hear that. Please reach out to our support team for further assistance.").send()
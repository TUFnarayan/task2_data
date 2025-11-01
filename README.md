# ✈️ Aurora Skies Airways FAQ Chatbot

A smart FAQ chatbot built with Streamlit, FAISS, and Google Gemini AI, designed to answer customer queries for Aurora Skies Airways using real FAQ data.

# ⚙️ Tech Stack

Python 3.10+

Streamlit – for UI

FAISS – for vector search

SentenceTransformer (MiniLM-L6-v2) – for embeddings

Google Gemini Pro – for generating natural answers

Pandas, NumPy – for data handling

# 🚀 How It Works

Loads FAQ data from faq_clean.csv

Converts questions & answers into embeddings

Uses FAISS to find the most relevant FAQs to your query

Sends the context to Gemini AI to generate a final response

Displays everything neatly on a Streamlit interface

# 🧰 Setup & Run
1️⃣ Clone & Navigate
git clone https://github.com/TUFnarayan/rag_chatbot.git
cd rag_chatbot

2️⃣ Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
 source venv/bin/activate   # macOS/Linux

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Add Your API Key

Create a .env file in the root:

GOOGLE_API_KEY=your_gemini_api_key_here

5️⃣ Run the App
streamlit run streamlit_app.py

# 💬 Example Queries

"Can I retain my ticket for future travel?"

"Are taxes refundable on unused tickets?"

"How can I cancel my flight?"

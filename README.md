# Airline FAQ Chatbot with RAG

An intelligent chatbot powered by Google's Gemini AI model that answers airline-related queries using Retrieval-Augmented Generation (RAG). The chatbot provides accurate responses by referencing a curated database of airline FAQs, combining the power of AI with reliable information.

## 🌟 Features

- Real-time question answering using Gemini AI
- Semantic search using FAISS for accurate FAQ retrieval
- Interactive chat interface powered by Chainlit
- Pre-processed airline FAQ database
- Efficient vector embeddings using Sentence Transformers

## 🛠️ Tech Stack

- Python
- Google Gemini AI
- Chainlit
- FAISS (Facebook AI Similarity Search)
- Sentence Transformers
- Pandas
- NumPy

## 📋 Prerequisites

- Python 3.9 or higher
- Google API key for Gemini AI
- Git (for cloning the repository)

## 🚀 Getting Started
Add config.toml to .chainlit folder
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd airline-rag-chatbot
   ```

2. **Create and activate a virtual environment** (Optional but recommended)
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a `.env` file in the project root
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

5. **Run the application**
   ```bash
   chainlit run app.py
   ```

6. Open your browser and navigate to `http://localhost:8000`


## 📸 Screenshot

![Airline Chatbot Interface](screenshot.png)


![](screenshot2.png)

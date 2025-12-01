# 🧠 Mitting Summarizer

A productivity-focused web app using **Google Gemini LLM** and **LangChain** to assist with:

- Meeting summarization
- Task prioritization
- Contextual note-taking and querying

Built with **Streamlit** for an interactive and user-friendly experience.

---

## Features

### 1. Meeting Summarizer
- Paste your meeting transcript.
- Generates a concise summary, action points, and follow-up suggestions.

### 2. Task Prioritizer
- Enter tasks line by line.
- Reorders tasks based on urgency, deadlines, and effort.
- Outputs tasks with assigned priority (high, medium, low).

### 3. Note Assistant
- Add notes and maintain them in a searchable vector database.
- Automatically generates a summary of all notes.
- Ask questions based on the notes for instant answers.

---

## Tech Stack

- **Streamlit** – Frontend UI
- **LangChain** – LLM orchestration and chains
- **Google Gemini LLM** – Large language model for text generation
- **FAISS** – Vector database for semantic search
- **HuggingFace Embeddings** – Embedding model for notes

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/google-gemini-productivity-assistant.git
cd google-gemini-productivity-assistant
Create and activate a virtual environment

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies

pip install -r requirements.txt

Set your Google Gemini API key

export GEMINI_API_KEY="your_api_key_here"   # Linux/Mac
set GEMINI_API_KEY="your_api_key_here"      # Windows

Run the Streamlit app:

streamlit run app.py

Choose a tool from the sidebar: Meeting Summarizer, Task Prioritizer, or Note Assistant.

Follow on-screen instructions to input transcripts, tasks, or notes.

Get instant summaries, prioritized tasks, or answers to note-based questions.

Notes Assistant Details
Notes are split into chunks for vector-based semantic search using FAISS.

The LLM generates summaries and answers queries by leveraging relevant context from stored notes.

Supports incremental addition of notes with real-time summary updates.

Environment Variables
GEMINI_API_KEY or GOOGLE_API_KEY – Required for authentication with Google Gemini.


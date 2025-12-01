import os
import streamlit as st
from typing import Optional, List

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from google import genai


# Google Gemini LLM Wrapper

class GoogleGeminiLLM(LLM):
    model: str = "gemini-2.5-flash"
    client: Optional[genai.Client] = None

    @property
    def _llm_type(self) -> str:
        return "google-gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        return resp.text

# Setup Gemini client

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set GEMINI_API_KEY environment variable.")
    st.stop()

client = genai.Client(api_key=api_key)
llm = GoogleGeminiLLM(client=client, model="gemini-2.5-flash")


# Prompt Templates

meeting_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are a meeting assistant. Summarize the transcript, give action points, and suggestions for follow-up.

Transcript:
{transcript}
"""
)

task_prompt = PromptTemplate(
    input_variables=["tasks"],
    template="""
You are a productivity assistant. Reorder these tasks based on urgency, deadlines, and effort.
Output points with 'task' and 'priority' (high, medium, low):

Tasks:
{tasks}
"""
)

meeting_chain = LLMChain(llm=llm, prompt=meeting_prompt)
task_chain = LLMChain(llm=llm, prompt=task_prompt)


# Note Assistant

class NoteAssistant:
    def __init__(self, llm):
        self.notes = []
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.llm = llm

    def add_notes(self, notes):
        self.notes.extend(notes)
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_text(" ".join(self.notes))
        self.vectorstore = FAISS.from_texts(docs, self.embeddings)

    def get_summary(self):
        if not self.notes:
            return "No notes added yet."
        combined_notes = "\n".join(self.notes)
        prompt = f"Summarize the following notes:\n{combined_notes}"
        return self.llm._call(prompt)

    def query(self, query):
        if not self.vectorstore:
            return "No notes added yet."
        docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}"
        return self.llm._call(prompt)


# Streamlit UI Setup

st.set_page_config(page_title="Google Gemini Productivity Assistant", layout="wide")
st.title("🧠 Google Gemini Productivity Assistant")

mode = st.sidebar.radio("Choose a tool", ["Meeting Summarizer", "Task Prioritizer", "Note Assistant"])


# Initialize session state

if "meeting_summary" not in st.session_state:
    st.session_state.meeting_summary = ""

if "task_result" not in st.session_state:
    st.session_state.task_result = ""

if "note_assistant" not in st.session_state:
    st.session_state.note_assistant = NoteAssistant(llm=llm)
    st.session_state.notes = []
    st.session_state.note_summary = ""
    st.session_state.note_query = ""
    st.session_state.note_answer = ""


# Meeting Summarizer

if mode == "Meeting Summarizer":
    st.header("Meeting Summarizer")
    transcript = st.text_area("Paste your meeting transcript here:", height=200)
    if st.button("Summarize"):
        if transcript.strip() != "":
            st.session_state.meeting_summary = meeting_chain.run(transcript=transcript)
        else:
            st.warning("Please enter a transcript.")

    if st.session_state.meeting_summary:
        st.subheader("Summary & Action Points")
        st.write(st.session_state.meeting_summary)


# Task Prioritizer

elif mode == "Task Prioritizer":
    st.header("Task Prioritizer")
    tasks_input = st.text_area("Enter tasks (one per line):", height=200)
    if st.button("Prioritize"):
        tasks = [t.strip() for t in tasks_input.split("\n") if t.strip()]
        if tasks:
            st.session_state.task_result = task_chain.run(tasks="\n".join(tasks))
        else:
            st.warning("Please enter some tasks.")

    if st.session_state.task_result:
        st.subheader("Prioritized Tasks")
        st.write(st.session_state.task_result)


# Note Assistant (Single View)

else:
    st.header("Contextual Note Assistant")

    # Add new notes
    new_notes_input = st.text_area("Add your notes here (one per line):", height=200)
    if st.button("Add Notes"):
        new_notes = [n.strip() for n in new_notes_input.split("\n") if n.strip()]
        if new_notes:
            st.session_state.notes.extend(new_notes)
            st.session_state.note_assistant.add_notes(new_notes)
            # Update summary immediately
            st.session_state.note_summary = st.session_state.note_assistant.get_summary()
            st.success("Notes added and summary generated!")

    # Display current summary
    if st.session_state.notes:
        st.subheader("Notes Summary")
        st.text_area("Summary of all notes", st.session_state.note_summary, height=200)

    # Ask question directly below
    query = st.text_input("Ask a question based on the notes:")
    if query.strip() != "":
        st.session_state.note_query = query
        st.session_state.note_answer = st.session_state.note_assistant.query(query)
        st.subheader("Answer")
        st.write(st.session_state.note_answer)

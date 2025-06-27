# -*- coding: utf-8 -*-
import streamlit as st
import requests
from bs4 import BeautifulSoup
import cohere
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
api_key = "Yk3Qjs1Vm6uio7srVW6jFybM5cGCrFBLMTkXldzE"

co = cohere.Client(api_key)

# Define URLs to fetch live content from
urls = [
    # Wikipedia
    "https://en.wikipedia.org/wiki/",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Deep_learning",

    # AI News / Blogs
    "https://www.bbc.com/news/",
    "https://www.bbc.com/news/topics/cz4pr2gd85pt/artificial-intelligence",
    "https://www.ibm.com/blogs/research/tag/ai/",
    "https://www.microsoft.com/en-us/research/blog/category/artificial-intelligence/",
    "https://openai.com/blog",
    "https://huggingface.co/blog",

    # Tech News / Startups
    "https://techcrunch.com/tag/artificial-intelligence/",
    "https://www.analyticsvidhya.com/blog/category/artificial-intelligence/",
    "https://venturebeat.com/category/ai/",
    "https://builtin.com/artificial-intelligence",

    # Government / Research Sites
    "https://www.niti.gov.in/ai",
    "https://www.nist.gov/artificial-intelligence",
    "https://www.drdo.gov.in/latest-news",

    # Educational / Frameworks
    "https://keras.io/",
    "https://pytorch.org/blog/",
    "https://scikit-learn.org/stable/whats_new.html"
]


# Function to fetch and clean webpage text
def fetch_page_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        return text.strip()
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

# Get and display documents
st.title("🌐 Live Semantic Search with Cohere")
st.info("Using Cohere `embed-english-v3.0` with `search_document` and `search_query` input types")

st.subheader("🔗 Fetched Pages")
documents = []
sources = []

for url in urls:
    text = fetch_page_text(url)
    if text and not text.startswith("Error"):
        documents.append(text)
        sources.append(url)
        st.markdown(f"- ✅ **{url}**")

if not documents:
    st.error("❌ Failed to fetch any documents.")
    st.stop()

# Embed documents
with st.spinner("Embedding live documents..."):
    doc_embeddings = co.embed(
        texts=documents,
        model="embed-english-v3.0",
        input_type="search_document"
    ).embeddings

# Define cosine similarity
def cosine_similarity(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    )

# Semantic search function
def semantic_search(query, top_k=3):
    query_embedding = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    similarities = [cosine_similarity(query_embedding, doc) for doc in doc_embeddings]
    top_matches = sorted(zip(documents, similarities, sources), key=lambda x: x[1], reverse=True)[:top_k]
    return top_matches

# User query input
st.subheader("🔍 Enter your search query:")
query = st.text_input("For example: 'AI in healthcare'")

# Show results
if query:
    with st.spinner("Running semantic search..."):
        results = semantic_search(query)

    st.subheader("📄 Top Matching Results")
    for i, (doc, score, url) in enumerate(results, 1):
        st.markdown(f"""
        ### {i}. [{url}]({url})
        - **Similarity Score:** `{score:.4f}`
        - **Preview:** {doc[:300]}...
        """)

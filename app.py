import streamlit as st
from search_utils import search_tenders

st.set_page_config(page_title="Tender Semantic Search", layout="wide")
st.title("🔍 Tender Semantic Search Engine")

query = st.text_input("Enter your search query (e.g., 'boat supply', 'air conditioner supply'):")

top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

if query:
    with st.spinner("Searching..."):
        results = search_tenders(query, top_k=top_k)

    st.success(f"Top {top_k} Results for: '{query}'")

    for i, row in results.iterrows():
        st.markdown(f"**Summary**: {row['summary']}")
        st.markdown(f"**Details**: {row['details']}")
        st.markdown(f"**Similarity Score**: {row['similarity_score']:.4f}")
        st.markdown("---")

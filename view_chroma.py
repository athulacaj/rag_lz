import streamlit as st
from langchain_chroma import Chroma
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="ChromaDB Explorer",
    layout="wide",
    page_icon="🧊"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card Styling */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: #262730;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #fff;
    }
    .metric-label {
        font-size: 14px;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Connection ---
@st.cache_resource
def get_db():
    return Chroma(
        collection_name="langchain",
        persist_directory="./vector_db",
        embedding_function=None
    )

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("ChromaDB Explorer")
        st.markdown("Visualizing your vector database collection **'langchain'**")
    
    try:
        db = get_db()
        # Fetch data (increase limit if needed)
        data = db._collection.get(limit=100)
        
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        
        total_docs = len(ids)
        
        # --- Sidebar ---
        st.sidebar.markdown("## ⚙️ Filter & Options")
        search_query = st.sidebar.text_input("🔍 Search Content", placeholder="Type to search...")
        
        # Stats in Sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Collection Stats")
        st.sidebar.markdown(f"**Total Documents:** `{total_docs}`")
        st.sidebar.markdown(f"**Persist Dir:** `./vector_db`")

        if not ids:
            st.warning("⚠️ The collection is empty.")
            return

        # View Toggle or Sort (Simple for now)
        
        # --- Filtering Logic ---
        filtered_indices = []
        for i in range(total_docs):
            doc_content = documents[i] if documents[i] else ""
            if search_query:
                if search_query.lower() in doc_content.lower():
                    filtered_indices.append(i)
            else:
                filtered_indices.append(i)

        st.markdown(f"### Showing {len(filtered_indices)} Documents")
        st.markdown("---")

        # --- Display Cards ---
        if not filtered_indices:
            st.info("No documents match your search.")
        
        for idx in filtered_indices:
            doc_id = ids[idx]
            content = documents[idx]
            meta = metadatas[idx]
            
            # Create a card-like container
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                
                with c1:
                    st.caption("🆔 DOCUMENT ID")
                    st.code(doc_id, language="text")
                    
                    if meta:
                        st.caption("📋 METADATA")
                        st.json(meta, expanded=False)
                    else:
                        st.caption("No Metadata")

                with c2:
                    st.caption("📄 CONTENT PREVIEW")
                    
                    screen_content = content[:400] + "..." if len(content) > 400 else content
                    st.markdown(f"**{screen_content}**")
                    
                    with st.expander("Show Full Content"):
                        st.text_area(label="Full Text", value=content, height=200, key=f"text_{doc_id}", disabled=True)

    except Exception as e:
        st.error(f"❌ Error connecting to ChromaDB: {e}")
        st.info("Ensure you have created the database first.")

if __name__ == "__main__":
    main()

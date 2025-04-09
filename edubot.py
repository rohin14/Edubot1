import streamlit as st
import os
import tempfile
import warnings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize embedding function - import only when needed to avoid path issues
@st.cache_resource
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# App title and configuration
st.set_page_config(page_title="Educational RAG App", page_icon="📚", layout="wide")
st.title("📚 Smart Study Assistant")
st.write("Upload your textbooks and get personalized learning content")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = "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"

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.success("API Key: ✓ Pre-configured")
    model_name = st.selectbox("Select Groq Model:", ["llama3-70b-8192", "llama3-8b-8192"], index=0)
    
    st.header("Uploaded Textbooks")
    for file in st.session_state.uploaded_files:
        st.write(f"- {file}")
    
    st.header("Learning Preferences")
    learning_style = st.selectbox("Learning Style:", ["Visual", "Auditory", "Read/Write", "Kinesthetic", "Balanced"], index=4)
    complexity_level = st.select_slider("Content Complexity:", ["Beginner", "Intermediate", "Advanced", "Expert"], value="Intermediate")
    include_examples = st.checkbox("Include examples in answers", value=True)
    include_analogies = st.checkbox("Include analogies in answers", value=True)
    include_questions = st.checkbox("Include practice questions", value=True)

def process_pdfs(pdf_files):
    """Processes uploaded PDFs and creates vector embeddings."""
    temp_dir = tempfile.mkdtemp()
    all_docs = []
    uploaded_filenames = []
    
    with st.spinner("Processing PDFs and creating vector embeddings..."):
        for pdf in pdf_files:
            temp_pdf_path = os.path.join(temp_dir, pdf.name)
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf.getbuffer())
            try:
                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load()
                if docs:
                    all_docs.extend(docs)
                    uploaded_filenames.append(pdf.name)
                    st.success(f"Successfully processed: {pdf.name}")
                else:
                    st.warning(f"No content extracted from: {pdf.name}")
            except Exception as e:
                st.error(f"Error processing {pdf.name}: {str(e)}")
    
    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)
        try:
            # Get embeddings using the cached function
            embeddings = get_embeddings()
            vectorstore = FAISS.from_documents(splits, embeddings)
            st.success(f"✅ Created embeddings for {len(splits)} text chunks")
            return vectorstore, uploaded_filenames
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            st.exception(e)
            return None, []
    else:
        return None, []

# Upload section
uploaded_pdfs = st.file_uploader("Upload PDF Textbooks", type="pdf", accept_multiple_files=True)

# Process button
if uploaded_pdfs and st.button("Process Textbooks"):
    vectorstore, filenames = process_pdfs(uploaded_pdfs)
    if vectorstore:
        st.session_state.vectorstore = vectorstore
        # Only add unique filenames
        for filename in filenames:
            if filename not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(filename)
        st.success("Textbooks processed and ready for questions!")

def create_rag_chain():
    """Creates a Retrieval-Augmented Generation (RAG) chain."""
    if st.session_state.vectorstore is None:
        return None
    
    os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model_name=model_name)
    
    template = """
    You are an expert educational assistant helping students learn effectively.
    
    Learning preferences:
    - Learning style: {learning_style}
    - Complexity level: {complexity_level}
    - Include examples: {include_examples}
    - Include analogies: {include_analogies}
    - Include practice questions: {include_questions}
    
    Use the following context from textbooks to answer the question:
    {context}
    
    Question: {question}
    
    Provide a comprehensive and educational response tailored to the specified learning preferences.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_input(question):
        docs = retriever.invoke(question)
        return {
            "context": "\n\n".join([doc.page_content for doc in docs]),
            "question": question,
            "learning_style": learning_style,
            "complexity_level": complexity_level,
            "include_examples": "Yes" if include_examples else "No",
            "include_analogies": "Yes" if include_analogies else "No",
            "include_questions": "Yes" if include_questions else "No"
        }
    
    chain = (
        RunnablePassthrough(format_input) 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

st.header("Learn From Your Textbooks")
tab1, tab2 = st.tabs(["Ask Questions", "Study Guide Generator"])

with tab1:
    if st.session_state.vectorstore is None:
        st.info("Please upload and process textbooks before asking questions.")
    else:
        question = st.text_input("What would you like to learn about?", key="question_input")
        if question and st.button("Get Answer", key="answer_btn"):
            try:
                rag_chain = create_rag_chain()
                if rag_chain:
                    with st.spinner("Finding the best answer for you..."):
                        response = rag_chain.invoke(question)
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

with tab2:
    if st.session_state.vectorstore is None:
        st.info("Please upload and process textbooks first.")
    else:
        topic = st.text_input("Topic for study guide:", key="topic_input")
        if topic and st.button("Generate Study Guide", key="guide_btn"):
            try:
                rag_chain = create_rag_chain()
                if rag_chain:
                    with st.spinner("Creating your personalized study guide..."):
                        response = rag_chain.invoke(f"Create a comprehensive study guide on {topic} that includes key concepts, explanations, examples, and practice questions.")
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

# Reset button at the bottom
if st.button("Reset App", key="reset_btn"):
    st.session_state.vectorstore = None
    st.session_state.uploaded_files = []
    st.success("App reset successfully. You can upload new textbooks now.")
    st.experimental_rerun()

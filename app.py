
import os
import streamlit as st

# Load credentials from Streamlit Cloud Secrets or local .env
try:
    if "OPENROUTER_API_KEY" in st.secrets:
        os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

    if "OPENROUTER_MODEL" in st.secrets:
        os.environ["OPENROUTER_MODEL"] = st.secrets["OPENROUTER_MODEL"]
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv()

from src.rag import retrieve
from src.generator import generate_answer


st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Document Assistant")

st.write(
    "Ask questions about your documents and receive "
    "context-grounded answers powered by AI."
)

st.info(
    "This application uses TF-IDF retrieval and "
    "OpenRouter for answer generation."
)

question = st.text_area(
    "Enter your question",
    placeholder="What information are you looking for?",
    height=100
)

if st.button("Generate Answer", type="primary", use_container_width=True):

    if not question.strip():
        st.warning("Please enter a question.")

    elif not os.getenv("OPENROUTER_API_KEY"):
        st.error("OpenRouter API key is not configured.")

    else:
        try:
            with st.spinner("Searching documents and generating answer..."):

                retrieved_documents = retrieve(question)

                if not retrieved_documents:
                    st.warning("No relevant information was found.")

                else:
                    result = generate_answer(
                        question,
                        retrieved_documents
                    )

                    answer = result.get("answer")

                    if answer:
                        st.subheader("🤖 Generated Answer")
                        st.write(answer)
                    else:
                        st.error(
                            "The model did not return an answer. "
                            "Please check your generator configuration."
                        )

                    with st.expander("View Retrieved Documents"):
                        for index, document in enumerate(
                            retrieved_documents, start=1
                        ):
                            st.markdown(f"**Passage {index}**")
                            st.write(document["text"])
                            st.caption(
                                f"Similarity score: {document['score']}"
                            )

        except Exception as error:
            st.error(f"An error occurred: {error}")
            st.exception(error)

st.divider()

st.caption(
    "Built with Python, Scikit-learn, TF-IDF, "
    "Streamlit, and OpenRouter."
)

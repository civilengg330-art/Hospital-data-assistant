```python
import os
import streamlit as st

from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hospital Knowledge Assistant",
    page_icon="🏥",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

FAISS_DIR = "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROK_MODEL = "grok-4.6"

TOP_K = 3


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🏥 Hospital Knowledge Assistant")

st.caption(
    "Ask questions about hospital policies and procedures."
)


# ============================================================
# LOAD API KEY FROM STREAMLIT SECRETS
# ============================================================

try:
    XAI_API_KEY = st.secrets["XAI_API_KEY"]

except Exception:
    st.error(
        "XAI_API_KEY is not configured in Streamlit Secrets."
    )
    st.stop()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    vectorstore = load_vectorstore()

except Exception as e:

    st.error("Unable to load the FAISS knowledge base.")

    st.exception(e)

    st.stop()


# ============================================================
# CREATE GROK CLIENT
# ============================================================

@st.cache_resource
def create_grok_client():

    return OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )


client = create_grok_client()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about hospital policies..."
)


if question:

    # --------------------------------------------------------
    # DISPLAY USER QUESTION
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # SEARCH FAISS
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching hospital policies..."):

            try:

                results = vectorstore.similarity_search(
                    question,
                    k=TOP_K
                )

            except Exception as e:

                st.error("Error searching the knowledge base.")

                st.exception(e)

                st.stop()


        if not results:

            st.warning(
                "I could not find relevant information "
                "in the hospital knowledge base."
            )

            st.stop()


        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context_parts = []

        for i, doc in enumerate(results, start=1):

            metadata = doc.metadata

            department = metadata.get(
                "department",
                "Unknown department"
            )

            source_file = metadata.get(
                "source_file",
                "Unknown source"
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"""
SOURCE {i}
Department: {department}
Document: {source_file}
Page: {page}

CONTENT:
{doc.page_content}
"""
            )


        context = "\n\n".join(context_parts)


        # ----------------------------------------------------
        # GROK PROMPT
        # ----------------------------------------------------

        system_prompt = """
You are a hospital knowledge-base assistant.

Answer the user's question using ONLY the information
provided in the retrieved hospital policy context.

Rules:

1. Do not invent hospital policies.
2. Do not use outside knowledge to answer the question.
3. If the answer cannot be found in the provided context,
   clearly say that the information was not found in the
   hospital knowledge base.
4. Give a clear and concise answer.
5. When useful, mention the relevant policy or procedure.
6. Do not mention the internal RAG system or FAISS.
"""


        user_prompt = f"""
Hospital policy context:

{context}

User question:

{question}

Answer the question based only on the context above.
"""


        # ----------------------------------------------------
        # CALL GROK
        # ----------------------------------------------------

        with st.spinner("Generating answer..."):

            try:

                response = client.chat.completions.create(
                    model=GROK_MODEL,

                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],

                    temperature=0.1
                )

                answer = response.choices[0].message.content

            except Exception as e:

                st.error("Error communicating with Grok.")

                st.exception(e)

                st.stop()


        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        st.markdown(answer)


        # ----------------------------------------------------
        # DISPLAY SOURCES
        # ----------------------------------------------------

        st.divider()

        st.markdown("**Sources**")

        displayed_sources = set()

        for doc in results:

            metadata = doc.metadata

            department = metadata.get(
                "department",
                "Unknown department"
            )

            source_file = metadata.get(
                "source_file",
                "Unknown source"
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            source_key = (
                department,
                source_file,
                page
            )

            if source_key not in displayed_sources:

                displayed_sources.add(source_key)

                st.caption(
                    f"📄 {source_file}  "
                    f"| Department: {department}  "
                    f"| Page: {page}"
                )


        # ----------------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
```

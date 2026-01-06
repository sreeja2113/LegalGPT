import os
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS  # or replace with Pinecone if needed
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_together import Together
from dotenv import load_dotenv

load_dotenv()

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
    return db

prompt_template = """
<s>[INST]
As a legal chatbot specializing in the Indian Penal Code, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
- Respond in a bullet-point format to clearly delineate distinct aspects of the legal query.
- Each point should accurately reflect the breadth of the legal provision in question, avoiding over-specificity unless directly relevant to the user's query.
- Clarify the general applicability of the legal rules or sections mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
- Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
- Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations unless otherwise specified.
- Conclude with a brief summary that captures the essence of the legal discussion and corrects any common misinterpretations related to the topic.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"]
)

def configure_retrieval_chain():
    api_key = os.getenv("TOGETHER_API_KEY")

    llm = Together(
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        temperature=0.5,
        max_tokens=1024,
        together_api_key=api_key
    )

    retriever = load_vectorstore().as_retriever(search_kwargs={"k": 3})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain

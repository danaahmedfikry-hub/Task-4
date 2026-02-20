import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

loader = TextLoader(r"D:\Task 5\arabic.txt", encoding="utf-8")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
docs = text_splitter.split_documents(data)

print(f"Total number of chunks after splitting the document: {len(docs)}\n")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = FAISS.from_documents(docs, embeddings)

template = """Use the following context to answer the question below.
If you don't know the answer, just say you don't know, do not make up an answer.
Answer the question in Arabic.

{context}

Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = input("Enter your question: ")
response = qa_chain.invoke({"query": question})

print(f"--- Answer ---\n{response['result']}\n")

retrieved_docs = response["source_documents"]
print(f"--- Retrieval Info ---")
print(f"Number of chunks retrieved: {len(retrieved_docs)}\n")

for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Chunk {i}:")
    print(doc.page_content)
    print("-" * 30)
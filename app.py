from dotenv import load_dotenv
load_dotenv()  # loading all the env variables
import streamlit as st
import os
import google.generativeai as genai
# pdf reader
from PyPDF2 import PdfReader

# faiss db
from langchain.vectorstores import FAISS
# langchain with gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# config your Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    # read pdf one by one
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # read page one by one
        for page in pdf_reader.pages:
            # append the extracted text to the var
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # set text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    # split the text
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_db(chunks):
    # set embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # start embedding and save locally
    vector_db = FAISS.from_texts(chunks, embedding = embeddings)
    vector_db.save_local("faiss_vector_db")

def get_conversational_chain():
    # set default prompt template
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # set default model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # set a prompt from prompt template
    prompt = PromptTemplate(template = prompt_template, input_variables= ['context', 'question'])
    # set QA chain for responses
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(question):
    # set embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # load faiss vector db
    vector_db = FAISS.load_local("faiss_vector_db", embeddings, allow_dangerous_deserialization=True)
    # search for similarity in db for the given question
    similar_context = vector_db.similarity_search(question)

    # chain
    chain = get_conversational_chain()

    # get response
    response = chain(
        {"input_documents" : similar_context, "question" : question}
    )
    # show response
    print(response)
    st.write("Reply:", response["output_text"])

def main():
    # set streamlit app
    st.set_page_config("PDFs Bot")
    st.header("Chat with Multiple PDF files")

    question = st.text_input("Ask a Question from the PDF Files")

    if question:
        user_input(question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # get text from pdfs
                text = get_pdf_text(pdf_docs)
                # get chunks from text
                chunks = get_text_chunks(text)
                # set a vector db
                get_vector_db(chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from uuid import uuid4
from time import strftime

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text


class GeminiPDFChatbot:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
        self.prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
        provided context say, "I'm unsure about the correct answer, as it is not fully available from the context", but try to provide as best an answer as you can nonetheless.\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        self.model = ChatGoogleGenerativeAI(model="gemini-pro",
                                            client=genai,
                                            temperature=0.3,
                                            )
        self.prompt = PromptTemplate(template=self.prompt_template,
                                     input_variables=["context", "question"])
        self.chain = load_qa_chain(llm=self.model, chain_type="stuff", prompt=self.prompt)
        self.faiss_index_directory = "faiss_index"

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            print(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=1000)
        chunks = splitter.split_text(text)
        return chunks  # list of strings

    def get_vector_store(self, chunks):
        vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
        print('storing into %s' % st.session_state.faiss_index_directory)
        vector_store.save_local(st.session_state.faiss_index_directory)

    def clear_chat_history(self):
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    def user_input(self, user_question):
        if 'faiss_index_directory' not in st.session_state:
            raise "No faiss index directory found"
        new_db = FAISS.load_local(st.session_state.faiss_index_directory, self.embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        response = self.chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

        print(response)
        return response


    def initialise_with_pdf(self, filename="display.pdf"):
        pdf_docs = [filename]
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        self.get_vector_store(text_chunks)

    def initialise_session(self):
        if "session_id" not in st.session_state:
            session_id = str(uuid4().hex)
            print('new session with id %s' % session_id)
            st.session_state.session_id = "%s" % session_id
            session_directory = "sessions/%s" % session_id
            os.makedirs(session_directory, exist_ok=True)
            st.session_state.session_directory = session_directory


    def main(self):
        self.initialise_session()
        st.set_page_config(
            page_title="Gemini PDF Chatbot",
            page_icon="🤖"
        )

        init_pdf_file = 'initial.pdf'

        # if the directory faiss_index is older than the file display.pdf
        if os.path.exists("faiss_index"):
            if os.path.getmtime("faiss_index/index.pkl") < os.path.getmtime(init_pdf_file):
                self.initialise_with_pdf(init_pdf_file)
        else:
            self.initialise_with_pdf(init_pdf_file)
        st.success("Ready")

        with st.sidebar:
            st.title("Menu:")
            st.markdown("proudly served to you by [L-CAS](https://lcas.lincoln.ac.uk/)")
            st.markdown("[![Logo](https://i0.wp.com/lcas.lincoln.ac.uk/wp/wp-content/uploads/2012/05/cropped-lcas_logo_150dpi-720x987.png?&h=100)](https://lcas.lincoln.ac.uk/)")
            st.write("This chat bot uses the [Gemini Pro model](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/) to answer questions about the CMP3103 module. It's conversation context is defined by the content of the PDF file provided below.")
            st.write("__DISCLAIMER:__ *The responses in this chat bot are generated by a generative artificial intelligence large language model. Please note that the information provided may by completely incorrect. You cannot rely upon the provided answers and we accept no liability if you get incorrect answers or responses. You may use this chat bot to explore the content, but you must not rely on it for any assessments. We do not assume any responsibility or liability for the use or interpretation of this content.*")

            with open(init_pdf_file, "rb") as file:
                file.seek(0)
                st.download_button(
                    label="Download context PDF file",
                    data=file,
                    file_name='context.pdf',
                    mime="application/pdf",
                    key="initial-pdf"
                )    

        # # Sidebar for uploading PDF files
        # with st.sidebar:
        #     st.title("Menu:")
        #     pdf_docs = st.file_uploader(
        #         "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        #     if st.button("Submit & Process"):
        #         with st.spinner("Processing..."):
        #             raw_text = get_pdf_text(pdf_docs)
        #             text_chunks = get_text_chunks(raw_text)
        #             get_vector_store(text_chunks)
        #             st.success("Done")

        # Main content area for displaying chat messages
        st.title("Chat about CMP3103 content using Gemini🤖")
        st.write("Welcome to the chat!")
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
        st.sidebar.write("*This service is provided 'as is' for students enrolled in the module CMP3103 at the University of Lincoln. It is not intended for any other use.*")

        # Chat input
        # Placeholder for chat messages

        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "Talk to me about the CMP3103 module"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        # Display chat messages and bot response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.user_input(prompt)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response['output_text']:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
                    timestamp_filename = "qa_%s.md" % strftime("%Y%m%d-%H%M%S")
                    with open("%s/%s.txt" % (st.session_state.session_directory, timestamp_filename), "w") as f:
                        f.write('# Prompt\n\n%s\n\n# Answer\n\n%s' % (prompt,full_response))

            if response is not None:
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)


if __name__ == "__main__":
    main()

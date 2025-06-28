import streamlit as st
from src.helper import get_pdf_text,get_text_chunks,get_vector_store,get_conversational_chain


def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.markdown(f"**🧑 User:** {message.content}")
        else:
            st.markdown(f"**🤖 Bot:** {message.content}")

def main():
    st.set_page_config("Information Retrieval")
    st.header("PDF-Reader📄")
    user_question = st.text_input("Ask any question from the PDF Files")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)
        

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your pdf files and click on theSubmit & Process Button",accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return

            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)

                st.success("Done")

if __name__=="__main__":
    main()
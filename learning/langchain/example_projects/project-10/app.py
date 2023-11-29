import os
import streamlit as st
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

load_dotenv(Path('../../.env'))

from user_utils import *

# Creating session variables
if 'HR_tickets' not in st.session_state:
    st.session_state['HR_tickets'] = []
if 'IT_tickets' not in st.session_state:
    st.session_state['IT_tickets'] = []
if 'Transport_tickets' not in st.session_state:
    st.session_state['Transport_tickets'] = []


def main():

    st.header("Automatic Ticket Classification Tool")
    # Capture user input
    st.write("We are here to help you, please ask your question:")
    if user_input := st.text_input("🔍"):
        # creating embeddings instance
        embeddings = create_embeddings()

        # Function to pull index data from Pinecone
        index = pull_from_pinecone("8fe9ac1f-b6b6-4c3e-af89-e9cefa72f26d",
                                   "us-west1-gcp-free",
                                   "project10",
                                   embeddings)

        # This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
        relavant_docs = get_similar_docs(index, user_input)

        # This will return the fine tuned response by LLM
        response = get_answer(relavant_docs, user_input)
        st.write(response)

        if button := st.button("Submit ticket?"):
            # Get Response

            embeddings = create_embeddings()
            query_result = embeddings.embed_query(user_input)

            # loading the ML model, so that we can use it to predit the class to which this compliant belongs to...
            department_value = predict(query_result)
            st.write(f"your ticket has been sumbitted to : {department_value}")

            # Appending the tickets to below list, so that we can view/use them later on...
            if department_value == "HR":
                st.session_state['HR_tickets'].append(user_input)
            elif department_value == "IT":
                st.session_state['IT_tickets'].append(user_input)
            else:
                st.session_state['Transport_tickets'].append(user_input)


if __name__ == '__main__':
    main()

import streamlit as st
from utils import *


# Define the Streamlit app
def main():
    st.title("Customer Care Call Summarization")

    if uploaded_files := st.file_uploader(
        "Upload recorded .mp3 files", type=["mp3"], accept_multiple_files=True
    ):
        st.write("Uploaded Files:")

        # Display uploaded files and buttons in a tabular form
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name


            col1, col2, col3 = st.columns([0.1, 1, 2])
            with col1:
                st.write("-")
            with col2:
                st.write(file_name)
            with col3:
                if send_button := st.button(f"Send Email for {file_name}"):
                    email_summary(file_name)
                    st.success(f"Send email for: {file_name}")


# Run the Streamlit app
if __name__ == "__main__":
    main()

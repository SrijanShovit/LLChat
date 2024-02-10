import streamlit as st
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load Hugging Face API token from environment variables
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
repo_id = "tiiuae/falcon-7b"

# Initialize HuggingFaceHub
llm = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id=repo_id,
    model_kwargs={"temperature": 0.7, "max_new_tokens": 500}
)

# Define prompt template
template = """
Question: {question}
Answer: Let's give a detailed answer.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)

# Streamlit app
def main():
    st.title("LLChat: Ask your essays here😜")
    
    # Input box for user to enter questions
    user_input = st.text_input("Enter your question:")
    
    # Button to submit the question
    if st.button("Ask"):
        # Generate response using LLMChain model
        response = chain.run({"question": user_input})
        
        # Display the response
        st.write("Response:", response)

# Run the Streamlit app
if __name__ == "__main__":
    main()
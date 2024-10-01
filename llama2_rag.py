# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:22:39 2024

@author: TMHAAS31
"""

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
import contextlib
import io
import logging

logging.getLogger("chromadb").setLevel(logging.WARNING)
torch.cuda.empty_cache()
#torch.cuda.empty_cache() -- clear GPU memory

# Function to load and extract text from a PDF file
def load_pdf_text(filename):
    reader = PdfReader(filename)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    pdf_texts = [text for text in pdf_texts if text]  # Filter out empty pages
    return pdf_texts

# Initialize character splitter
character_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '. ', ' ', ''], chunk_size=1000
)

# Split text based on characters
def split_texts(pdf_texts):
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

    # Token splitter (Using SentenceTransformers for embedding)
    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50  # Assuming each chunk represents a token-like division
    )
    
    # Split text into token-like chunks
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)
    
    return token_split_texts


# Define a custom embedding function that conforms to the new ChromaDB interface
class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    # The __call__ method is required to adhere to the EmbeddingFunction interface
    def __call__(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

# Function to retrieve documents for a specific query and process them
def retrieve_and_process_query(single_query, chroma_collection, cross_encoder, model, tokenizer):
    # Query the collection for the current single query
    result = chroma_collection.query(
        query_texts=[single_query], n_results=10, include=["documents", 'embeddings']
    )

    retrieved_documents = result['documents']

    # Filter unique documents
    unique_documents = set()
    for documents in retrieved_documents:
        for document in documents:
            unique_documents.add(document)
    unique_documents = list(unique_documents)

    # Create document-query pairs for scoring
    pairs = []
    for doc in unique_documents:
        pairs.append([single_query, doc])

    # Score the document-query pairs
    scores = cross_encoder.predict(pairs)

    # Reorder documents based on scores
    top_indices = np.argsort(scores)[::-1][:5]
    top_documents = [unique_documents[i] for i in top_indices]

    # Prepare the final context for the LLaMA2 model
    context = '\n\n'.join(top_documents)

    # Generate response using the LLaMA2 model
    response = generated_multi_query(single_query, context, model, tokenizer)
    
    # Extract only the answer to the query
    answer_start = response.lower().find("answer the query:") + len("answer the query:")
    answer = response[answer_start:].strip()  # Get everything after the prompt

    return answer

# Function to process multiple queries in sequence
def process_queries(queries, chroma_collection, cross_encoder, model, tokenizer):
    for query in queries:
        response = retrieve_and_process_query(query, chroma_collection, cross_encoder, model, tokenizer)
        print('\n\n')
        print(response)
        #return response
    
# Function to generate responses using LLaMA2 model
def generated_multi_query(query, context, model, tokenizer):
    prompt = f"""You are a financial expert providing detailed insights based on the company's annual financial report. 
    Using the context below, answer the query in a clear and accurate manner.
    
    Based on the following context:
    {context}
    
    Answer the query: {query}"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate a response
    outputs = model.generate(**inputs, max_length=2048, do_sample=True, temperature=0.1)
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Suppress output during embedding addition
def add_embeddings_to_collection(chroma_collection, ids, documents):
    # Suppress standard output
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        chroma_collection.add(ids=ids, documents=documents)

# Main function to run the document extraction and querying process
def main(pdf_filename):
    # Load PDF and extract text
    pdf_texts = load_pdf_text(pdf_filename)

    # Split text into manageable chunks
    token_split_texts = split_texts(pdf_texts)

    # Initialize the embedding function
    embedding_function = CustomEmbeddingFunction(r'I:/LLM/all-MiniLM-L6-v2')

    # ChromaDB client and collection setup
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.get_or_create_collection(
        'microsoft_collect', embedding_function=embedding_function
    )

    # Adding documents and their embeddings to Chroma collection
    ids = [str(i) for i in range(len(token_split_texts))]
    add_embeddings_to_collection(chroma_collection, ids, token_split_texts)
    #chroma_collection.add(ids=ids, documents=token_split_texts)


    # Define the queries

    queries = [
            "What is the company's revenue growth compared to the previous year?",
            "What is the company's net income for the year compared to previous year? Show the income breakdown for the year."
            ]
    """
    "How has the company's earnings per share (EPS) changed year-over-year?",
    "What is the company’s gross profit margin?",
    "What are the company’s total assets and liabilities?",
    "What is the company’s debt-to-equity ratio?",
    "How much has the company invested in research and development (R&D)?",
    "What is the company’s cash flow from operations?",
    "What is the company’s free cash flow for the year?",
    "How much has the company spent on capital expenditures (CapEx)?",
    "What are the company’s major revenue streams?",
    "What are the company's key cost drivers or operating expenses?",
    "How does the company’s profit margin compare to previous years?",
    "What are the company’s short-term and long-term debt obligations?",
    "Has the company declared dividends? If yes, how much?",
    "What is the company's return on equity (ROE) and return on assets (ROA)?",
    "What are the company’s current liquidity ratios (e.g., current ratio, quick ratio)?",
    "How does the company’s stock performance compare to its peers in the same industry?",
    "What are the company's plans for future growth and capital investments?",
    "What are the key risks and uncertainties mentioned in the report that could affect future performance?",
    """

    # Initialize CrossEncoder model for scoring
    cross_encoder = CrossEncoder(r'I:\LLM\ms-marco-MiniLM-L-6-v2')

    # Load LLaMA2 model and tokenizer
    model_name = r'I:\LLM\llama\Llama-2-7b-chat-hf'
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

    # Process each query
    process_queries(queries, chroma_collection, cross_encoder, model, tokenizer)

# Run the main function with the desired PDF file
if __name__ == "__main__":
    pdf_filename = r'I:/AS Projects/llama2 Rag/Meta-2021-Annual-Report.pdf'  # Change this path as needed
    main(pdf_filename)

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

# Load PDF and extract text
reader = PdfReader(r'I:/AS Projects/llama2 Rag/Meta-2021-Annual-Report.pdf')
pdf_texts = [p.extract_text().strip() for p in reader.pages]
pdf_texts = [text for text in pdf_texts if text]  # Filter out empty pages

# Initialize character splitter
character_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '. ', ' ', ''], chunk_size=1000)

# Split text based on characters
character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

# Token splitter (Using SentenceTransformers for embedding)
token_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=50  # Assuming each chunk represents a token-like division
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# Import necessary packages
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

# Define a custom embedding function that conforms to the new ChromaDB interface
class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    # The __call__ method is required to adhere to the EmbeddingFunction interface
    def __call__(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

# Initialize the embedding function
embedding_function = CustomEmbeddingFunction(r'I:/LLM/all-MiniLM-L6-v2')

# ChromaDB client and collection setup
import chromadb
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    'microsoft_collect', embedding_function=embedding_function
)

# Adding documents and their embeddings to Chroma collection
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

# Verify document count
count = chroma_collection.count()
print(f"Number of documents added: {count}")



# Define the queries
original_query = "You are a financial expert."
generated_queries = [
    'What has been the investment in research and development?',
    'What has been the year-over-year growth?'
]
queries = [original_query] + generated_queries

# Query the collection
results = chroma_collection.query(
    query_texts=queries, n_results=10, include=["documents", 'embeddings']
)

retrieved_documents = results['documents']

# Filter unique documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)
unique_documents = list(unique_documents)

# Initialize CrossEncoder model for scoring
cross_encoder = CrossEncoder(r'I:\LLM\ms-marco-MiniLM-L-6-v2')

# Create document-query pairs
pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc])

# Score the document-query pairs
scores = cross_encoder.predict(pairs)

# Display the scores and reorder documents
print('Scores:')
for score in scores:
    print(score)

print('New ordering:')
for o in np.argsort(scores)[::-1]:
    print(o)

# Get top documents based on scores
top_indices = np.argsort(scores)[::-1][:5]
top_documents = [unique_documents[i] for i in top_indices]

# Prepare the final context for LLaMA2 model
context = '\n\n'.join(top_documents)

# Load LLaMA2 model and tokenizer
model_name = r'I:\LLM\llama\Llama-2-7b-chat-hf'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

# Function to generate responses using LLaMA2 model
def generated_multi_query(query, context, model, tokenizer):
    prompt = f"""You are a knowledgeable financial assistant.
    Your users are inquiring about a financial report.
    
    Based on the following context:
    {context}
    
    Answer the query: {query}"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate a response
    outputs = model.generate(**inputs, max_length=1024, do_sample=True, temperature=0.7)
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Call the function to get results
res = generated_multi_query(query=original_query, context=context, model=model, tokenizer=tokenizer)

# Print the generated response
print(res)

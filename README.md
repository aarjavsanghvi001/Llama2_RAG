# Llama2_RAG

**Overview**
This project implements a query system that extracts information from PDF documents, specifically focusing on financial reports. It leverages advanced natural language processing (NLP) techniques using the LLaMA model and ChromaDB for efficient document retrieval and embedding management.

**Key Components**

PDF Text Extraction:
The code utilizes the pypdf library to read and extract text from PDF files, ensuring that the content is clean and free of empty pages.

Text Splitting:
Text is split into manageable chunks using the RecursiveCharacterTextSplitter from LangChain. This allows for efficient processing and ensures that the model handles input within its token limits.

Embedding Generation:
A custom embedding function, built with the SentenceTransformer, converts text chunks into dense vector representations. This enables semantic understanding of the text for better retrieval.

ChromaDB Integration:
ChromaDB is employed to manage and query document embeddings. The system can efficiently retrieve relevant documents based on user queries.

Query Processing:
The system processes user-defined queries against the retrieved documents. A CrossEncoder model scores the relevance of each document to the query, ensuring the best responses are selected.

Response Generation:
The LLaMA model generates detailed answers based on the context of the top retrieved documents, providing insightful responses to financial queries.

**Functionality**
Users can input queries related to the financial report, such as revenue growth, net income, or cash flow.
The system retrieves relevant sections from the PDF, ranks them based on their relevance, and generates comprehensive answers using the LLaMA model.

**Usage**
To use the system:
Ensure you have the necessary libraries installed, including pypdf, transformers, sentence-transformers, and chromadb.
Specify the path to the PDF file containing the financial report.
Define your queries related to the content of the PDF.
Run the script to extract and analyze the document, receiving structured insights in response to your queries.

**Acknowledgements**
This project integrates cutting-edge NLP models and techniques to provide a robust solution for financial document analysis. Contributions and improvements are welcome!



import openai
import os
import sys
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Step 1: Parse PDF documents, extract text by page, and store with page references
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page_num in range(len(reader.pages)):
        page_text = reader.pages[page_num].extract_text()
        if page_text:
            pages.append({"text": page_text, "page_num": page_num + 1})
    return pages

# Step 2: Generate embeddings for each page and store along with page details
def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

def create_embeddings_for_pdf_directory(directory_path):
    document_embeddings = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            pages = extract_text_from_pdf(pdf_path)
            for page in pages:
                embedding = get_embedding(page['text'])
                document_embeddings.append({
                    "embedding": embedding,
                    "text": page['text'],
                    "page_num": page['page_num'],
                    "document": filename
                })
    return document_embeddings

# Step 3: Read query and PDF directory from command-line arguments
if len(sys.argv) < 3:
    print("Usage: python rag_pipeline.py '<user_query>' '<pdf_directory>'")
    sys.exit(1)

user_query = sys.argv[1]
pdf_directory = sys.argv[2]

# Step 4: Create embeddings for documents in the specified directory
document_embeddings = create_embeddings_for_pdf_directory(pdf_directory)

# Step 5: Generate query embedding and find the most similar document page
query_embedding = get_embedding(user_query)
similarities = [cosine_similarity([query_embedding], [doc['embedding']]).flatten()[0] for doc in document_embeddings]
most_similar_index = np.argmax(similarities)
relevant_document = document_embeddings[most_similar_index]

# Step 6: Use OpenAI's GPT-4 to generate a response
prompt = f"""
You are a bot that retrieves specific document pages based on user questions. Answer briefly and include the document name and page number.
This is the relevant content: "{relevant_document['text']}"
Document: {relevant_document['document']}, Page: {relevant_document['page_num']}
User input: "{user_query}"
Provide a concise response based on this document content.
"""

response = openai.Completion.create(
    model="gpt-4",
    prompt=prompt,
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].text.strip())


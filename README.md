# RAG System with PDF Documents

This project is a Retrieval-Augmented Generation (RAG) system that retrieves relevant pages from PDF documents based on a user query, then uses GPT-4 to generate a concise response.

## Features

- Extracts text from each page of PDF documents.
- Uses embeddings to find the most similar document page to the query.
- Provides page number and document name in the response.

## Requirements

- Python 3.7+
- Packages: `openai`, `PyPDF2`, `scikit-learn`, `numpy`
- OpenAI API key

## Setup

1. **Clone Repo**
2. **Install Packages**

   ```bash
   pip install openai PyPDF2 scikit-learn numpy
   ```

3. **Set Up API Key**  
   Replace `'your-api-key'` in the script.

4. **Place PDFs**  
   Add PDF files to a directory, e.g., `pdf_files`, and update `pdf_directory` in the script to match.

## Usage

Run the RAG pipeline with a query by passing it as an argument:

```bash
python rag.py "Your query here" "path/folder/*.pdf"
```

## Code Overview

- **Text Extraction**: Parses and extracts text from each PDF page.
- **Embeddings**: Generates embeddings for each page.
- **Similarity Search**: Uses cosine similarity to find the best page.
- **LLM Response**: GPT-4 generates a response based on the relevant document content.

## Example Output

```
Document: sample_document.pdf, Page: 3
Response: "Based on your interest in hiking, this page suggests exploring nearby trails for a refreshing outdoor experience."
```

### License

MIT License

# StudyBuddy AI

StudyBuddy AI is an intelligent learning assistant that leverages advanced AI technologies to enhance the studying experience. It uses a sophisticated Retrieval-Augmented Generation (RAG) system to provide personalized answers and generate topic-specific flashcards based on your study materials.

## Features

- **Document Processing**: Ingest and process PDF study materials.
- **Intelligent Q&A**: Get accurate answers to questions about your study material.
- **Dynamic Flashcard Generation**: Create custom flashcards on any topic within your documents.
- **RAG Technology**: Utilizes state-of-the-art Retrieval-Augmented Generation for context-aware responses.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/YourUsername/StudyBuddyAI.git
   cd StudyBuddyAI
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)

5. Pull the Llama 2 model:
   ```
   ollama pull llama2
   ```

## Usage

1. Place your PDF study materials in the `documents/` directory.

2. Run the main script:
   ```
   python studybuddy_ai.py
   ```

3. Follow the on-screen prompts to:
   - Process PDF documents
   - Load the vector database
   - Set up the QA chain
   - Ask questions about your study material
   - Generate flashcards on specific topics

## How It Works

StudyBuddy AI uses a RAG (Retrieval-Augmented Generation) system:

1. **Document Processing**: PDFs are loaded and split into chunks.
2. **Embedding and Indexing**: Text chunks are converted to vectors and indexed for fast retrieval.
3. **Query Processing**: User questions are vectorized and matched with relevant document chunks.
4. **Answer Generation**: An AI model generates answers based on retrieved chunks.
5. **Flashcard Creation**: The same process is used to generate relevant flashcards.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is part of the "52 Weeks of AI Innovation" challenge.
- Special thanks to the creators and maintainers of the libraries used in this project.
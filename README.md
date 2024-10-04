# StudyBuddy AI

StudyBuddy AI is an intelligent learning assistant that leverages advanced AI technologies to enhance the studying experience. It uses a sophisticated Retrieval-Augmented Generation (RAG) system to provide personalized answers, generate topic-specific flashcards, create custom exams, and offer intelligent scoring and feedback based on your study materials.
If you don't know what RAG is and want to build one from scratch, you can follow my tutorial [here](https://github.com/FarazFazelifar/RAG-Demo1)

## Features

- **Document Processing**: Ingest and process PDF study materials.
- **Intelligent Q&A**: Get accurate answers to questions about your study material.
- **Dynamic Flashcard Generation**: Create custom flashcards on any topic within your documents, with difficulty levels (easy, medium, hard).
- **Exam Generation**: Generate custom exams based on specific page ranges and question types.
- **Exam Taking & Scoring**: Take exams within the application and receive AI-powered scoring and feedback.
- **Progress Tracking**: Keep track of your exam history and performance over time.
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

2. follow the main script:
   ```
   python studybuddy_ai.ipynb
   ```

3. Follow the on-screen prompts to:
   - Process PDF documents
   - Load the vector database
   - Set up the QA chain
   - Ask questions about your study material
   - Generate flashcards on specific topics
   - Create and take custom exams
   - View your exam history and progress

## How It Works

StudyBuddy AI uses an advanced RAG (Retrieval-Augmented Generation) system:

1. **Document Processing**: PDFs are loaded and split into chunks.
2. **Embedding and Indexing**: Text chunks are converted to vectors and indexed for fast retrieval.
3. **Query Processing**: User questions are vectorized and matched with relevant document chunks.
4. **Answer Generation**: An AI model generates answers based on retrieved chunks.
5. **Flashcard Creation**: The system generates relevant flashcards with varying difficulty levels.
6. **Exam Generation**: Custom exams are created based on user-specified parameters.
7. **Intelligent Scoring**: AI evaluates answers, providing detailed feedback and partial credit where appropriate.
8. **Progress Tracking**: Exam results are stored and analyzed to track improvement over time.

## Advanced Features

- **Chain of Thought (CoT) Prompting**: Enhances AI responses for more nuanced and accurate answers.
- **ReAct Prompting**: Implements step-by-step reasoning in AI evaluations.
- **Few-shot Learning**: Improves AI understanding of different question types.
- **Pandas Integration**: Manages exam history and progress data efficiently.

## Acknowledgments

- This project is part of the "52 Weeks of AI Innovation" challenge.
- Special thanks to the creators and maintainers of the libraries used in this project.

Stay tuned for more updates and feel free to suggest features or improvements!

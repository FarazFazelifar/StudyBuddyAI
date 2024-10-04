import os
import pickle
import json
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import math
from datetime import datetime 
import pandas as pd

class AdvancedStudyBuddyAI:
    def __init__(self, pdf_directory: str, db_path: str = "studybuddy_vectordb"):
        self.pdf_directory = pdf_directory
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.qa_chain = None
        self.llm = Ollama(model="llama3.1", temperature=0.2)

    def load_and_process_pdfs(self):
        documents = []
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.pdf_directory, filename)
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = filename  
                documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            if 'page' not in chunk.metadata:
                chunk.metadata['page'] = chunk.metadata.get('page', 1)  

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.db_path)
        print(f"Processed and saved {len(chunks)} chunks from {len(documents)} documents.")

    def load_vectorstore(self):
        if os.path.exists(self.db_path):
            self.vectorstore = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
            print("Vector database loaded successfully.")
        else:
            print("No existing vector database found. Please process PDFs first.")

    def setup_qa_chain(self):
        if self.vectorstore is None:
            print("Please load or process documents before setting up the QA chain.")
            return

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    def query(self, question: str) -> Dict:
        if self.qa_chain is None:
            print("Please set up the QA chain before querying.")
            return {"answer": "QA chain not set up", "sources": []}

        result = self.qa_chain({"query": question})
        return {
            "answer": result['result'],
            "sources": [doc.page_content for doc in result['source_documents']]
        }

    def generate_and_save_flashcards(self, topic: str, num_cards: int):
        if self.vectorstore is None:
            print("Please load or process documents before generating flashcards.")
            return

        relevant_chunks = self.vectorstore.similarity_search(topic, k=6)
        combined_content = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        chunk_metadata = [
            {
                "source": chunk.metadata['source'],
                "page": chunk.metadata['page']
            } for chunk in relevant_chunks
        ]

        num_easy = max(1, int(num_cards * 0.1))
        num_medium = int(num_cards * 0.4)
        num_hard = num_cards - num_easy - num_medium

        flashcard_prompt = f"""
        Based on the following content about '{topic}', generate {num_cards} flashcards:

        {combined_content}

        Create flashcards with the following distribution:
        - {num_easy} Easy flashcard(s)
        - {num_medium} Medium flashcards
        - {num_hard} Hard flashcards

        Use the following examples as a guide for each difficulty level:

        Easy:
        Q: What is the capital of France?
        A: Paris

        Medium:
        Q: Explain the concept of supply and demand in economics.
        A: Supply and demand is a fundamental economic principle that describes the relationship between the quantity of a good or service available (supply) and the desire for it from buyers (demand). When supply increases and demand remains unchanged, it leads to lower prices. Conversely, when demand increases and supply remains unchanged, it leads to higher prices.

        Hard:
        Q: Analyze the impact of the Industrial Revolution on global socioeconomic structures.
        A: The Industrial Revolution had far-reaching effects on global socioeconomic structures. It led to urbanization as people moved from rural areas to cities for factory jobs, created a new working class and middle class, widened the wealth gap between industrialized and non-industrialized nations, sparked technological innovations that changed daily life and work, and ultimately shifted the global balance of power towards industrialized nations. These changes laid the groundwork for modern capitalist economies and continue to influence global socioeconomic dynamics today.

        For each flashcard:
        1. Consider the topic and difficulty level.
        2. Formulate a question that tests understanding at the appropriate level.
        3. Provide a comprehensive answer that fully addresses the question.
        4. Ensure the question and answer are directly related to the topic '{topic}'.
        5. Determine which chunk(s) of information the flashcard is primarily based on.

        Format the response as a JSON array with 'question', 'answer', 'difficulty', and 'chunk_indices' keys for each flashcard. The 'chunk_indices' should be a list of indices (0-5) indicating which chunks were used to create the flashcard.

        Your output must ONLY include a JSON file, with no additional words.
        Review your output 3 times and format the response in JSON.
        **Remember**: It is critical to **ONLY** output the JSON file.
        """

        response = self.llm(flashcard_prompt)
        try:
            flashcards = json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse flashcard JSON: {response}")
            return

        for card in flashcards:
            card['metadata'] = [chunk_metadata[i] for i in card['chunk_indices']]
            del card['chunk_indices'] 

        if not os.path.exists("flashcards"):
            os.makedirs("flashcards")
        
        filename = os.path.join("flashcards", f"{topic.replace(' ', '_')}.json")
        with open(filename, "w") as f:
            json.dump(flashcards, f, indent=2)
        
        print(f"Generated and saved {len(flashcards)} flashcards on the topic '{topic}' to {filename}")

        readable_filename = os.path.join("flashcards", f"{topic.replace(' ', '_')}_readable.txt")
        with open(readable_filename, "w") as f:
            for i, card in enumerate(flashcards, 1):
                f.write(f"Flashcard {i} ({card['difficulty']}):\n")
                f.write(f"Q: {card['question']}\n")
                f.write(f"A: {card['answer']}\n")
                f.write("Sources:\n")
                for source in card['metadata']:
                    f.write(f"  - {source['source']}, Page {source['page']}\n")
                f.write("\n")
        
        print(f"Also saved a human-readable version to {readable_filename}")

    def generate_exam(self, start_page: int, end_page: int, num_questions: int, question_types: List[str]):
        if self.vectorstore is None:
            print("Please load or process documents before generating an exam.")
            return

        all_chunks = self.vectorstore.similarity_search("", k=10000) 
        relevant_chunks = [chunk for chunk in all_chunks if start_page <= chunk.metadata['page'] <= end_page]

        if not relevant_chunks:
            print(f"No content found between pages {start_page} and {end_page}.")
            return

        chunks_per_question = math.ceil(len(relevant_chunks) / num_questions)
        chunk_groups = [relevant_chunks[i:i+chunks_per_question] for i in range(0, len(relevant_chunks), chunks_per_question)]

        exam_questions = []
        for i, chunk_group in enumerate(chunk_groups, 1):
            combined_content = "\n\n".join([chunk.page_content for chunk in chunk_group])
            question_type = question_types[i % len(question_types)]  

            exam_prompt = f"""
                Based on the following content, generate a {question_type} question:

                {combined_content}

                Question type: {question_type}

                Follow these steps to create a challenging question that tests deep understanding of the material:

                1. Analyze: Carefully read through the content and identify key concepts, important facts, or critical relationships.
                2. Conceptualize: Formulate a question that requires understanding of these key elements, not just memorization.
                3. Structure: Craft the question according to the specified type ({question_type}).
                4. Answer: Provide the correct answer, ensuring it's comprehensive and accurate.
                5. Distract (if applicable): For multiple choice, create plausible but incorrect options.
                6. Review: Ensure the question is clear, unambiguous, and truly tests understanding.

                Here are examples of how to approach different question types:

                1. Multiple Choice:
                Analyze: Content discusses the causes of the French Revolution.
                Conceptualize: Question about the primary cause.
                Structure: "What was the primary cause of the French Revolution?"
                Answer: "The extreme inequality between social classes"
                Distract: ["The rise of Napoleon", "The American Revolution", "The Industrial Revolution"]

                2. Fill in the Blank:
                Analyze: Content explains photosynthesis process.
                Conceptualize: Question testing understanding of inputs and outputs.
                Structure: "In photosynthesis, plants use sunlight, water, and ______ to produce ______ and oxygen."
                Answer: "carbon dioxide, glucose"

                3. Short Answer:
                Analyze: Content covers Newton's laws of motion.
                Conceptualize: Question applying the concept to a real-world scenario.
                Structure: "Explain how Newton's Third Law applies when a person is ice skating."
                Answer: "When a person ice skates, they push backwards against the ice (action force). The ice exerts an equal and opposite force forward on the skater (reaction force), propelling them forward."

                Now, create a {question_type} question following this approach.

                Format the response as a JSON object with the following structure:
                {{
                    "question_type": "{question_type}",
                    "question": "The question text",
                    "correct_answer": "The correct answer",
                    "options": ["Option A", "Option B", "Option C", "Option D"],  // Only for multiple choice
                    "explanation": "A brief explanation of why this is the correct answer",
                    "thought_process": "A step-by-step breakdown of how you formulated this question"
                }}

                Remember to make the question challenging and ensure it tests deep understanding rather than mere recall.

                Your output must ONLY include a JSON file, with no additional words.
                Review your output 3 times and format the response in JSON.
                **Remember**: It is critical to **ONLY** output the JSON file.
                """

            response = self.llm(exam_prompt)
            try:
                question = json.loads(response)
                question['metadata'] = {
                    'source': chunk_group[0].metadata['source'],
                    'start_page': chunk_group[0].metadata['page'],
                    'end_page': chunk_group[-1].metadata['page']
                }
                exam_questions.append(question)
            except json.JSONDecodeError:
                print(f"Failed to parse question JSON for group {i}: {response}")

        if not os.path.exists("exams"):
            os.makedirs("exams")
        
        filename = os.path.join("exams", f"exam_p{start_page}-p{end_page}.json")
        with open(filename, "w") as f:
            json.dump(exam_questions, f, indent=2)
        
        print(f"Generated and saved an exam with {len(exam_questions)} questions to {filename}")

        readable_filename = os.path.join("exams", f"exam_p{start_page}-p{end_page}_readable.txt")
        with open(readable_filename, "w") as f:
            for i, question in enumerate(exam_questions, 1):
                f.write(f"Question {i} ({question['question_type']}):\n")
                f.write(f"Q: {question['question']}\n")
                f.write(f"A: {question['correct_answer']}\n")
                if 'options' in question:
                    f.write("Options:\n")
                    for option in question['options']:
                        f.write(f"  - {option}\n")
                f.write(f"Source: {question['metadata']['source']}, ")
                f.write(f"Pages: {question['metadata']['start_page']}-{question['metadata']['end_page']}\n\n")
        
        print(f"Also saved a human-readable version to {readable_filename}")

    def list_available_exams(self):
        if not os.path.exists("exams"):
            print("No exams available. Generate an exam first.")
            return None
        
        exams = [f for f in os.listdir("exams") if f.endswith(".json")]
        if not exams:
            print("No exams available. Generate an exam first.")
            return None
        
        print("Available exams:")
        for i, exam in enumerate(exams, 1):
            print(f"{i}. {exam}")
        
        choice = int(input("Enter the number of the exam you want to take: ")) - 1
        if 0 <= choice < len(exams):
            return os.path.join("exams", exams[choice])
        else:
            print("Invalid choice.")
            return None

    def take_exam(self, exam_file: str):
        with open(exam_file, 'r') as f:
            exam_data = json.load(f)
        
        user_answers = []
        for i, question in enumerate(exam_data, 1):
            print(f"\nQuestion {i} ({question['question_type']}):")
            print(question['question'])
            if 'options' in question:
                for j, option in enumerate(question['options'], 1):
                    print(f"{j}. {option}")
            
            user_answer = input("Your answer: ")
            user_answers.append(user_answer)
        
        return self.score_exam(exam_data, user_answers)

    def score_exam(self, exam_data: List[Dict], user_answers: List[str]):
        correct_answers = 0
        feedback = []

        for question, user_answer in zip(exam_data, user_answers):
            is_correct = False
            score = 0

            if question['question_type'].lower() == 'multiple choice':
                # For multiple choice, we can directly compare the answers
                is_correct = user_answer.lower() == question['correct_answer'].lower()
                score = 1 if is_correct else 0
                explanation = "Your answer is correct." if is_correct else "Your answer is incorrect."
            else:
                # For other question types, use the LLM to evaluate
                scoring_prompt = f"""
                Task: Evaluate the user's answer to the following question and determine if it's correct.

                Question: {question['question']}
                Question Type: {question['question_type']}
                Correct Answer: {question['correct_answer']}
                User's Answer: {user_answer}

                Please follow these steps to evaluate the answer:

                1. Understand the question and the correct answer.
                2. Compare the user's answer to the correct answer, looking for semantic similarity and key points.
                3. Consider partial credit for partially correct answers.
                4. Be lenient with minor spelling mistakes or slight variations in phrasing.
                5. Determine if the answer is correct, partially correct, or incorrect.
                6. Assign a score: 1 for correct, 0.5 for partially correct, 0 for incorrect.

                Provide your evaluation in the following JSON format:
                {{
                    "is_correct": true/false,
                    "score": 1/0.5/0,
                    "explanation": "Brief explanation of why the answer is correct, partially correct, or incorrect"
                }}
                """

                response = self.llm(scoring_prompt)
                try:
                    score_data = json.loads(response)
                    is_correct = score_data['is_correct']
                    score = score_data['score']
                    explanation = score_data['explanation']
                except json.JSONDecodeError:
                    print(f"Failed to parse scoring JSON: {response}")
                    explanation = "Error in scoring this question."

            correct_answers += score

            if not is_correct:
                # If the answer is incorrect, ask the LLM for detailed feedback
                feedback_prompt = f"""
                The user's answer to the following question was incorrect or partially correct:

                Question: {question['question']}
                Correct Answer: {question['correct_answer']}
                User's Answer: {user_answer}

                Please provide:
                1. A clear explanation of why the user's answer is incorrect or partially correct.
                2. A detailed explanation of the correct answer.
                3. Advice on how the user can improve their understanding of this topic.

                Format your response as a JSON object with the following structure:
                {{
                    "error_explanation": "Explanation of why the user's answer is wrong",
                    "correct_answer_explanation": "Detailed explanation of the correct answer",
                    "improvement_advice": "Advice for the user to improve their understanding"
                }}
                """

                response = self.llm(feedback_prompt)
                try:
                    feedback_data = json.loads(response)
                    feedback.append(f"Question: {question['question']}\n"
                                    f"Your answer: {user_answer}\n"
                                    f"Correct answer: {question['correct_answer']}\n"
                                    f"Explanation: {explanation}\n"
                                    f"Why it's incorrect: {feedback_data['error_explanation']}\n"
                                    f"Correct answer explained: {feedback_data['correct_answer_explanation']}\n"
                                    f"How to improve: {feedback_data['improvement_advice']}\n")
                except json.JSONDecodeError:
                    print(f"Failed to parse feedback JSON: {response}")
                    feedback.append(f"Question: {question['question']}\n"
                                    f"Your answer: {user_answer}\n"
                                    f"Correct answer: {question['correct_answer']}\n"
                                    f"Explanation: {explanation}\n"
                                    f"Error in generating detailed feedback for this question.")
            else:
                feedback.append(f"Question: {question['question']}\n"
                                f"Your answer: {user_answer}\n"
                                f"Correct answer: {question['correct_answer']}\n"
                                f"Explanation: {explanation}\n")

        score_percentage = (correct_answers / len(exam_data)) * 100
        
        print("\nExam Results:")
        print(f"Correct Answers: {correct_answers}/{len(exam_data)}")
        print(f"Overall Score: {score_percentage:.2f}%")
        print("\nDetailed Feedback:")
        for i, fb in enumerate(feedback, 1):
            print(f"\nQuestion {i}:\n{fb}")

        return correct_answers, score_percentage

    def update_exam_history(self, exam_file: str, num_questions: int, correct_answers: int, score: float):
        history_file = "exam_history.csv"
        exam_number = 1
        exam_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exam_pages = exam_file.split('_')[-1].split('.')[0]  # Extract p{start}-p{end} from filename

        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            exam_number = df['exam_number'].max() + 1
        else:
            df = pd.DataFrame(columns=['exam_number', 'exam_date', 'exam_pages', 'number_of_questions', 'correct_answers', 'overall_score'])

        new_row = pd.DataFrame({
            'exam_number': [exam_number],
            'exam_date': [exam_date],
            'exam_pages': [exam_pages],
            'number_of_questions': [num_questions],
            'correct_answers': [correct_answers],
            'overall_score': [score]
        })

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(history_file, index=False)
        print(f"Exam history updated in {history_file}")
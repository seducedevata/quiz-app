"""
qa_extractor.py - Scaffolding for structured Q&A and MCQ extraction/formatting
"""

import csv
import json
from typing import List, Dict, Callable, Optional


def extract_questions_from_book(text_content_from_question_book: str) -> List[Dict]:
    """
    Placeholder: Extracts questions from raw text of a question book.
    Returns a list of dicts: [{"id": ..., "text": ...}, ...]
    """
    # TODO: Implement regex/ML-based extraction for real books
    return []


def extract_answers_from_book(
    text_content_from_answer_book: str, list_of_question_ids: List[str]
) -> Dict[str, str]:
    """
    Placeholder: Extracts answers from raw text of an answer/solution book.
    Returns a dict: {question_id: answer_text, ...}
    """
    # TODO: Implement answer extraction logic
    return {}


def link_q_and_a(questions_list: List[Dict], answers_dict: Dict[str, str]) -> List[Dict]:
    """
    Links extracted questions with their corresponding answers.
    Returns a list of dicts: [{"id": ..., "question": ..., "answer": ...}, ...]
    """
    # TODO: Implement linking logic
    return []


def generate_mcq_options_and_format_for_training(
    qa_pair: Dict, llm_api_func_for_distractors: Optional[Callable] = None
) -> str:
    """
    Given a dict with keys (question_text, option_a, option_b, option_c, option_d, correct_letter, explanation),
    format as a single MCQ string for training. If llm_api_func_for_distractors is provided, use it to generate distractors.
    """
    # If using manual MCQ data, just format as expected by train_model
    if all(
        k in qa_pair
        for k in [
            "question_text",
            "option_a",
            "option_b",
            "option_c",
            "option_d",
            "correct_letter",
            "explanation",
        ]
    ):
        return (
            f"Question: {qa_pair['question_text']}\n"
            f"A) {qa_pair['option_a']}\n"
            f"B) {qa_pair['option_b']}\n"
            f"C) {qa_pair['option_c']}\n"
            f"D) {qa_pair['option_d']}\n"
            f"Correct Answer: {qa_pair['correct_letter']}\n"
            f"Explanation: {qa_pair['explanation']}"
        )
    # TODO: If only Q&A, use LLM to generate options
    return ""


def load_manual_mcq_data(filepath: str) -> List[Dict]:
    """
    Loads a manually created MCQ CSV or JSONL file.
    Returns a list of dicts, one per MCQ.
    """
    data = []
    if filepath.lower().endswith(".csv"):
        with open(filepath, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    elif filepath.lower().endswith(".jsonl") or filepath.lower().endswith(".json"):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError("Unsupported file type for MCQ data. Use .csv or .jsonl")
    return data
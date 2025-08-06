#!/usr/bin/env python3
"""
🔍 CRITICAL INVESTIGATION: Find All Missing Questions
Comprehensive search for all generated questions across all storage locations
"""

import sys
import os
import json
import sqlite3
import time
import re
from pathlib import Path
import glob

def find_all_questions():
    """Find all questions across all possible storage locations"""
    
    print("🔍 COMPREHENSIVE QUESTION SEARCH")
    print("=" * 60)
    
    total_questions_found = 0
    
    # 1. Check main database
    print("\n1️⃣ Checking main question history database...")
    db_path = Path("user_data/question_history.sqlite")
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM question_history")
            db_count = cursor.fetchone()[0]
            print(f"   📊 Questions in main database: {db_count}")
            total_questions_found += db_count
            
            if db_count > 0:
                cursor.execute("SELECT id, question_text, topic, generated_at FROM question_history ORDER BY generated_at DESC LIMIT 10")
                recent = cursor.fetchall()
                print(f"   📋 Recent questions:")
                for i, (qid, qtext, topic, generated) in enumerate(recent, 1):
                    print(f"      {i}. {qtext[:60]}... (Topic: {topic})")
            
            conn.close()
        except Exception as e:
            print(f"   ❌ Database error: {e}")
    else:
        print(f"   ❌ Database not found: {db_path}")
    
    # 2. Check current quiz storage
    print("\n2️⃣ Checking current quiz storage...")
    quiz_path = Path("user_data/current_quiz.json")
    if quiz_path.exists():
        try:
            with open(quiz_path, 'r', encoding='utf-8') as f:
                quiz_data = json.load(f)
            
            questions = quiz_data.get("questions", [])
            print(f"   📊 Questions in current quiz: {len(questions)}")
            total_questions_found += len(questions)
            
            if questions:
                print(f"   📋 Current quiz questions:")
                for i, q in enumerate(questions, 1):
                    print(f"      {i}. {q.get('question', 'No question')[:60]}...")
                    
        except Exception as e:
            print(f"   ❌ Current quiz error: {e}")
    else:
        print(f"   ❌ Current quiz not found: {quiz_path}")
    
    # 3. Search for any other JSON files with questions
    print("\n3️⃣ Searching for other JSON files with questions...")
    json_files = []
    
    # Search in common directories
    search_dirs = [".", "user_data", "data", "cache", "src"]
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            pattern = os.path.join(search_dir, "**", "*.json")
            json_files.extend(glob.glob(pattern, recursive=True))
    
    question_files = []
    for json_file in json_files:
        try:
            if os.path.getsize(json_file) > 100:  # Skip tiny files
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if this file contains questions
                questions_in_file = 0
                if isinstance(data, dict):
                    if "questions" in data and isinstance(data["questions"], list):
                        questions_in_file = len(data["questions"])
                    elif "question" in data:
                        questions_in_file = 1
                elif isinstance(data, list):
                    # Check if it's a list of questions
                    for item in data:
                        if isinstance(item, dict) and "question" in item:
                            questions_in_file += 1
                
                if questions_in_file > 0:
                    question_files.append((json_file, questions_in_file))
                    total_questions_found += questions_in_file
                    
        except Exception as e:
            # Skip files that can't be parsed
            pass
    
    if question_files:
        print(f"   📊 Found {len(question_files)} JSON files with questions:")
        for file_path, count in question_files:
            print(f"      📄 {file_path}: {count} questions")
    else:
        print("   📊 No additional JSON files with questions found")
    
    # 4. Check for any other SQLite databases
    print("\n4️⃣ Searching for other SQLite databases...")
    sqlite_files = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            pattern = os.path.join(search_dir, "**", "*.sqlite")
            sqlite_files.extend(glob.glob(pattern, recursive=True))
            pattern = os.path.join(search_dir, "**", "*.db")
            sqlite_files.extend(glob.glob(pattern, recursive=True))
    
    other_db_questions = 0
    for db_file in sqlite_files:
        if db_file != str(db_path):  # Skip the main database we already checked
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Check if it has a questions table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    if 'question' in table.lower():
                        # 🛡️ SECURITY FIX: Validate table name to prevent SQL injection
                        # Only allow alphanumeric characters and underscores in table names
                        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table):
                            print(f"   ⚠️ Skipping invalid table name: {table}")
                            continue

                        # Use parameterized query with quoted identifier for table name
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                            count = cursor.fetchone()[0]
                            if count > 0:
                                print(f"   📄 {db_file} table '{table}': {count} records")
                                other_db_questions += count
                        except sqlite3.Error as e:
                            print(f"   ⚠️ Error querying table {table}: {e}")
                            continue
                
                conn.close()
            except Exception as e:
                # Skip databases that can't be opened
                pass
    
    if other_db_questions > 0:
        total_questions_found += other_db_questions
    else:
        print("   📊 No other databases with questions found")
    
    # 5. Check logs for generation activity
    print("\n5️⃣ Checking logs for question generation activity...")
    log_files = glob.glob("*.log") + glob.glob("**/*.log", recursive=True)
    
    generation_count = 0
    save_count = 0
    error_count = 0
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Count generation events
                generation_count += content.count("questions generated")
                generation_count += content.count("MCQ generated")
                generation_count += content.count("Question generated")
                
                # Count save events
                save_count += content.count("Saved question to history")
                save_count += content.count("save_question called")
                
                # Count errors
                error_count += content.count("Failed to save question")
                error_count += content.count("save_to_question_history")
                
        except Exception as e:
            pass
    
    print(f"   📊 Log analysis:")
    print(f"      🎯 Generation events found: {generation_count}")
    print(f"      💾 Save events found: {save_count}")
    print(f"      ❌ Save errors found: {error_count}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"   🎯 Total questions found across all locations: {total_questions_found}")
    print(f"   📊 Main database: {db_count if 'db_count' in locals() else 0}")
    print(f"   📊 Current quiz: {len(questions) if 'questions' in locals() else 0}")
    print(f"   📊 Other files: {total_questions_found - (db_count if 'db_count' in locals() else 0) - (len(questions) if 'questions' in locals() else 0)}")
    
    if total_questions_found < 10:  # User said they tested "lots"
        print("\n⚠️  WARNING: Very few questions found!")
        print("   This suggests questions are either:")
        print("   1. Not being saved properly")
        print("   2. Being saved to a different location")
        print("   3. Being overwritten")
        print("   4. The save process is failing silently")
    
    return total_questions_found

if __name__ == "__main__":
    total = find_all_questions()
    print(f"\n🎯 Investigation complete. Found {total} total questions.")

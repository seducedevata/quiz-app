#!/usr/bin/env python3
"""
Generate HIGH-QUALITY training data to replace the garbage data
"""

import sys
import os
import json
from pathlib import Path
import time

# Add project to path
sys.path.append(str(Path(__file__).parent / "src"))

from knowledge_app.core.document_processor import AdvancedDocumentProcessor

def generate_quality_training_data():
    print("🚀 GENERATING HIGH-QUALITY TRAINING DATA")
    print("=" * 60)
    print("🗑️  REPLACING GARBAGE DATA WITH ENHANCED PROCESSING")
    print("=" * 60)
    
    # Initialize enhanced processor
    processor = AdvancedDocumentProcessor(
        preserve_educational_content=True,
        use_semantic_chunking=True
    )
    
    # Get all text files
    books_dir = Path("data/uploaded_books")
    text_files = list(books_dir.glob("*.txt"))
    
    print(f"📚 Found {len(text_files)} text files to process")
    
    # Process each file
    all_training_chunks = []
    total_chunks = 0
    
    for i, file_path in enumerate(text_files, 1):
        print(f"\n📖 Processing {i}/{len(text_files)}: {file_path.name}")
        
        try:
            # Extract and process
            content_data = processor.extract_text_with_structure(str(file_path))
            training_chunks = processor.generate_enhanced_training_dataset(
                content_data, chunk_size=500
            )
            
            # Add source information
            for chunk in training_chunks:
                chunk['source_file'] = file_path.name
                chunk['processing_method'] = 'enhanced'
                
            all_training_chunks.extend(training_chunks)
            total_chunks += len(training_chunks)
            
            print(f"   ✅ Generated {len(training_chunks)} quality chunks")
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path.name}: {e}")
            continue
    
    print(f"\n📊 TOTAL QUALITY CHUNKS: {total_chunks}")
    
    # Save in JSONL format (what the training pipeline expects)
    output_dir = Path("data/processed_training")
    output_dir.mkdir(exist_ok=True)
    
    # Create unique filename
    timestamp = int(time.time())
    output_file = output_dir / f"enhanced_training_data_{timestamp}.jsonl"
    
    print(f"\n💾 SAVING TO: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_training_chunks:
            # Format as expected by training pipeline
            training_record = {
                "text": chunk.get('text', ''),
                "metadata": {
                    "source": chunk.get('source_file', 'unknown'),
                    "quality_score": chunk.get('quality_score', 0),
                    "semantic_coherence": chunk.get('semantic_coherence', 0),
                    "method": chunk.get('method', 'enhanced'),
                    "word_count": chunk.get('word_count', 0),
                    "has_math": chunk.get('text', '').count('[MATH:') > 0
                }
            }
            f.write(json.dumps(training_record, ensure_ascii=False) + '\n')
    
    # Also update the legacy combined file
    legacy_file = Path("data/processed_docs/combined_training_dataset.txt")
    print(f"\n📄 UPDATING LEGACY FILE: {legacy_file}")
    
    with open(legacy_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(all_training_chunks):
            f.write(f"Source: {chunk.get('source_file', 'Enhanced Processing')}\n")
            f.write(f"Chunk ID: {i}\n")
            f.write(f"Text: {chunk.get('text', '')}\n\n")
    
    # Create a sample for verification
    sample_file = Path("enhanced_training_sample.json")
    print(f"\n🔍 SAVING SAMPLE: {sample_file}")
    
    sample_chunks = all_training_chunks[:5]  # First 5 chunks
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_chunks": len(all_training_chunks),
            "sample_chunks": sample_chunks,
            "summary": {
                "avg_quality": sum(c.get('quality_score', 0) for c in all_training_chunks) / len(all_training_chunks),
                "avg_coherence": sum(c.get('semantic_coherence', 0) for c in all_training_chunks) / len(all_training_chunks),
                "total_words": sum(c.get('word_count', 0) for c in all_training_chunks),
                "math_chunks": sum(1 for c in all_training_chunks if '[MATH:' in c.get('text', ''))
            }
        }, f, indent=2, ensure_ascii=False)
    
    print("\n🎉 HIGH-QUALITY TRAINING DATA GENERATED!")
    print("=" * 60)
    print(f"✅ Total chunks: {len(all_training_chunks)}")
    print(f"✅ JSONL file: {output_file}")
    print(f"✅ Legacy file updated: {legacy_file}")
    print(f"✅ Sample file: {sample_file}")
    print("=" * 60)
    print("🚀 NOW YOUR MODEL WILL TRAIN ON QUALITY DATA INSTEAD OF GARBAGE!")

if __name__ == "__main__":
    generate_quality_training_data() 
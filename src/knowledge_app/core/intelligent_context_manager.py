"""
🧠 Intelligent Context Manager for DeepSeek-R1 14B (128K Token Context)

This system intelligently manages DeepSeek-R1 14B's massive 128K token context window
to maximize document processing efficiency and training data extraction quality.

Features:
- Dynamic token counting and optimization
- Smart document chunking with context preservation
- Hierarchical content caching and offloading
- Context window utilization monitoring
- Intelligent content prioritization
- Memory-efficient processing pipelines
"""

import json
import logging
import asyncio
import hashlib
import time
import re
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque, defaultdict
import tiktoken
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TokenMetrics:
    """Token usage metrics for optimization"""
    total_tokens: int = 0
    content_tokens: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    efficiency_ratio: float = 0.0
    context_utilization: float = 0.0

@dataclass
class ContentChunk:
    """Optimized content chunk with metadata"""
    id: str
    content: str
    token_count: int
    priority_score: float
    content_type: str  # 'text', 'code', 'math', 'table', 'reference'
    semantic_hash: str
    dependencies: List[str] = field(default_factory=list)
    extracted_concepts: List[str] = field(default_factory=list)
    processing_timestamp: float = field(default_factory=time.time)

@dataclass
class ContextWindow:
    """128K context window manager"""
    max_tokens: int = 128000  # DeepSeek-R1 14B context limit
    reserved_tokens: int = 8000  # Reserve for prompts and responses
    available_tokens: int = field(init=False)
    current_usage: int = 0
    context_utilization: float = 0.0  # Track utilization percentage
    chunks: List[ContentChunk] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    
    def __post_init__(self):
        self.available_tokens = self.max_tokens - self.reserved_tokens

@dataclass
class ProcessingSession:
    """Document processing session state"""
    session_id: str
    documents: List[Dict[str, Any]]
    context_window: ContextWindow
    cache: Dict[str, Any] = field(default_factory=dict)
    metrics: TokenMetrics = field(default_factory=TokenMetrics)
    processing_queue: deque = field(default_factory=deque)
    completed_chunks: List[str] = field(default_factory=list)
    failed_chunks: List[str] = field(default_factory=list)

class IntelligentContextManager:
    """
    🧠 Intelligent Context Manager for DeepSeek-R1 14B
    
    Maximizes the 128K token context window efficiency through:
    1. Smart content analysis and prioritization
    2. Dynamic chunk sizing based on content complexity
    3. Hierarchical caching with LRU eviction
    4. Context-aware document processing
    5. Real-time token optimization
    """
    
    def __init__(self, model_name: str = "deepseek-r1:14b"):
        self.model_name = model_name
        self.tokenizer = self._initialize_tokenizer()
        self.context_window = ContextWindow()
        self.content_cache = {}  # Semantic content cache
        self.processing_cache = {}  # Processing results cache
        self.session_history = {}
        self.optimization_stats = {
            "total_sessions": 0,
            "average_context_utilization": 0.0,
            "cache_hit_rate": 0.0,
            "processing_efficiency": 0.0,
            "token_savings": 0
        }
        
        logger.info(f"🧠 Intelligent Context Manager initialized for {model_name}")
        logger.info(f"📊 Context window: {self.context_window.max_tokens:,} tokens")
        logger.info(f"💾 Available for content: {self.context_window.available_tokens:,} tokens")
    
    def _initialize_tokenizer(self) -> tiktoken.Encoding:
        """Initialize tokenizer for accurate token counting"""
        try:
            # Use GPT-4 tokenizer as approximation for DeepSeek
            return tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load model-specific tokenizer: {e}")
            # Fallback to cl100k_base encoding
            return tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Accurately count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"⚠️ Token counting failed: {e}")
            # Rough estimation: ~4 chars per token
            return len(text) // 4
    
    def create_processing_session(self, documents: List[Dict[str, Any]]) -> ProcessingSession:
        """Create a new document processing session"""
        session_id = f"deepseek_session_{int(time.time())}_{hash(str(documents)) % 10000}"
        
        session = ProcessingSession(
            session_id=session_id,
            documents=documents,
            context_window=ContextWindow()
        )
        
        self.session_history[session_id] = session
        self.optimization_stats["total_sessions"] += 1
        
        logger.info(f"🆕 Created processing session: {session_id}")
        logger.info(f"📄 Documents to process: {len(documents)}")
        
        return session
    
    def analyze_content_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze content complexity for intelligent chunking"""
        tokens = self.count_tokens(content)
        
        # Content type detection
        content_types = []
        if re.search(r'```|`[^`]+`', content):
            content_types.append('code')
        if re.search(r'\$[^$]+\$|\\[a-zA-Z]+\{', content):
            content_types.append('math')
        if re.search(r'\|[^|]+\|[^|]+\|', content):
            content_types.append('table')
        if re.search(r'\[[0-9]+\]|\([A-Z][a-z]+\s+et\s+al\.,?\s+[0-9]{4}\)', content):
            content_types.append('reference')
        if not content_types:
            content_types.append('text')
        
        # Calculate complexity score
        complexity_factors = {
            'length': min(tokens / 1000, 1.0),
            'technical_terms': len(re.findall(r'\b[A-Z]{2,}|\b[a-z]+[A-Z][a-z]*\b', content)) / 100,
            'mathematical': len(re.findall(r'[∑∫∂∇=<>≤≥≠]|\$[^$]+\$', content)) / 10,
            'references': len(re.findall(r'\[[0-9]+\]', content)) / 20,
            'code_blocks': len(re.findall(r'```[\s\S]*?```', content)) / 5
        }
        
        complexity_score = sum(min(score, 1.0) for score in complexity_factors.values()) / len(complexity_factors)
        
        return {
            'token_count': tokens,
            'content_types': content_types,
            'complexity_score': complexity_score,
            'complexity_factors': complexity_factors,
            'estimated_processing_tokens': int(tokens * (1.2 + complexity_score * 0.5))
        }
    
    def create_intelligent_chunks(self, content: str, document_name: str) -> List[ContentChunk]:
        """Create optimized content chunks based on context window and content analysis"""
        analysis = self.analyze_content_complexity(content)
        
        # Dynamic chunk sizing based on complexity
        base_chunk_size = 8000  # Conservative base size for complex content
        if analysis['complexity_score'] < 0.3:
            base_chunk_size = 15000  # Larger chunks for simple content
        elif analysis['complexity_score'] > 0.7:
            base_chunk_size = 5000   # Smaller chunks for complex content
        
        chunks = []
        content_length = len(content)
        overlap_size = max(200, int(base_chunk_size * 0.1))  # 10% overlap
        
        # Smart boundary detection for better chunking
        start = 0
        chunk_id = 0
        
        while start < content_length:
            end = min(start + base_chunk_size, content_length)
            chunk_content = content[start:end]
            
            # Find natural boundaries
            if end < content_length:
                # Look for section boundaries first
                section_boundaries = [
                    chunk_content.rfind('\n# '),
                    chunk_content.rfind('\n## '),
                    chunk_content.rfind('\n### '),
                    chunk_content.rfind('\n\n'),
                    chunk_content.rfind('. '),
                    chunk_content.rfind('? '),
                    chunk_content.rfind('! ')
                ]
                
                best_boundary = max([b for b in section_boundaries if b > len(chunk_content) * 0.7], default=-1)
                
                if best_boundary > 0:
                    chunk_content = chunk_content[:best_boundary + 1]
                    end = start + best_boundary + 1
            
            if len(chunk_content.strip()) < 100:  # Skip tiny chunks
                start = end - overlap_size if end < content_length else content_length
                continue
            
            # Calculate priority score
            priority_score = self._calculate_chunk_priority(chunk_content, analysis)
            
            # Create semantic hash for deduplication
            semantic_hash = hashlib.sha256(
                re.sub(r'\s+', ' ', chunk_content.lower().strip()).encode()
            ).hexdigest()[:16]
            
            # Extract concepts for this chunk
            concepts = self._extract_chunk_concepts(chunk_content)
            
            chunk = ContentChunk(
                id=f"{document_name}_chunk_{chunk_id}",
                content=chunk_content.strip(),
                token_count=self.count_tokens(chunk_content),
                priority_score=priority_score,
                content_type=analysis['content_types'][0],
                semantic_hash=semantic_hash,
                extracted_concepts=concepts
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move start position with overlap
            start = end - overlap_size if end < content_length else content_length
            
            # Prevent infinite loops
            if start >= end:
                break
        
        logger.info(f"📝 Created {len(chunks)} intelligent chunks for {document_name}")
        logger.info(f"🎯 Average chunk size: {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens")
        
        return chunks
    
    def _calculate_chunk_priority(self, content: str, analysis: Dict[str, Any]) -> float:
        """Calculate priority score for chunk processing order"""
        factors = {
            'complexity': analysis['complexity_score'],
            'length': min(len(content) / 10000, 1.0),
            'information_density': len(re.findall(r'\b[A-Z][a-z]*\b', content)) / max(len(content.split()), 1),
            'technical_content': len(re.findall(r'\b(?:algorithm|method|system|process|analysis|result)\b', content.lower())) / 10,
            'mathematical_content': len(re.findall(r'[∑∫∂∇=<>≤≥≠]|\$[^$]+\$', content)) / 5
        }
        
        # Weight the factors
        weights = {
            'complexity': 0.3,
            'length': 0.2,
            'information_density': 0.2,
            'technical_content': 0.2,
            'mathematical_content': 0.1
        }
        
        priority = sum(min(factors[k], 1.0) * weights[k] for k in factors)
        return min(priority, 1.0)
    
    def _extract_chunk_concepts(self, content: str) -> List[str]:
        """Extract key concepts from chunk content"""
        # Technical terms
        technical_terms = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', content)
        
        # Mathematical concepts
        math_concepts = re.findall(r'\b(?:equation|formula|theorem|proof|algorithm|method|function)\b', content.lower())
        
        # Domain-specific terms
        domain_terms = re.findall(r'\b(?:model|system|process|analysis|data|result|conclusion|hypothesis)\b', content.lower())
        
        # Combine and deduplicate
        all_concepts = list(set(technical_terms + math_concepts + domain_terms))
        
        # Filter and rank by frequency
        concept_counts = {}
        for concept in all_concepts:
            concept_counts[concept] = content.lower().count(concept.lower())
        
        # Return top concepts
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, count in sorted_concepts[:10] if count >= 2]
    
    def optimize_context_window(self, session: ProcessingSession, new_chunks: List[ContentChunk]) -> List[ContentChunk]:
        """Optimize context window usage with smart caching and prioritization"""
        logger.info(f"🔧 Optimizing context window for {len(new_chunks)} new chunks")
        
        # Calculate total tokens needed
        total_new_tokens = sum(chunk.token_count for chunk in new_chunks)
        current_tokens = session.context_window.current_usage
        
        logger.info(f"📊 Current usage: {current_tokens:,} tokens")
        logger.info(f"📊 New chunks need: {total_new_tokens:,} tokens")
        logger.info(f"📊 Available: {session.context_window.available_tokens:,} tokens")
        
        # Check if we need to optimize
        if current_tokens + total_new_tokens <= session.context_window.available_tokens:
            # No optimization needed
            session.context_window.chunks.extend(new_chunks)
            session.context_window.current_usage += total_new_tokens
            logger.info("✅ No optimization needed - sufficient context window space")
            return new_chunks
        
        # Need optimization - implement LRU cache eviction and smart prioritization
        logger.info("🎯 Context window optimization required")
        
        # Sort existing chunks by priority and age
        existing_chunks = session.context_window.chunks.copy()
        existing_chunks.sort(key=lambda x: (x.priority_score, -x.processing_timestamp), reverse=True)
        
        # Sort new chunks by priority
        new_chunks.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Calculate optimal chunk selection
        optimized_chunks = []
        total_tokens = 0
        
        # Prioritize high-value new chunks
        for chunk in new_chunks:
            if total_tokens + chunk.token_count <= session.context_window.available_tokens:
                optimized_chunks.append(chunk)
                total_tokens += chunk.token_count
            else:
                # Cache the chunk for later processing
                self._cache_chunk(chunk, session)
        
        # Fill remaining space with high-priority existing chunks
        for chunk in existing_chunks:
            if total_tokens + chunk.token_count <= session.context_window.available_tokens:
                if chunk not in optimized_chunks:
                    optimized_chunks.append(chunk)
                    total_tokens += chunk.token_count
            else:
                # Move to cache
                self._cache_chunk(chunk, session)
        
        # Update context window
        session.context_window.chunks = optimized_chunks
        session.context_window.current_usage = total_tokens
        
        # Update metrics
        session.context_window.context_utilization = total_tokens / session.context_window.available_tokens
        
        logger.info(f"🎯 Optimization complete:")
        logger.info(f"   📝 Chunks in context: {len(optimized_chunks)}")
        logger.info(f"   💾 Chunks cached: {len(session.cache)}")
        logger.info(f"   📊 Context utilization: {session.context_window.context_utilization:.1%}")
        
        return [chunk for chunk in optimized_chunks if chunk in new_chunks]
    
    def _cache_chunk(self, chunk: ContentChunk, session: ProcessingSession):
        """Cache chunk with semantic deduplication"""
        # Check for semantic duplicates
        for cached_hash in session.cache:
            if chunk.semantic_hash == cached_hash:
                session.context_window.cache_hits += 1
                logger.debug(f"🎯 Semantic duplicate found for chunk {chunk.id}")
                return
        
        # Cache the chunk
        session.cache[chunk.semantic_hash] = {
            'chunk': chunk,
            'cached_at': time.time(),
            'access_count': 0
        }
        session.context_window.cache_misses += 1
        
        # Implement LRU eviction if cache gets too large
        if len(session.cache) > 1000:  # Arbitrary limit
            self._evict_lru_chunks(session)
    
    def _evict_lru_chunks(self, session: ProcessingSession, max_size: int = 800):
        """Evict least recently used chunks from cache"""
        cache_items = list(session.cache.items())
        cache_items.sort(key=lambda x: (x[1]['access_count'], x[1]['cached_at']))
        
        # Remove oldest, least accessed items
        items_to_remove = len(cache_items) - max_size
        for i in range(items_to_remove):
            hash_key = cache_items[i][0]
            del session.cache[hash_key]
        
        logger.info(f"🗑️ Evicted {items_to_remove} chunks from cache")
    
    def prepare_context_for_processing(self, session: ProcessingSession, chunks: List[ContentChunk]) -> str:
        """Prepare optimized context for DeepSeek processing - SIMPLIFIED"""
        logger.info(f"📋 Preparing context for {len(chunks)} chunks")
        
        # Sort chunks by priority and dependencies
        sorted_chunks = self._sort_chunks_by_dependencies(chunks)
        
        # Build minimal context - just the content
        context_parts = []
        
        # Add chunks with minimal separators
        for i, chunk in enumerate(sorted_chunks):
            context_parts.append(f"\n--- Document {i+1} ---")
            context_parts.append(chunk.content)
        
        context = "\n".join(context_parts)
        
        # Calculate final token count
        try:
            token_count = len(self.tokenizer.encode(context))
        except:
            token_count = len(context) // 4  # Fallback estimate
            
        logger.info(f"📊 Prepared context: {token_count:,} tokens ({token_count/self.max_context_tokens*100:.1f}% of available)")
        
        return context
        
        context = '\n'.join(context_parts)
        
        # Verify token count
        total_tokens = self.count_tokens(context)
        logger.info(f"📊 Prepared context: {total_tokens:,} tokens ({total_tokens/session.context_window.available_tokens:.1%} of available)")
        
        return context
    
    def _sort_chunks_by_dependencies(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        """Sort chunks considering dependencies and priority"""
        # Simple topological sort based on priority and content relationships
        chunks_with_refs = []
        chunks_without_refs = []
        
        for chunk in chunks:
            # Check if chunk references other chunks
            has_references = any(
                other_chunk.id in chunk.content or
                any(concept in chunk.content.lower() for concept in other_chunk.extracted_concepts)
                for other_chunk in chunks if other_chunk != chunk
            )
            
            if has_references:
                chunks_with_refs.append(chunk)
            else:
                chunks_without_refs.append(chunk)
        
        # Sort each group by priority
        chunks_without_refs.sort(key=lambda x: x.priority_score, reverse=True)
        chunks_with_refs.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Return foundational chunks first, then referencing chunks
        return chunks_without_refs + chunks_with_refs
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get context manager optimization statistics"""
        if self.optimization_stats["total_sessions"] > 0:
            self.optimization_stats["cache_hit_rate"] = sum(
                session.context_window.cache_hits / max(
                    session.context_window.cache_hits + session.context_window.cache_misses, 1
                ) for session in self.session_history.values()
            ) / len(self.session_history)
            
            self.optimization_stats["average_context_utilization"] = sum(
                session.context_window.context_utilization 
                for session in self.session_history.values()
            ) / len(self.session_history)
        
        return self.optimization_stats.copy()
    
    def cleanup_session(self, session_id: str):
        """Clean up session data to free memory"""
        if session_id in self.session_history:
            session = self.session_history[session_id]
            
            # Log final stats
            logger.info(f"🧹 Cleaning up session {session_id}")
            logger.info(f"   📊 Final context utilization: {session.context_window.context_utilization:.1%}")
            logger.info(f"   💾 Cache hit rate: {session.context_window.cache_hits / max(session.context_window.cache_hits + session.context_window.cache_misses, 1):.1%}")
            logger.info(f"   ✅ Completed chunks: {len(session.completed_chunks)}")
            logger.info(f"   ❌ Failed chunks: {len(session.failed_chunks)}")
            
            # Clear session data
            del self.session_history[session_id]
            logger.info(f"✅ Session {session_id} cleaned up")

def get_intelligent_context_manager() -> IntelligentContextManager:
    """Get singleton instance of intelligent context manager"""
    if not hasattr(get_intelligent_context_manager, '_instance'):
        get_intelligent_context_manager._instance = IntelligentContextManager()
    return get_intelligent_context_manager._instance

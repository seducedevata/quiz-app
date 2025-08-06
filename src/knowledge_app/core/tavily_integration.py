"""
Tavily Search API Integration for Knowledge App
Provides real-time web search capabilities for enhanced quiz generation
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TavilySearchResult:
    """Structured representation of Tavily search results"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time: float
    success: bool
    error: Optional[str] = None

class TavilyIntegration:
    """
    Tavily Search API integration for enhanced quiz generation
    
    Features:
    - Real-time web search for current information
    - Content extraction from specific URLs
    - Domain-specific search filtering
    - Error handling and fallback mechanisms
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        self._client = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize the Tavily client"""
        try:
            if not self.api_key:
                logger.warning("❌ Tavily API key not provided")
                return False
                
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self.api_key)
            
            # Test the connection with a simple query
            test_result = self._client.search("test", max_results=1)
            if test_result and 'results' in test_result:
                self._initialized = True
                logger.info("✅ Tavily API initialized successfully")
                return True
            else:
                logger.error("❌ Tavily API test query failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Tavily API: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Tavily API is available and initialized"""
        return self._initialized and self._client is not None
    
    def search_for_topic(self, topic: str, max_results: int = 5) -> TavilySearchResult:
        """
        Search for current information about a topic
        
        Args:
            topic: The topic to search for
            max_results: Maximum number of results to return
            
        Returns:
            TavilySearchResult with search results and metadata
        """
        if not self.is_available():
            return TavilySearchResult(
                query=topic,
                results=[],
                total_results=0,
                search_time=0.0,
                success=False,
                error="Tavily API not initialized"
            )
        
        try:
            import time
            start_time = time.time()
            
            # Enhanced search query for educational content
            enhanced_query = f"latest research {topic} educational academic scientific"
            
            logger.info(f"🔍 Searching Tavily for: '{enhanced_query}'")
            
            response = self._client.search(
                query=enhanced_query,
                max_results=max_results,
                include_domains=["edu", "org", "gov"],  # Focus on educational sources
                exclude_domains=["reddit.com", "quora.com"],  # Exclude forums
                include_raw_content=True
            )
            
            search_time = time.time() - start_time
            
            if response and 'results' in response:
                logger.info(f"✅ Found {len(response['results'])} results in {search_time:.2f}s")
                return TavilySearchResult(
                    query=topic,
                    results=response['results'],
                    total_results=len(response['results']),
                    search_time=search_time,
                    success=True
                )
            else:
                logger.warning(f"⚠️ No results found for '{topic}'")
                return TavilySearchResult(
                    query=topic,
                    results=[],
                    total_results=0,
                    search_time=search_time,
                    success=False,
                    error="No results found"
                )
                
        except Exception as e:
            logger.error(f"❌ Tavily search failed: {e}")
            return TavilySearchResult(
                query=topic,
                results=[],
                total_results=0,
                search_time=0.0,
                success=False,
                error=str(e)
            )
    
    def extract_content(self, urls: List[str]) -> Dict[str, str]:
        """
        Extract content from specific URLs
        
        Args:
            urls: List of URLs to extract content from
            
        Returns:
            Dictionary mapping URLs to their extracted content
        """
        if not self.is_available():
            return {}
        
        try:
            logger.info(f"📄 Extracting content from {len(urls)} URLs")
            
            response = self._client.extract(urls)
            
            content_map = {}
            if response and 'results' in response:
                for result in response['results']:
                    url = result.get('url', '')
                    content = result.get('raw_content', '')
                    if url and content:
                        content_map[url] = content
                        
            logger.info(f"✅ Extracted content from {len(content_map)} URLs")
            return content_map
            
        except Exception as e:
            logger.error(f"❌ Content extraction failed: {e}")
            return {}
    
    def get_current_context(self, topic: str) -> str:
        """
        Get current context about a topic for quiz generation
        
        Args:
            topic: The topic to get context for
            
        Returns:
            Formatted context string for quiz generation
        """
        search_result = self.search_for_topic(topic, max_results=3)
        
        if not search_result.success:
            return f"No current information available for '{topic}'"
        
        context_parts = [f"Current information about '{topic}':"]
        
        for i, result in enumerate(search_result.results[:3], 1):
            title = result.get('title', 'Untitled')
            content = result.get('content', '')
            url = result.get('url', '')
            
            if content:
                # Truncate content to reasonable length
                truncated_content = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"\n{i}. {title}")
                context_parts.append(f"   Source: {url}")
                context_parts.append(f"   Content: {truncated_content}")
        
        return "\n".join(context_parts)
    
    def set_api_key(self, api_key: str) -> bool:
        """
        Set a new API key and reinitialize
        
        Args:
            api_key: The new Tavily API key
            
        Returns:
            True if initialization successful, False otherwise
        """
        self.api_key = api_key
        self._initialized = False
        self._client = None
        return self.initialize()
    
    def get_status(self):
        """Get the current status of Tavily integration"""
        if not self.api_key:
            return {
                "status": "no_api_key",
                "api_key_configured": False,
                "message": "Tavily API key not configured"
            }
        
        if not self._client:
            return {
                "status": "not_initialized",
                "api_key_configured": True,
                "message": "Tavily client not initialized"
            }
        
        return {
            "status": "ready",
            "api_key_configured": True,
            "message": "Tavily is ready for searches"
        }

    def test_connection(self, test_query: str = "test") -> Dict[str, Any]:
        """Test the Tavily API connection with a simple query"""
        try:
            if not self._client:
                return {
                    "success": False,
                    "error": "Tavily client not initialized"
                }
            
            # Perform a simple search to test the connection
            result = self.search_for_topic(test_query, max_results=1)
            
            if result.success and len(result.results) > 0:
                return {
                    "success": True,
                    "message": f"Connection successful, found {len(result.results)} results",
                    "test_query": test_query
                }
            elif result.success:
                return {
                    "success": True,
                    "message": "Connection successful but no results found",
                    "test_query": test_query
                }
            else:
                return {
                    "success": False,
                    "error": result.error or "Unknown error",
                    "test_query": test_query
                }
                
        except Exception as e:
            logger.error(f"❌ Tavily connection test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "test_query": test_query
            }

# Global instance
_tavily_integration = None

def get_tavily_integration() -> TavilyIntegration:
    """Get the global Tavily integration instance"""
    global _tavily_integration
    if _tavily_integration is None:
        # Get API key from secure storage or environment
        api_key = None
        try:
            from .secure_api_key_manager import SecureApiKeyManager
            key_manager = SecureApiKeyManager()
            api_key = key_manager.get_api_key('tavily')
        except Exception:
            # Fallback to environment variable
            api_key = os.environ.get('TAVILY_API_KEY')
        
        _tavily_integration = TavilyIntegration(api_key=api_key)
        _tavily_integration.initialize()
    return _tavily_integration

def initialize_tavily(api_key: Optional[str] = None) -> bool:
    """Initialize Tavily with optional API key"""
    tavily = get_tavily_integration()
    if api_key:
        return tavily.set_api_key(api_key)
    else:
        return tavily.initialize()

def is_tavily_available() -> bool:
    """Check if Tavily is available"""
    return get_tavily_integration().is_available()

def search_tavily(topic: str, max_results: int = 5) -> TavilySearchResult:
    """Search Tavily for a topic"""
    return get_tavily_integration().search_for_topic(topic, max_results)

def get_tavily_context(topic: str) -> str:
    """Get Tavily context for a topic"""
    return get_tavily_integration().get_current_context(topic)

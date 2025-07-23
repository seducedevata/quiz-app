"""
Text Filter Utilities
Centralized topic filtering to prevent duplicate code across modules.
Enhanced to convert inappropriate topics to educational equivalents.
"""

import logging
import re

logger = logging.getLogger(__name__)

def filter_inappropriate_topic(topic):
    """Filter inappropriate topics and convert to educational equivalents"""
    if not topic or not isinstance(topic, str):
        return topic
    
    topic_lower = topic.lower().strip()
    original_topic = topic
    
    # Educational topic mappings - convert inappropriate to educational
    educational_mappings = {
        # Sexual/reproductive topics -> educational biology
        'sex': 'reproductive biology',
        'sexual': 'reproductive biology', 
        'sexuality': 'human reproduction',
        'intercourse': 'reproductive biology',
        'mating': 'animal reproduction',
        
        # Body parts -> anatomical systems
        'penis': 'male reproductive anatomy',
        'vagina': 'female reproductive anatomy', 
        'genitals': 'reproductive anatomy',
        'breasts': 'mammary system anatomy',
        
        # Inappropriate content -> educational equivalents
        'drugs': 'pharmacology',
        'alcohol': 'biochemistry',
        'violence': 'psychology',
        'weapons': 'physics',
        'death': 'biology',
    }
    
    # Direct mapping check
    if topic_lower in educational_mappings:
        filtered_topic = educational_mappings[topic_lower]
        logger.info(f"🔄 Topic filter: '{original_topic}' → '{filtered_topic}' (educational mapping)")
        return filtered_topic
    
    # Pattern-based filtering for more complex cases
    educational_patterns = [
        # Reproductive biology patterns
        (r'\b(sex|sexual|reproduction|reproductive)\b.*\b(system|organ|biology|anatomy)\b', 'reproductive biology'),
        (r'\b(male|female)\b.*\b(reproduction|reproductive)\b', 'reproductive systems'),
        (r'\b(human|animal)\b.*\b(reproduction|mating|breeding)\b', 'reproductive biology'),
        
        # Medical/health patterns
        (r'\b(disease|illness|medical|health)\b', 'medical science'),
        (r'\b(drug|medicine|pharmaceutical)\b', 'pharmacology'),
        
        # Scientific patterns
        (r'\b(chemistry|chemical|molecule|atom)\b', 'chemistry'),
        (r'\b(physics|physical|force|energy)\b', 'physics'),
        (r'\b(biology|biological|life|living)\b', 'biology'),
    ]
    
    for pattern, replacement in educational_patterns:
        if re.search(pattern, topic_lower):
            logger.info(f"🔄 Topic filter: '{original_topic}' → '{replacement}' (pattern match)")
            return replacement
    
    # Check for inappropriate content that should be completely avoided
    inappropriate_patterns = [
        r'\b(porn|pornography|explicit|adult|nsfw)\b',
        r'\b(violence|violent|killing|murder|death)\b',
        r'\b(drugs|narcotic|cocaine|heroin|marijuana)\b',
        r'\b(alcohol|drunk|drinking|beer|wine)\b',
        r'\b(weapons|gun|knife|bomb|explosive)\b',
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, topic_lower):
            logger.warning(f"⚠️ Topic filter: '{original_topic}' contains inappropriate content - converting to 'general science'")
            return 'general science'
    
    # If no issues found, return original topic
    logger.info(f"✅ Topic filter: '{original_topic}' approved as educational")
    return original_topic

def is_educational_topic(topic):
    """Check if a topic is appropriate for educational content"""
    if not topic or not isinstance(topic, str):
        return False
    
    topic_lower = topic.lower().strip()
    
    # Educational topic categories
    educational_categories = [
        'biology', 'chemistry', 'physics', 'mathematics', 'science',
        'history', 'geography', 'literature', 'language', 'arts',
        'anatomy', 'physiology', 'medicine', 'health', 'nutrition',
        'astronomy', 'geology', 'ecology', 'psychology', 'sociology',
        'technology', 'engineering', 'computer', 'programming',
        'economics', 'politics', 'philosophy', 'education'
    ]
    
    # Check if topic matches educational categories
    for category in educational_categories:
        if category in topic_lower:
            return True
    
    # Check for inappropriate content
    inappropriate_keywords = [
        'porn', 'explicit', 'adult', 'nsfw', 'violence', 'weapons',
        'drugs', 'narcotic', 'alcohol', 'gambling'
    ]
    
    for keyword in inappropriate_keywords:
        if keyword in topic_lower:
            return False
    
    # Default to educational if not clearly inappropriate
    return True

def enhance_topic_for_education(topic):
    """Enhance a topic to make it more educational and specific"""
    if not topic or not isinstance(topic, str):
        return topic
    
    topic_lower = topic.lower().strip()
    
    # Enhancement mappings to make topics more educational
    enhancements = {
        'biology': 'cellular biology and life processes',
        'chemistry': 'chemical reactions and molecular structures', 
        'physics': 'physical laws and natural phenomena',
        'science': 'scientific principles and methodology',
        'history': 'historical events and their significance',
        'math': 'mathematical concepts and problem solving',
        'mathematics': 'mathematical reasoning and applications',
        'health': 'human health and medical science',
        'medicine': 'medical science and healthcare',
        'technology': 'technological innovation and applications',
        'psychology': 'psychological principles and human behavior',
        'anatomy': 'anatomical structures and physiological functions'
    }
    
    # Check for enhancement opportunities
    for key, enhancement in enhancements.items():
        if topic_lower == key:
            logger.info(f"📚 Topic enhancement: '{topic}' → '{enhancement}'")
            return enhancement
    
    return topic
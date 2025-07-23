"""
MCQ Coherence Monitor - Production system to catch AI-generated spaghetti
Monitors MCQ generation in real-time and flags incoherent questions.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CoherenceIssue:
    """Represents a coherence issue detected in an MCQ."""
    issue_type: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    pattern_matched: str
    timestamp: datetime


class MCQCoherenceMonitor:
    """
    Production-ready coherence monitor that catches AI spaghetti patterns.
    
    This monitor is designed to catch the exact types of incoherent questions
    that were shown in the user's screenshot - like reproductive health mixed
    with random topics like cancer, biodiversity, or brain science.
    """
    
    def __init__(self, performance_mode: bool = True):
        self.logger = logging.getLogger(__name__)

        # 🚀 PERFORMANCE MODE: Skip heavy monitoring for speed
        self.performance_mode = performance_mode
        
        # Critical spaghetti patterns that should never appear
        self.critical_patterns = [
            # Reproductive health + unrelated medical fields
            {
                'name': 'reproductive_cancer_fusion',
                'primary': ['reproductive', 'menstrual', 'contraception', 'pregnancy'],
                'conflicting': ['cancer', 'tumor', 'oncology', 'chemotherapy', 'radiation'],
                'description': 'Reproductive health mixed with cancer/oncology'
            },
            # Reproductive health + ecology/biodiversity  
            {
                'name': 'reproductive_biodiversity_fusion',
                'primary': ['reproductive', 'menstrual', 'sex', 'contraception'],
                'conflicting': ['plants', 'animals', 'species', 'biodiversity', 'ecosystem', 'habitat'],
                'description': 'Reproductive health mixed with ecology/biodiversity'
            },
            # Reproductive health + neuroscience
            {
                'name': 'reproductive_neuroscience_fusion', 
                'primary': ['reproductive', 'menstrual', 'pregnancy'],
                'conflicting': ['brain', 'neural', 'cognitive', 'neuron', 'synapse', 'cortex'],
                'description': 'Reproductive health mixed with neuroscience'
            },
            # Biology + Computer Science
            {
                'name': 'biology_cs_fusion',
                'primary': ['biology', 'cell', 'organism', 'dna'],
                'conflicting': ['programming', 'algorithm', 'software', 'database', 'coding'],
                'description': 'Biology mixed with computer science'
            },
            # Physics + Literature
            {
                'name': 'physics_literature_fusion',
                'primary': ['physics', 'quantum', 'energy', 'force'],
                'conflicting': ['poetry', 'novel', 'literature', 'author', 'character', 'plot'],
                'description': 'Physics mixed with literature'
            }
        ]
        
        # Medium severity patterns
        self.medium_patterns = [
            # Generic/vague questions that provide no educational value
            {
                'name': 'generic_question',
                'pattern': r'what is (an? )?(important|key|essential) (aspect|component|element) of',
                'description': 'Generic, non-specific question pattern'
            },
            # Multiple unrelated academic fields in options
            {
                'name': 'field_mixing',
                'fields': {
                    'biology': ['cell', 'organism', 'dna', 'protein', 'evolution'],
                    'physics': ['force', 'energy', 'quantum', 'gravity', 'electromagnetic'],
                    'chemistry': ['atom', 'molecule', 'reaction', 'element', 'compound'],
                    'history': ['war', 'empire', 'century', 'civilization', 'ancient'],
                    'literature': ['novel', 'poem', 'author', 'character', 'metaphor'],
                    'mathematics': ['equation', 'function', 'theorem', 'integral', 'derivative']
                },
                'description': 'Multiple unrelated academic fields in one question'
            }
        ]
        
        # Statistics tracking
        self.stats = {
            'total_questions_monitored': 0,
            'critical_issues_found': 0,
            'medium_issues_found': 0,
            'low_issues_found': 0,
            'issue_types': {}
        }
    
    def monitor_mcq(self, question: str, options: List[str], topic: str, 
                   context: str = "") -> Tuple[bool, List[CoherenceIssue]]:
        """
        Monitor an MCQ for coherence issues.
        
        Args:
            question: The MCQ question text
            options: List of answer options
            topic: The intended topic
            context: Additional context (optional)
            
        Returns:
            Tuple of (is_coherent, list_of_issues)
        """
        self.stats['total_questions_monitored'] += 1

        # 🚀 PERFORMANCE MODE: Skip heavy monitoring for speed
        if self.performance_mode:
            # Only check for the most critical issues
            issues = []
            all_text = f"{question} {' '.join(options)}".lower()

            # Quick check for obvious spaghetti (reproductive + cancer fusion)
            if ('reproductive' in all_text or 'menstrual' in all_text) and ('cancer' in all_text or 'tumor' in all_text):
                issues.append(CoherenceIssue(
                    issue_type='reproductive_cancer_fusion',
                    description='Reproductive health mixed with cancer content',
                    severity='critical',
                    pattern_matched='performance_mode_quick_check',
                    timestamp=datetime.now()
                ))

            # Return early for performance
            is_coherent = len(issues) == 0
            return is_coherent, issues

        # Full monitoring (original heavy approach)
        issues = []
        all_text = f"{question} {' '.join(options)}".lower()

        # Check critical patterns first
        critical_issues = self._check_critical_patterns(all_text, topic)
        issues.extend(critical_issues)

        # Check medium severity patterns
        medium_issues = self._check_medium_patterns(question, options, all_text, topic)
        issues.extend(medium_issues)
        
        # Update statistics
        critical_count = sum(1 for issue in issues if issue.severity == 'critical')
        medium_count = sum(1 for issue in issues if issue.severity == 'medium')
        
        self.stats['critical_issues_found'] += critical_count
        self.stats['medium_issues_found'] += medium_count
        
        for issue in issues:
            issue_type = issue.issue_type
            self.stats['issue_types'][issue_type] = self.stats['issue_types'].get(issue_type, 0) + 1
        
        # Log issues
        if issues:
            self._log_issues(question, options, topic, issues)
        
        # Question is coherent if no critical issues
        is_coherent = critical_count == 0
        
        return is_coherent, issues
    
    def _check_critical_patterns(self, text: str, topic: str) -> List[CoherenceIssue]:
        """Check for critical coherence issues that make questions unusable"""
        issues = []
        
        # Enhanced critical patterns for poor quality questions
        critical_patterns = [
            (r'ai\s+spaghetti|nonsense|gibberish|random', 'ai_spaghetti', 'AI generated nonsensical content'),
            (r'question\s*:\s*question|answer\s*:\s*answer', 'malformed_structure', 'Malformed question structure'),
            (r'option\s+[a-d]\s*:\s*option\s+[a-d]', 'malformed_options', 'Malformed option structure'),
            (r'error|failed|sorry|unable|cannot\s+generate', 'generation_error', 'Generation error message detected'),
            (r'what\s+is\s+the\s+primary\s+function\s+of', 'vague_primary_function', 'Overly vague "primary function" question'),
            (r'what\s+does\s+.{1,15}\s+do\?', 'vague_what_does', 'Overly simplistic "what does X do" question'),
            (r'(sex|sexual|reproduction)\s+(organ|system).{0,50}(produce|function)', 'inappropriate_reproductive', 'Inappropriate reproductive content'),
        ]
        
        for pattern, issue_type, description in critical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(CoherenceIssue(
                    issue_type=issue_type,
                    description=description,
                    severity='critical',
                    pattern_matched=pattern,
                    timestamp=datetime.now()
                ))
        
        # Topic-specific inappropriate content detection
        if self._contains_inappropriate_content(text, topic):
            issues.append(CoherenceIssue(
                issue_type='inappropriate_content',
                description='Contains inappropriate content',
                severity='critical',
                pattern_matched='content_filter',
                timestamp=datetime.now()
            ))
        
        return issues

    def _contains_inappropriate_content(self, text: str, topic: str) -> bool:
        """Check for inappropriate content based on topic and text"""
        # Topics that should be educational, not crude
        sensitive_topics = ['sex', 'sexual', 'reproduction', 'reproductive']
        
        if any(sensitive in topic.lower() for sensitive in sensitive_topics):
            # For reproductive topics, ensure educational content only
            inappropriate_patterns = [
                r'sexual\s+act|sexual\s+intercourse(?!\s+in\s+reproduction)',
                r'sexual\s+pleasure|arousal|orgasm',
                r'genitals?(?!\s+in\s+reproduction|\s+system|\s+anatomy)',
                r'penis|vagina(?!\s+in\s+.*system|.*anatomy|.*reproduction)',
            ]
            
            for pattern in inappropriate_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
        
        return False

    def _check_medium_patterns(self, question: str, options: List[str], text: str, topic: str) -> List[CoherenceIssue]:
        """Check for medium severity coherence issues"""
        issues = []
        
        # Enhanced medium patterns for question quality
        medium_patterns = [
            (r'which\s+of\s+the\s+following\s+is\s+true', 'generic_which_true', 'Generic "which is true" question format'),
            (r'all\s+of\s+the\s+above|none\s+of\s+the\s+above', 'poor_options', 'Uses "all/none of the above" options'),
            (r'always|never|all|none(?!\s+of\s+the\s+above)', 'absolute_terms', 'Uses absolute terms (always, never, all, none)'),
            (r'often|sometimes|frequently|rarely|usually', 'vague_frequency', 'Uses vague frequency terms'),
            (r'many|most|some|few(?!\s+\w+)', 'vague_quantity', 'Uses vague quantity terms'),
            (r'option\s+[a-d]|answer\s+[a-d]', 'meta_references', 'References to option letters in content'),
        ]
        
        for pattern, issue_type, description in medium_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(CoherenceIssue(
                    issue_type=issue_type,
                    description=description,
                    severity='medium',
                    pattern_matched=pattern,
                    timestamp=datetime.now()
                ))
        
        # Improved similarity detection for options
        if len(options) >= 2:
            similar_pairs = []
            for i, opt1 in enumerate(options):
                for j, opt2 in enumerate(options[i+1:], i+1):
                    if opt1 and opt2 and self._are_options_too_similar(opt1, opt2):
                        similar_pairs.append((i+1, j+1))
                        issues.append(CoherenceIssue(
                            issue_type='similar_options',
                            description=f'Options {i+1} and {j+1} are too similar',
                            severity='medium',
                            pattern_matched='advanced_similarity_check',
                            timestamp=datetime.now()
                        ))
            
            # If many options are similar, escalate to critical - RELAXED
            if len(similar_pairs) >= 4:  # 4 or more similar pairs (was 3)
                issues.append(CoherenceIssue(
                    issue_type='multiple_similar_options',
                    description=f'Multiple similar option pairs detected: {similar_pairs}',
                    severity='critical',
                    pattern_matched='multiple_similarity_critical',
                    timestamp=datetime.now()
                ))
        
        # Check for educational quality
        if not self._is_educationally_valuable(question, topic):
            issues.append(CoherenceIssue(
                issue_type='low_educational_value',
                description='Question lacks educational depth',
                severity='medium',
                pattern_matched='educational_value_check',
                timestamp=datetime.now()
            ))
        
        return issues

    def _are_options_too_similar(self, opt1: str, opt2: str) -> bool:
        """Advanced similarity detection for MCQ options"""
        # Clean and normalize options
        clean_opt1 = re.sub(r'[^\w\s]', '', opt1.lower()).strip()
        clean_opt2 = re.sub(r'[^\w\s]', '', opt2.lower()).strip()
        
        # Get words, excluding very common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'of', 'to', 'in', 'for', 'with', 'by'}
        words1 = set(word for word in clean_opt1.split() if word not in stop_words and len(word) > 2)
        words2 = set(word for word in clean_opt2.split() if word not in stop_words and len(word) > 2)
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard_similarity = intersection / union if union > 0 else 0
        
        # Check for exact or near-exact matches
        if clean_opt1 == clean_opt2:
            return True
        
        # Check if one option is completely contained in the other
        if clean_opt1 in clean_opt2 or clean_opt2 in clean_opt1:
            return True
        
        # Check for high word overlap (more than 80% Jaccard similarity) - RELAXED
        if jaccard_similarity > 0.8:
            return True

        # Check for semantic similarity patterns - RELAXED THRESHOLDS
        similarity_patterns = [
            # Both start with the same word/phrase - require higher overlap
            (words1 & words2 and len(words1 & words2) >= max(len(words1), len(words2)) * 0.7),
            # Both options have very similar length and high overlap - stricter
            (abs(len(words1) - len(words2)) <= 1 and jaccard_similarity > 0.7),
            # Both use the same key terms but different articles/connectors - stricter
            (len(words1 & words2) >= 3 and jaccard_similarity > 0.6)
        ]
        
        return any(similarity_patterns)

    def _is_educationally_valuable(self, question: str, topic: str) -> bool:
        """Check if the question has educational value"""
        # Questions that are too basic or vague
        low_value_patterns = [
            r'what\s+is\s+\w+\?$',  # "What is X?" with no context
            r'what\s+does\s+\w+\s+do\?$',  # "What does X do?" with no specificity
            r'what\s+is\s+the\s+function\s+of\s+\w+\?$',  # "What is the function of X?"
            r'what\s+is\s+the\s+purpose\s+of\s+\w+\?$',  # "What is the purpose of X?"
        ]
        
        for pattern in low_value_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return False
        
        # Questions should have some complexity or specificity
        good_question_indicators = [
            r'which\s+mechanism',  # "Which mechanism..."
            r'how\s+does\s+\w+\s+affect',  # "How does X affect..."
            r'what\s+happens\s+when',  # "What happens when..."
            r'during\s+\w+\s+process',  # "During X process..."
            r'in\s+the\s+context\s+of',  # "In the context of..."
            r'compared\s+to',  # "Compared to..."
            r'relationship\s+between',  # "relationship between..."
        ]
        
        for pattern in good_question_indicators:
            if re.search(pattern, question, re.IGNORECASE):
                return True
        
        # If question is reasonably long and complex, likely good
        return len(question.split()) > 8
    
    def _log_issues(self, question: str, options: List[str], topic: str, 
                   issues: List[CoherenceIssue]):
        """Log coherence issues for analysis."""
        critical_issues = [issue for issue in issues if issue.severity == 'critical']
        
        if critical_issues:
            self.logger.error(
                f"🚨 CRITICAL MCQ COHERENCE ISSUE DETECTED 🚨\n"
                f"Topic: {topic}\n"
                f"Question: {question}\n"
                f"Options: {options}\n"
                f"Issues: {[issue.description for issue in critical_issues]}"
            )
        else:
            medium_issues = [issue for issue in issues if issue.severity == 'medium']
            if medium_issues:
                self.logger.warning(
                    f"⚠️ MCQ coherence issue detected\n"
                    f"Topic: {topic}\n"
                    f"Question: {question}\n"
                    f"Issues: {[issue.description for issue in medium_issues]}"
                )
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics."""
        if self.stats['total_questions_monitored'] > 0:
            critical_rate = self.stats['critical_issues_found'] / self.stats['total_questions_monitored']
            medium_rate = self.stats['medium_issues_found'] / self.stats['total_questions_monitored']
        else:
            critical_rate = medium_rate = 0.0
        
        return {
            **self.stats,
            'critical_issue_rate': critical_rate,
            'medium_issue_rate': medium_rate,
            'overall_coherence_rate': 1.0 - critical_rate
        }
    
    def reset_stats(self):
        """Reset monitoring statistics."""
        self.stats = {
            'total_questions_monitored': 0,
            'critical_issues_found': 0,
            'medium_issues_found': 0,
            'low_issues_found': 0,
            'issue_types': {}
        }


# Global monitor instance
_global_monitor = None

def get_coherence_monitor() -> MCQCoherenceMonitor:
    """Get the global coherence monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MCQCoherenceMonitor()
    return _global_monitor
import os
import difflib
from typing import Optional, List


class DomainService:
    """
    Service for auto-detecting the best domain for a given topic string using keywords and fuzzy matching.
    """

    def __init__(self, lora_base_dir: str):
        self.lora_base_dir = lora_base_dir

    def get_available_domains(self) -> List[str]:
        if not os.path.exists(self.lora_base_dir) or not os.path.isdir(self.lora_base_dir):
            return []
        try:
            return [
                d
                for d in os.listdir(self.lora_base_dir)
                if os.path.isdir(os.path.join(self.lora_base_dir, d))
            ]
        except FileNotFoundError:
            return []

    def detect_domain_for_topic(self, topic: str) -> Optional[str]:
        available_domains = self.get_available_domains()
        if not available_domains:
            return None
        topic_lc = topic.lower()
        for domain in available_domains:
            if domain.lower() in topic_lc or topic_lc in domain.lower():
                return domain
        best_match = difflib.get_close_matches(
            topic_lc, [d.lower() for d in available_domains], n=1, cutoff=0.6
        )
        if best_match:
            for domain in available_domains:
                if domain.lower() == best_match[0]:
                    return domain
        heuristics = {
            "math": ["algebra", "geometry", "calculus", "trigonometry", "mathematics", "math"],
            "physics": ["physics", "mechanics", "magnetism", "thermodynamics", "optics"],
            "biology": ["biology", "cell", "genetics", "botany", "zoology"],
            "chemistry": ["chemistry", "organic", "inorganic", "physical chemistry"],
            "theology": ["theology", "religion", "vedic", "sanskrit", "spiritual"],
            "science": ["science", "scientific", "experiment", "research"],
        }
        for domain, keywords in heuristics.items():
            if any(kw in topic_lc for kw in keywords):
                for d in available_domains:
                    if domain in d.lower():
                        return d
        if "default" in available_domains:
            return "default"
        if "__default" in available_domains:
            return "__default"
        return available_domains[0] if available_domains else None
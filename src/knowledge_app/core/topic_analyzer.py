"""
Topic Analysis Engine - Intelligent Question Type Recommendation with AI Spell Correction
Analyzes user topics to determine suitable question types and guide UI behavior
Enhanced with smart typo detection and auto-correction capabilities
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Enhanced comprehensive misspelling dictionary with AI-like correction
COMMON_MISSPELLINGS: Dict[str, List[str]] = {
    # Psychology and related fields (MASSIVELY EXPANDED)
    "psychology": [
        "pschology", "psichology", "psycology", "psycholgy", "phychology",
        "pyschology", "psycholigy", "pshychology", "psyhcology", "psycohology",
        "psyhology", "pychology", "psychollogy", "psycholoy", "phsychology",
        "psych", "pysch", "pyscho", "psycho", "psyc", "phsyc", "phsych"
    ],
    "psychiatry": ["pshychiatry", "psychiaty", "phychiatry", "psyciatry", "psychitry"],
    "therapy": ["tharapy", "thereapy", "theraphy", "theropy", "terapy"],
    "counseling": ["counceling", "counselling", "counciling", "counsiling"],
    
    # Mathematics (EXPANDED)
    "mathematics": [
        "mathamatics", "mathimatics", "mathmatics", "mathemathics", "mathmatic",
        "maths", "math", "mathe", "matth", "matematics", "mathematik"
    ],
    "calculus": ["calculis", "calculas", "calcullus", "calulus", "calculeus"],
    "algebra": ["algerba", "algerbra", "algebera", "algebrra", "algarba"],
    "geometry": ["geometery", "geomtry", "geometrie", "geommetry", "geomety"],
    "trigonometry": ["trigonemetry", "triganometry", "trigonomtry", "trigonametry"],
    "statistics": ["statisitics", "statistcs", "statitstics", "staistics"],
    
    # Physics (EXPANDED) 
    "physics": ["phisics", "physiks", "phyics", "physiccs", "physic", "fisics"],
    "thermodynamics": ["thermodynamix", "thermodinamics", "thermodynamycs"],
    "quantum": ["quantom", "quantem", "quantam", "kuantum"],
    "mechanics": ["mecanics", "mechaniks", "mecanicks", "mechancs"],
    
    # Chemistry (EXPANDED)
    "chemistry": ["chemestry", "chemisty", "chemestry", "chemitry", "kemistry"],
    "organic": ["orgaanic", "organik", "organc", "orgnic"],
    "stoichiometry": ["stochiometry", "stoikiometry", "stoychiometry"],
    
    # Biology (NEW)
    "biology": ["biologi", "biolgy", "bioligy", "biollogy", "biologey"],
    "anatomy": ["anatomi", "anatamy", "anatomey", "anotomy"],
    "physiology": ["phisiology", "physiolgy", "physiologi", "phsyiology"],
    
    # Computer Science (NEW)
    "programming": ["programing", "programmin", "programimg", "progamming"],
    "algorithm": ["algorythm", "algoritm", "algorithem", "algorythim"],
    "computer": ["compter", "computr", "compuer", "computar"],
    "software": ["sofware", "softwar", "softwre", "softwaer"],
    "database": ["databse", "datbase", "databaes", "databas"],
    
    # Social Sciences (NEW)
    "sociology": ["sociolgy", "sociologi", "socioligy", "sociollogy"],
    "anthropology": ["anthropologi", "antropology", "anthopology", "anthropolgy"],
    "political": ["politcal", "politial", "poltical", "politicl"],
    "economics": ["economiks", "economcs", "econmics", "economis"],
    
    # History & Humanities (NEW)
    "history": ["histroy", "histori", "histery", "hsitory"],
    "philosophy": ["philosphy", "philosofy", "philsophy", "philosophy"],
    "literature": ["literatur", "literatue", "literture", "liturature"],
    "language": ["langauge", "languag", "langage", "languge"],
    
    # Arts (NEW)
    "music": ["musci", "musik", "musick", "muisc"],
    "theater": ["theatre", "theather", "theatr", "theator"],
    "literature": ["literatur", "literatue", "literture", "liturature"],
    
    # Business (NEW)
    "management": ["managment", "managemnt", "mangemnt", "manegement"],
    "marketing": ["marketting", "marketng", "markting", "merkaeting"],
    "finance": ["finace", "finanace", "finanse", "fiannce"],
    "accounting": ["acounting", "accountng", "accaunting", "accouting"],
    
    # Medical/Health (NEW)
    "medicine": ["medicin", "medecine", "medicne", "medcine"],
    "health": ["helth", "healt", "helath", "heatlh"],
    "nutrition": ["nutition", "nutritoin", "nutirition", "nutritien"],
    
    # Engineering (NEW)
    "engineering": ["enginering", "enginneering", "enginnering", "engineerng"],
    "mechanical": ["mecanical", "mechanicl", "mechancial", "mechanial"],
    "electrical": ["electical", "electricl", "electriacl", "electircal"],
    "chemical": ["chemcial", "chemiacl", "chemicl", "chmeical"],
    
    # General academic terms (NEW)
    "education": ["educaion", "educatin", "eduction", "educaton"],
    "science": ["sciene", "scinece", "scienc", "scince"],
    "research": ["reserach", "researh", "reseach", "rsearch"],
    "analysis": ["anaylsis", "analsis", "anlaysis", "analyis"],
    "theory": ["theiry", "theori", "teory", "theorey"],
    "method": ["methid", "metod", "methd", "methode"],
    "study": ["studi", "studie", "stdy", "studey"],
    
    # 🤖 NLP & LINGUISTICS MISSPELLINGS (Comprehensive from research)
    # Core NLP Terms
    "natural language processing": [
        "natural language proccessing", "natural languag processing", "natual language processing",
        "natural language procesing", "natrual language processing", "natural languge processing"
    ],
    "tokenization": ["tokenisation", "tokeniztion", "tokennization", "tokenizaton"],
    "stemming": ["steming", "stemmng", "stemimg", "stemmig"],
    "lemmatization": ["lemmatisation", "lemmatiztion", "lematization", "lemmatizaton"],
    "sentiment": ["sentimnt", "sentimemt", "senntiment", "sentimnet"],
    "semantic": ["semantik", "semanic", "semantc", "semnatic"],
    "syntax": ["sintax", "syntx", "syntak", "syntacs"],
    "pragmatics": ["pragmatiks", "pragmatics", "pragmatcs", "pragmatix"],
    "morphology": ["morphologi", "morpholgy", "morpholigy", "morphollogy"],
    "phonology": ["phonologi", "phonolgy", "phonoligy", "phonollogy"],
    "phonetics": ["phonetiks", "phonetcs", "phonetix", "phonetic"],
    "discourse": ["discorse", "discours", "discoures", "discourze"],
    "coreference": ["coreference", "co-reference", "corerefrence", "coreferance"],
    "anaphora": ["anafora", "anaphoora", "anaphra", "anaphorra"],
    
    # Advanced NLP
    "embeddings": ["embedings", "embeddngs", "embeding", "embeddings"],
    "bigrams": ["bi-grams", "bigram", "bigrms", "bigrmas"],
    "trigrams": ["tri-grams", "trigram", "trigrms", "trigrmas"],
    "corpus": ["corpous", "corpuss", "corupus", "korpus"],
    "corpora": ["corpoora", "copora", "corporaa", "korpora"],
    "parsing": ["parsng", "parsig", "persing", "parsing"],
    "disambiguation": ["desambiguation", "disambiguaton", "disambiguaion", "disambiguatin"],
    
    # Linguistics
    "linguistics": ["linguistiks", "linguistcs", "linguistix", "lingustics"],
    "sociolinguistics": ["socio-linguistics", "sociolinguistiks", "sociolinguistcs"],
    "psycholinguistics": ["psycho-linguistics", "psycholinguistiks", "psycholinguistcs"],
    "neurolinguistics": ["neuro-linguistics", "neurolinguistiks", "neurolinguistcs"],
    "pragmatics": ["pragmatiks", "pragmatcs", "pragmatix", "pragmatics"],
    
    # Communication & Rhetoric
    "rhetoric": ["retoric", "rethoric", "rhetorik", "rhetorric"],
    "oratory": ["oratory", "oraotry", "oreatory", "oratery"],
    "persuasion": ["persuation", "persuasion", "perswaion", "persuashin"],
    "argumentation": ["arguementaion", "argumentaton", "argumetation", "argumentaion"],
    
    # Literary Terms
    "narrative": ["narative", "narrativ", "narritive", "narrattive"],
    "metaphor": ["metafor", "metaphore", "methaphor", "metafore"],
    "allegory": ["allegori", "alegory", "allegoory", "allegroy"],
    "symbolism": ["simbolism", "symbolizm", "symbolysm", "simbolysm"],
    "imagery": ["imagry", "imagerry", "imagary", "iamgery"],
    
    # Philosophy & Ethics
    "epistemology": ["epistemologi", "epistemolgy", "epistemollogy", "epistmology"],
    "metaphysics": ["metaphysiks", "metaphysix", "metaphyscs", "metafysics"],
    "deontology": ["deontologi", "deontolgy", "deontollogy", "deontalogy"],
    "utilitarianism": ["utilitarian", "utilitarianizm", "utiliterianism", "utilitarism"],
    "existentialism": ["existentialism", "existencialisim", "existentializm", "existenshalism"],
    "phenomenology": ["phenomenologi", "phenomenolgy", "phenomenollogy", "fenomenology"],
    
    # Design & Creative
    "aesthetics": ["aesthetiks", "esthetics", "aestethics", "aestehtics"],
    "choreography": ["choreografi", "choreograpy", "choreography", "choreagraphy"],
    "dramaturgy": ["dramaturgi", "dramaturgy", "dramatrgy", "dramatugy"],
    
    # Cognitive Science
    "consciousness": ["conciousness", "consciouness", "conciousnes", "consciousnes"],
    "metacognition": ["meta-cognition", "metacognition", "metacognitionn", "metacognition"],
    "perception": ["preception", "percption", "preceptio", "perseption"],
    "cognition": ["cogntion", "congition", "cognitio", "cogniton"]
}

# Phonetic patterns for sound-based matching
PHONETIC_PATTERNS: Dict[str, List[str]] = {
    "psychology": ["sychology", "sikology", "psikology"],
    "physics": ["fisics", "fisiks"],
    "chemistry": ["kemistry", "chemestry"], 
    "philosophy": ["filosofy", "filosophy"],
    "mathematics": ["matematics", "mathemetics"],
    "history": ["histery", "istoric"],
    "biology": ["biolology", "biologi"]
}

# Define keywords for different categories
# This is simple, fast, and highly effective for this purpose.
CONCEPTUAL_KEYWORDS: Set[str] = {
    # Core Humanities & Social Sciences
    "history", "art", "philosophy", "literature", "geopolitics", "social",
    "psychology", "sociology", "law", "ethics", "theory", "qualitative",
    "sex ed", "reproductive health", "biology", "anatomy", "culture",
    "religion", "politics", "language", "music", "writing", "communication",
    "anthropology", "archaeology", "linguistics", "education", "teaching",
    "learning", "pedagogy", "curriculum", "assessment", "evaluation",
    
    # Enhanced: Psychology & Mental Health
    "psychology", "psych", "psycho", "psychological", "mental health", "therapy",
    "counseling", "psychiatry", "cognitive", "behavioral", "emotional", "feelings",
    "mind", "brain", "consciousness", "personality", "memory", "learning",
    "motivation", "stress", "anxiety", "depression", "trauma", "wellbeing",
    
    # Social Sciences & Human Studies  
    "sociology", "social work", "social studies", "anthropology", "criminology",
    "political science", "government", "democracy", "citizenship", "human rights",
    "gender studies", "women studies", "diversity", "inclusion", "equality",
    "race", "ethnicity", "identity", "community", "society", "culture",
    
    # Humanities & Arts
    "literature", "poetry", "drama", "theater", "film studies", "media studies",
    "art history", "fine arts", "visual arts", "performing arts", "dance",
    "music theory", "composition", "creative writing", "journalism",
    "philosophy", "ethics", "moral", "logic", "metaphysics", "epistemology",
    
    # Language & Communication
    "linguistics", "language", "grammar", "syntax", "semantics", "phonetics",
    "communication", "rhetoric", "public speaking", "debate", "argumentation",
    "reading", "writing", "composition", "literacy", "bilingual", "multilingual",
    
    # Religious & Spiritual Studies
    "religion", "theology", "spirituality", "faith", "belief", "christianity",
    "islam", "judaism", "hinduism", "buddhism", "atheism", "agnosticism",
    "scripture", "bible", "quran", "prayer", "worship", "meditation",
    
    # Legal & Political Studies
    "law", "legal", "justice", "rights", "constitution", "court", "judge",
    "attorney", "lawyer", "legislation", "policy", "governance", "democracy",
    "politics", "political", "voting", "election", "government", "congress",
    
    # Health & Medical (Conceptual Aspects)
    "health", "medical", "medicine", "healthcare", "nursing", "wellness",
    "nutrition", "diet", "exercise", "fitness", "public health", "epidemiology",
    "anatomy", "physiology", "pathology", "disease", "illness", "recovery",
    "sex", "sexuality", "reproduction", "reproductive", "pregnancy", "birth",
    
    # Education & Development
    "education", "teaching", "learning", "pedagogy", "curriculum", "instruction",
    "assessment", "evaluation", "childhood", "development", "parenting",
    "child development", "adolescence", "adult learning", "special education",
    
    # Business & Management (Conceptual Aspects)
    "management", "leadership", "business ethics", "organizational behavior",
    "human resources", "marketing", "strategy", "entrepreneurship", "innovation",
    "corporate culture", "team dynamics", "communication", "negotiation",
    
    # Environmental & Sustainability (Conceptual)
    "environmental studies", "sustainability", "conservation", "ecology",
    "climate change", "global warming", "renewable energy", "green technology",
    "environmental policy", "biodiversity", "ecosystem", "natural resources",
    
    # Cultural & International Studies
    "cultural studies", "international relations", "globalization", "diplomacy",
    "foreign policy", "world history", "comparative politics", "area studies",
    "migration", "immigration", "multiculturalism", "cross-cultural",
    
    # Philosophy & Critical Thinking
    "critical thinking", "logical reasoning", "argumentation", "debate",
    "problem solving", "decision making", "creativity", "innovation",
    "research methods", "qualitative research", "case study", "ethnography"
}

NUMERICAL_KEYWORDS: Set[str] = {
    # Core Mathematical Fields
    "math", "mathematics", "maths", "mathematical", "arithmetic", "algebra",
    "calculus", "geometry", "trigonometry", "statistics", "probability",
    "differential", "integral", "linear algebra", "discrete math", "number theory",
    
    # Physics & Engineering
    "physics", "mechanical", "electrical", "civil", "chemical engineering",
    "thermodynamics", "mechanics", "dynamics", "kinematics", "optics",
    "quantum", "relativity", "electromagnetism", "waves", "energy",
    
    # Chemistry & Quantitative Sciences
    "chemistry", "chemical", "molecular", "atomic", "atoms", "atom", "reactions", "stoichiometry",
    "organic chemistry", "inorganic chemistry", "physical chemistry", "biochemistry",
    "analytical chemistry", "chemical analysis", "concentration", "molarity", "elements",
    "periodic table", "compounds", "molecules", "ions", "electrons", "protons", "neutrons",
    
    # Computer Science & Technology (Quantitative)
    "programming", "algorithms", "data structures", "computational", "coding",
    "software engineering", "computer science", "machine learning", "AI",
    "artificial intelligence", "data science", "big data", "databases",
    "cybersecurity", "networking", "systems", "hardware", "software",
    
    # Finance & Economics (Quantitative)
    "finance", "financial", "economics", "accounting", "budgeting", "investment",
    "banking", "stock market", "trading", "portfolio", "risk management",
    "financial analysis", "cost accounting", "financial modeling", "valuation",
    "economics", "microeconomics", "macroeconomics", "econometrics",
    
    # Data & Analytics
    "statistics", "statistical", "data", "analysis", "analytics", "metrics",
    "measurement", "quantitative", "numerical", "computational", "modeling",
    "regression", "correlation", "hypothesis testing", "sampling", "survey",
    
    # Engineering & Technical Fields
    "engineering", "technical", "mechanical", "electrical", "civil", "aerospace",
    "biomedical engineering", "industrial engineering", "systems engineering",
    "structural", "materials", "design", "CAD", "simulation", "optimization",
    
    # Scientific Research & Lab Work
    "science", "scientific", "experiment", "laboratory", "research", "formula",
    "equation", "calculation", "computing", "measuring", "testing", "analysis",
    "methodology", "procedures", "protocols", "instrumentation", "calibration",
    
    # Business Analytics & Operations
    "operations research", "supply chain", "logistics", "inventory", "forecasting",
    "quality control", "process optimization", "lean manufacturing", "six sigma",
    "project management", "scheduling", "resource allocation", "capacity planning",
    
    # Applied Mathematics
    "applied mathematics", "mathematical modeling", "operations research",
    "optimization", "simulation", "game theory", "cryptography", "coding theory",
    "signal processing", "image processing", "pattern recognition", "neural networks"
}

# Mixed topics that can support both conceptual and numerical questions
MIXED_KEYWORDS: Set[str] = {
    "business", "management", "marketing", "economics",
    "environmental science", "medicine", "health", "nutrition", "fitness",
    "project management", "operations", "logistics", "supply chain"
}

# 🔥 NEW: Topics that should ONLY allow conceptual questions (NO numerical)
CONCEPTUAL_ONLY_KEYWORDS: Set[str] = {
    # Psychology & Mental Health (STRICT - NO numerical)
    "psychology", "psychological", "psych", "psychiatry", "therapy", "counseling",
    "mental health", "cognitive", "behavioral", "emotional", "personality",
    "depression", "anxiety", "trauma", "wellbeing", "mindfulness",
    
    # Social Sciences (STRICT - NO numerical)  
    "sociology", "anthropology", "social work", "criminology", "political science",
    "gender studies", "women studies", "diversity", "inclusion", "race", "ethnicity",
    "identity", "community", "society", "culture", "social studies",
    
    # Humanities & Arts (STRICT - NO numerical)
    "literature", "poetry", "drama", "theater", "philosophy", "ethics",
    "art history", "fine arts", "visual arts", "performing arts", "dance",
    "music theory", "creative writing", "journalism", "media studies",
    "film studies", "art", "music", "painting", "sculpture",
    
    # Language & Communication (STRICT - NO numerical)
    "linguistics", "language", "grammar", "rhetoric", "communication",
    "public speaking", "debate", "writing", "composition", "literacy",
    "reading", "english", "spanish", "french", "german", "chinese",
    
    # Religious & Spiritual (STRICT - NO numerical)
    "religion", "theology", "spirituality", "faith", "belief", "christianity",
    "islam", "judaism", "hinduism", "buddhism", "prayer", "worship",
    "meditation", "scripture", "bible", "quran",
    
    # Legal & Political (STRICT - NO numerical)
    "law", "legal", "justice", "rights", "constitution", "court", "politics",
    "political", "government", "democracy", "voting", "election", "policy",
    "legislation", "governance", "congress", "parliament",
    
    # History & Cultural Studies (STRICT - NO numerical)
    "history", "historical", "ancient", "medieval", "modern", "contemporary",
    "cultural studies", "international relations", "diplomacy", "foreign policy",
    "world history", "american history", "european history", "asian history",
    
    # Education & Human Development (STRICT - NO numerical)
    "education", "teaching", "learning", "pedagogy", "curriculum", "instruction",
    "childhood", "development", "parenting", "child development", "adolescence",
    "adult learning", "special education", "educational theory",
    
    # Human Sexuality & Relationships (STRICT - NO numerical)
    "sex", "sexuality", "sex education", "reproductive health", "relationships",
    "dating", "marriage", "family", "parenting", "love", "romance",
    
    # Philosophy & Critical Thinking (STRICT - NO numerical)
    "philosophy", "philosophical", "ethics", "moral", "logic", "metaphysics",
    "epistemology", "critical thinking", "reasoning", "argumentation",
    
    # 🤖 NLP & LINGUISTICS (Comprehensive from research - STRICT conceptual-only)
    # Core NLP Concepts
    "nlp", "natural language processing", "tokenization", "stemming", "lemmatization",
    "pos tagging", "part of speech", "named entity recognition", "ner", "parsing",
    "sentiment analysis", "text classification", "word embeddings", "bag of words",
    "n-grams", "bigrams", "trigrams", "corpus", "corpora", "stop words",
    
    # Linguistic Analysis
    "syntax", "semantics", "pragmatics", "morphology", "phonology", "phonetics",
    "discourse", "discourse analysis", "coreference", "coreference resolution",
    "anaphora", "cataphora", "word sense disambiguation", "lexical semantics",
    
    # Language Understanding & Generation
    "natural language understanding", "nlu", "natural language generation", "nlg",
    "language models", "contextual embeddings", "dialog systems", "chatbots",
    "question answering", "machine translation", "spell correction", "ocr",
    "optical character recognition", "speech recognition", "text to speech",
    
    # Advanced NLP Concepts
    "topic modeling", "information extraction", "summarization", "text summarization",
    "question generation", "language vectors", "sequence to sequence", "transducers",
    "multimodal learning", "knowledge graphs", "ontologies", "linguistic rules",
    "probabilistic models", "evaluation metrics", "low resource languages",
    
    # Specialized Linguistics Fields
    "sociolinguistics", "psycholinguistics", "neurolinguistics", "forensic linguistics",
    "stylistics", "morphophonology", "readability", "text simplification",
    "code switching", "multiword expressions", "collocation", "idioms",
    
    # Communication Studies
    "rhetoric", "oratory", "persuasion", "argumentation theory", "debate theory",
    "public speaking", "interpersonal communication", "mass communication",
    "media literacy", "semiotics", "discourse markers", "pragmatic markers",
    
    # 📚 LITERARY & TEXTUAL ANALYSIS (STRICT - NO numerical)
    "literary analysis", "literary theory", "narrative", "storytelling", "plot",
    "character development", "symbolism", "metaphor", "allegory", "irony",
    "tone", "mood", "theme", "motif", "imagery", "figurative language",
    "prose", "verse", "sonnet", "haiku", "epic", "novel", "short story",
    "autobiography", "biography", "memoir", "essay", "editorial",
    
    # 🎭 CULTURAL & PERFORMANCE STUDIES (STRICT - NO numerical)
    "cultural anthropology", "ethnography", "folklore", "mythology", "oral tradition",
    "performance studies", "theater studies", "dramaturgy", "stage design",
    "costume design", "choreography", "dance theory", "music theory",
    "musicology", "ethnomusicology", "art criticism", "aesthetic theory",
    
    # 🌍 AREA & REGIONAL STUDIES (STRICT - NO numerical)
    "african studies", "asian studies", "european studies", "latin american studies",
    "middle eastern studies", "american studies", "canadian studies", "british studies",
    "comparative literature", "world literature", "postcolonial studies",
    "globalization studies", "migration studies", "diaspora studies",
    
    # 🤝 SOCIAL THEORY & HUMAN BEHAVIOR (STRICT - NO numerical)
    "social theory", "social psychology", "group dynamics", "organizational psychology",
    "consumer psychology", "environmental psychology", "developmental psychology",
    "personality psychology", "abnormal psychology", "clinical psychology",
    "counseling psychology", "educational psychology", "school psychology",
    
    # ⚖️ LAW & JUSTICE STUDIES (STRICT - NO numerical)
    "constitutional law", "criminal law", "civil law", "international law",
    "human rights law", "environmental law", "business law", "family law",
    "immigration law", "intellectual property", "legal theory", "jurisprudence",
    "legal ethics", "legal writing", "legal research", "case law",
    
    # 🏛️ POLITICAL & GOVERNANCE STUDIES (STRICT - NO numerical)
    "political theory", "comparative politics", "public policy", "public administration",
    "international relations", "diplomacy", "foreign policy", "national security",
    "democratic theory", "authoritarian studies", "electoral systems", "voting behavior",
    "political economy", "political philosophy", "political communication",
    
    # 🧘 PHILOSOPHY & ETHICS (STRICT - NO numerical)
    "moral philosophy", "ethical theory", "applied ethics", "bioethics", "business ethics",
    "environmental ethics", "medical ethics", "research ethics", "professional ethics",
    "virtue ethics", "deontology", "consequentialism", "utilitarianism",
    "existentialism", "phenomenology", "hermeneutics", "dialectics",
    
    # 🎨 CREATIVE & DESIGN STUDIES (STRICT - NO numerical)
    "design thinking", "user experience", "user interface design", "graphic design",
    "industrial design", "fashion design", "interior design", "landscape architecture",
    "urban planning", "architectural theory", "design philosophy", "aesthetic design",
    "creative process", "artistic expression", "art therapy", "music therapy",
    
    # 📖 EDUCATIONAL THEORY & PEDAGOGY (STRICT - NO numerical)
    "educational theory", "learning theory", "constructivism", "behaviorism",
    "cognitive learning", "social learning", "adult learning theory", "andragogy",
    "pedagogy", "curriculum theory", "instructional design", "assessment theory",
    "educational psychology", "classroom management", "inclusive education",
    
    # 🧠 COGNITIVE & MENTAL PROCESSES (STRICT - NO numerical)
    "cognitive science", "cognitive psychology", "memory", "attention", "perception",
    "consciousness", "decision making", "problem solving", "creativity", "innovation",
    "critical thinking", "metacognition", "executive function", "working memory",
    "long term memory", "semantic memory", "episodic memory", "procedural memory"
}

class AISpellCorrector:
    """
    AI-like spell correction and typo detection system
    Uses fuzzy matching, phonetic patterns, and comprehensive misspelling dictionaries
    """
    
    def __init__(self):
        self.misspelling_dict = COMMON_MISSPELLINGS
        self.phonetic_patterns = PHONETIC_PATTERNS
        self.correction_cache = {}  # Cache for performance
        logger.info("🤖 AI Spell Corrector initialized with comprehensive misspelling database")
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def similarity_ratio(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings (0-1)"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def find_best_correction(self, word: str, candidates: List[str], threshold: float = 0.6) -> Optional[str]:
        """
        Find the best correction for a misspelled word using multiple algorithms
        """
        if not word or not candidates:
            return None
            
        word_lower = word.lower().strip()
        
        # Check cache first
        cache_key = f"{word_lower}:{','.join(sorted(candidates))}"
        if cache_key in self.correction_cache:
            return self.correction_cache[cache_key]
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # Exact match
            if word_lower == candidate_lower:
                self.correction_cache[cache_key] = candidate
                return candidate
            
            # Check if word is contained in candidate or vice versa
            if word_lower in candidate_lower or candidate_lower in word_lower:
                containment_score = min(len(word_lower), len(candidate_lower)) / max(len(word_lower), len(candidate_lower))
                if containment_score > best_score:
                    best_score = containment_score
                    best_match = candidate
            
            # Similarity ratio check
            similarity = self.similarity_ratio(word_lower, candidate_lower)
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = candidate
                
            # Levenshtein distance check (for close typos)
            if len(word_lower) > 3 and len(candidate_lower) > 3:  # Only for longer words
                max_len = max(len(word_lower), len(candidate_lower))
                distance = self.levenshtein_distance(word_lower, candidate_lower)
                distance_score = 1 - (distance / max_len)
                
                if distance_score > best_score and distance_score >= threshold:
                    best_score = distance_score 
                    best_match = candidate
        
        # Cache the result
        self.correction_cache[cache_key] = best_match
        
        if best_match and best_score >= threshold:
            logger.debug(f"🔧 AI Correction: '{word}' → '{best_match}' (score: {best_score:.2f})")
            return best_match
            
        return None
    
    def correct_topic_spelling(self, topic: str) -> Tuple[str, List[str]]:
        """
        Correct spelling in a topic string and return corrected version + corrections made
        
        Returns:
            (corrected_topic, list_of_corrections_made)
        """
        if not topic or len(topic.strip()) < 2:
            return topic, []
        
        words = re.findall(r'\b\w+\b', topic.lower())
        corrections_made = []
        corrected_topic = topic
        
        for word in words:
            if len(word) < 3:  # Skip very short words
                continue
                
            # Check direct misspelling dictionary first
            for correct_word, misspellings in self.misspelling_dict.items():
                if word in misspellings or word == correct_word:
                    if word != correct_word:
                        # Replace in the original topic (case-insensitive)
                        corrected_topic = re.sub(r'\b' + re.escape(word) + r'\b', 
                                               correct_word, corrected_topic, flags=re.IGNORECASE)
                        corrections_made.append(f"{word} → {correct_word}")
                        logger.info(f"🎯 Direct correction: {word} → {correct_word}")
                    break
            else:
                # If not found in direct dictionary, try fuzzy matching
                all_correct_words = list(self.misspelling_dict.keys())
                
                # Add keywords to search space (enhanced with conceptual-only terms)
                search_space = (all_correct_words + 
                              list(CONCEPTUAL_KEYWORDS) + 
                              list(NUMERICAL_KEYWORDS) + 
                              list(CONCEPTUAL_ONLY_KEYWORDS))
                
                correction = self.find_best_correction(word, search_space, threshold=0.7)
                if correction and correction != word:
                    corrected_topic = re.sub(r'\b' + re.escape(word) + r'\b', 
                                           correction, corrected_topic, flags=re.IGNORECASE)
                    corrections_made.append(f"{word} → {correction}")
                    logger.info(f"🧠 AI fuzzy correction: {word} → {correction}")
        
        return corrected_topic, corrections_made
    
    def detect_phonetic_matches(self, word: str) -> List[str]:
        """Detect phonetic matches for a word"""
        matches = []
        word_lower = word.lower()
        
        for correct_word, patterns in self.phonetic_patterns.items():
            if word_lower in patterns or word_lower == correct_word:
                matches.append(correct_word)
                
        return matches

class TopicAnalyzer:
    """
    Analyzes a topic string to determine its nature and recommend appropriate question types.
    Enhanced with AI-like spell correction and intelligent semantic mapping.
    """

    def __init__(self, use_semantic_mapping: bool = True):
        self.conceptual_keywords = CONCEPTUAL_KEYWORDS
        self.numerical_keywords = NUMERICAL_KEYWORDS
        self.mixed_keywords = MIXED_KEYWORDS
        self.conceptual_only_keywords = CONCEPTUAL_ONLY_KEYWORDS
        self.spell_corrector = AISpellCorrector()
        self.use_semantic_mapping = use_semantic_mapping

        # Initialize semantic mapper if enabled
        self.semantic_mapper = None
        if use_semantic_mapping:
            try:
                from .intelligent_semantic_mapper import get_semantic_mapper
                self.semantic_mapper = get_semantic_mapper()
                logger.info("🧠 TopicAnalyzer initialized with AI semantic mapping and spell correction")
            except Exception as e:
                logger.warning(f"⚠️ Semantic mapping unavailable, using keyword fallback: {e}")
                self.use_semantic_mapping = False

        if not self.use_semantic_mapping:
            logger.info("🧠 TopicAnalyzer initialized with keyword matching and spell correction")

    def get_topic_profile(self, topic: str) -> Dict[str, any]:
        """
        Analyzes the topic and returns a profile of possible question types.
        Enhanced with AI semantic mapping and spell correction capabilities.

        Args:
            topic: The topic string to analyze

        Returns:
            A dictionary with analysis results including corrections made
        """
        if not topic or len(topic.strip()) < 2:
            # For very short topics, enable everything
            return {
                "is_conceptual_possible": True,
                "is_numerical_possible": True,
                "confidence": "low",
                "detected_type": "unknown",
                "original_topic": topic,
                "corrected_topic": topic,
                "corrections_made": []
            }

        # 🚀 CRITICAL FIX: Check keyword analysis first for clearly numerical topics
        # This ensures topics like "atoms", "physics", "chemistry" are correctly identified as numerical
        corrected_topic, corrections_made = self.spell_corrector.correct_topic_spelling(topic)
        topic_lower = corrected_topic.lower().strip()

        # Quick check for clearly numerical topics that should bypass semantic mapping
        clearly_numerical_topics = [
            'atoms', 'atom', 'atomic', 'physics', 'chemistry', 'math', 'mathematics',
            'calculus', 'algebra', 'geometry', 'statistics', 'molecular', 'elements'
        ]
        is_clearly_numerical = any(num_topic in topic_lower for num_topic in clearly_numerical_topics)

        if is_clearly_numerical:
            logger.info(f"🔢 CLEARLY NUMERICAL topic detected: '{topic}' - forcing numerical analysis")
            # Force numerical analysis for clearly numerical topics
            profile = {
                "is_conceptual_possible": True,  # Numerical topics can still have conceptual questions
                "is_numerical_possible": True,
                "confidence": "high",
                "detected_type": "numerical",
                "original_topic": topic,
                "corrected_topic": corrected_topic,
                "corrections_made": corrections_made
            }
            return profile

        # 🧠 Try intelligent semantic mapping for other topics
        if self.use_semantic_mapping and self.semantic_mapper:
            try:
                semantic_profile = self.semantic_mapper.get_enhanced_topic_profile(topic)
                logger.info(f"🎯 Semantic analysis: '{topic}' → {semantic_profile['detected_type']} (confidence: {semantic_profile['confidence']})")

                # If semantic mapping has high confidence, use it
                if semantic_profile['confidence'] in ['high', 'medium']:
                    return semantic_profile
                else:
                    logger.debug(f"🔄 Low semantic confidence, falling back to keyword analysis")
            except Exception as e:
                logger.warning(f"⚠️ Semantic mapping failed, using keyword fallback: {e}")

        # Fallback to traditional keyword analysis
        # (topic already corrected and processed above)
        
        # Check for keyword matches (enhanced with corrected topic)
        is_conceptual = any(keyword in topic_lower for keyword in self.conceptual_keywords)
        is_numerical = any(keyword in topic_lower for keyword in self.numerical_keywords)
        is_mixed = any(keyword in topic_lower for keyword in self.mixed_keywords)
        is_conceptual_only = any(keyword in topic_lower for keyword in self.conceptual_only_keywords)

        # Advanced pattern matching for better accuracy
        conceptual_score = self._calculate_conceptual_score(topic_lower)
        numerical_score = self._calculate_numerical_score(topic_lower)

        # 🔥 CRITICAL FIX: Determine profile with CONCEPTUAL_ONLY taking highest priority
        if is_conceptual_only:
            profile = {
                "is_conceptual_possible": True,
                "is_numerical_possible": False,  # 🚫 BLOCK numerical completely
                "confidence": "high",
                "detected_type": "conceptual_only"
            }
            logger.info(f"🚫 CONCEPTUAL ONLY detected for '{corrected_topic}' - numerical options DISABLED")
        elif is_mixed and not is_conceptual_only:  # Only allow mixed if NOT conceptual-only
            profile = {
                "is_conceptual_possible": True,
                "is_numerical_possible": True,
                "confidence": "high",
                "detected_type": "mixed"
            }
        elif is_conceptual or conceptual_score > numerical_score:
            profile = {
                "is_conceptual_possible": True,
                "is_numerical_possible": False,
                "confidence": "high" if is_conceptual else "medium",
                "detected_type": "conceptual"
            }
        elif is_numerical or numerical_score > conceptual_score:
            profile = {
                "is_conceptual_possible": True,  # Numerical topics can still have conceptual questions
                "is_numerical_possible": True,
                "confidence": "high" if is_numerical else "medium",
                "detected_type": "numerical"
            }
        else:
            # If we don't recognize the topic, assume all types are possible as a safe default
            profile = {
                "is_conceptual_possible": True,
                "is_numerical_possible": True,
                "confidence": "low",
                "detected_type": "general"
            }

        # Add AI correction information
        profile.update({
            "original_topic": topic,
            "corrected_topic": corrected_topic,
            "corrections_made": corrections_made,
            "spelling_corrected": len(corrections_made) > 0
        })

        logger.debug(f"🎯 Topic '{topic}' analyzed: {profile}")
        if corrections_made:
            logger.info(f"🤖 AI Corrections applied: {corrections_made}")
        
        return profile

    def _calculate_conceptual_score(self, topic_lower: str) -> int:
        """Calculate how conceptual a topic appears to be"""
        score = 0
        
        # Strong conceptual indicators with enhanced patterns
        conceptual_patterns = [
            r'\b(history|philosophy|literature|art|culture)\b',
            r'\b(social|society|sociology|psychology|psych)\b',  # Enhanced psych detection
            r'\b(ethics|law|legal|policy|politics|political)\b',
            r'\b(theory|theoretical|concept|idea|conceptual)\b',
            r'\b(education|learning|teaching|pedagogy)\b',
            # Enhanced patterns for biological/health topics
            r'\b(sex|sexuality|reproduction|reproductive)\b',
            r'\b(health|medical|anatomy|physiology)\b',
            r'\b(behavioral|cultural|moral|ethics)\b',
            r'\b(human|body|biology|biological)\b',
            # NEW: Psychology variations with typo tolerance (AI-enhanced)
            r'\b(psychology|psychological|psycho|psych)\b',
            r'\b(pysch|pyscho|psycholog|pschology|psichology)\b',  # AI typo detection
            r'\b(mental|therapy|counseling|emotional)\b',
            # NEW: Humanities and social sciences
            r'\b(humanities|anthropology|linguistics|communication)\b',
            r'\b(religion|theology|spirituality|faith)\b',
            r'\b(government|democracy|citizenship|rights)\b',
            r'\b(gender|diversity|identity|community)\b',
            # NEW: Arts and creative fields
            r'\b(music|dance|theater|drama|poetry|creative)\b',
            r'\b(journalism|media|film|visual|performing)\b',
            # AI-enhanced patterns for common misspellings (MASSIVELY EXPANDED)
            r'\b(histroy|philosphy|literatur|sociolgy)\b',  # Common typos
            r'\b(managment|marketting|educaion|reserach)\b',
            # NLP & Linguistics misspellings
            r'\b(linguistiks|linguistcs|semantik|sintax|pragmatiks)\b',
            r'\b(tokenisation|tokeniztion|lemmatisation|sentimnt)\b',
            r'\b(narative|metafor|simbolism|imagry|retoric)\b',
            # Philosophy & ethics misspellings  
            r'\b(epistemologi|metaphysiks|deontologi|conciousness)\b',
            r'\b(existencialisim|phenomenologi|utiliterianism)\b',
            # Academic field misspellings
            r'\b(anthropologi|sociolinguistiks|psycholinguistiks)\b'
        ]
        
        for pattern in conceptual_patterns:
            if re.search(pattern, topic_lower):
                score += 2
                
        # Moderate conceptual indicators
        conceptual_words = [
            'why', 'how', 'what', 'explain', 'describe', 'analyze', 'understand',
            'meaning', 'significance', 'importance', 'impact', 'influence', 'effect',
            'compare', 'contrast', 'evaluate', 'discuss', 'explore', 'examine'
        ]
        if any(word in topic_lower for word in conceptual_words):
            score += 1
            
        # Special handling for short but clearly conceptual topics (MASSIVELY EXPANDED)
        short_conceptual_topics = [
            # Core Conceptual Fields
            'sex', 'love', 'art', 'music', 'history', 'politics', 'culture', 'religion',
            'psychology', 'psych', 'sociology', 'philosophy', 'ethics', 'law', 'legal',
            'education', 'health', 'medicine', 'literature', 'writing', 'language',
            'communication', 'management', 'leadership', 'business', 'marketing',
            'government', 'democracy', 'society', 'community', 'identity', 'gender',
            'race', 'ethnicity', 'diversity', 'faith', 'belief', 'spirituality',
            
            # NLP & Linguistics (Conceptual Only)
            'nlp', 'linguistics', 'syntax', 'semantics', 'pragmatics', 'morphology',
            'phonology', 'phonetics', 'discourse', 'rhetoric', 'tokenization', 'parsing',
            'sentiment', 'coreference', 'anaphora', 'corpus', 'corpora', 'embeddings',
            
            # Literary & Creative
            'narrative', 'metaphor', 'allegory', 'symbolism', 'imagery', 'poetry',
            'drama', 'theater', 'novel', 'essay', 'biography', 'memoir', 'prose',
            
            # Philosophy & Ethics
            'epistemology', 'metaphysics', 'deontology', 'utilitarianism', 'existentialism',
            'phenomenology', 'consciousness', 'cognition', 'perception', 'aesthetics',
            
            # Social Sciences
            'anthropology', 'criminology', 'ethnography', 'folklore', 'mythology',
            
            # Communication & Media
            'journalism', 'oratory', 'persuasion', 'argumentation', 'semiotics',
            
            # Common misspellings for short topics (AI-enhanced)
            'psych', 'pysch', 'histroy', 'philosphy', 'sociolgy', 'managment',
            'linguistiks', 'semantik', 'sintax', 'retoric', 'narative', 'metafor',
            'epistemologi', 'conciousness', 'anthropologi'
        ]
        if topic_lower in short_conceptual_topics:
            score += 3  # High boost for clearly conceptual single words
            
        return score

    def _calculate_numerical_score(self, topic_lower: str) -> int:
        """Calculate how numerical a topic appears to be (AI-enhanced)"""
        score = 0
        
        # Strong numerical indicators with enhanced patterns and AI typo detection
        numerical_patterns = [
            r'\b(math|mathematics|mathematical|arithmetic|maths)\b',
            r'\b(physics|chemistry|engineering|science|scientific)\b',
            r'\b(statistics|statistical|data|analytics|metrics)\b',
            r'\b(finance|financial|economics|accounting|investment)\b',
            r'\b(programming|algorithm|computational|technical)\b',
            # NEW: Specific mathematical fields
            r'\b(calculus|algebra|geometry|trigonometry|probability)\b',
            r'\b(differential|integral|linear|discrete|number)\b',
            # NEW: Engineering disciplines
            r'\b(mechanical|electrical|civil|chemical|aerospace)\b',
            r'\b(biomedical|industrial|software|systems)\b',
            # NEW: Quantitative business fields
            r'\b(operations|logistics|supply|inventory|forecasting)\b',
            r'\b(optimization|simulation|modeling|analysis)\b',
            # NEW: Computer science & tech
            r'\b(AI|machine learning|data science|algorithms|coding)\b',
            r'\b(databases|cybersecurity|networking|hardware|software)\b',
            # AI-enhanced patterns for common misspellings
            r'\b(mathamatics|mathimatics|phisics|chemestry|programing)\b',
            r'\b(algorythm|finace|enginering|compter|databse)\b'
        ]
        
        for pattern in numerical_patterns:
            if re.search(pattern, topic_lower):
                score += 2
                
        # Look for numbers or mathematical expressions
        if re.search(r'\d+|[\+\-\*/=<>%\^]|\bpi\b|\be\b', topic_lower):
            score += 2  # Increased weight for actual math symbols/numbers
            
        # Moderate numerical indicators (EXPANDED)
        numerical_words = [
            'calculate', 'measure', 'count', 'formula', 'equation', 'compute',
            'quantify', 'analyze', 'model', 'simulate', 'estimate', 'predict',
            'solve', 'derive', 'prove', 'theorem', 'hypothesis', 'experiment',
            'test', 'validate', 'verify', 'optimize', 'maximize', 'minimize'
        ]
        if any(word in topic_lower for word in numerical_words):
            score += 1
            
        # Special handling for clearly numerical topics (AI-enhanced with misspellings)
        short_numerical_topics = [
            'math', 'maths', 'physics', 'chemistry', 'calculus', 'algebra',
            'geometry', 'statistics', 'finance', 'economics', 'accounting',
            'engineering', 'programming', 'algorithms', 'data', 'analytics',
            'computation', 'coding', 'software', 'hardware', 'AI', 'ML',
            'atoms', 'atom', 'atomic', 'molecular', 'molecules', 'elements',
            # Add common misspellings (AI-enhanced)
            'mathamatics', 'mathimatics', 'phisics', 'chemestry', 'calculis',
            'algerba', 'programing', 'algorythm', 'finace', 'enginering'
        ]
        if any(topic in topic_lower for topic in short_numerical_topics):
            score += 3  # High boost for clearly numerical terms
            
        return score

    def suggest_optimal_question_type(self, topic: str) -> str:
        """
        Suggests the optimal question type based on topic analysis.
        Enhanced with AI spell correction.
        
        Returns:
            "conceptual", "numerical", or "mixed"
        """
        profile = self.get_topic_profile(topic)
        
        if profile["detected_type"] == "mixed":
            return "mixed"
        elif profile["detected_type"] == "conceptual":
            return "conceptual"
        elif profile["detected_type"] == "numerical":
            return "numerical"
        else:
            return "mixed"  # Safe default

    def get_topic_recommendations(self, topic: str) -> Dict[str, any]:
        """
        Get comprehensive recommendations for a topic including question types and difficulty suggestions.
        Enhanced with AI spell correction and user feedback.
        
        Returns:
            Dictionary with recommendations for UI guidance including correction information
        """
        profile = self.get_topic_profile(topic)
        optimal_type = self.suggest_optimal_question_type(topic)
        
        recommendations = {
            **profile,
            "optimal_question_type": optimal_type,
            "ui_recommendations": {
                "highlight_conceptual": profile["detected_type"] == "conceptual",
                "highlight_numerical": profile["detected_type"] == "numerical", 
                "highlight_mixed": profile["detected_type"] == "mixed",
                "disable_numerical": not profile["is_numerical_possible"]
            }
        }
        
        # Add AI correction feedback for UI
        if profile.get("spelling_corrected", False):
            recommendations["ui_feedback"] = {
                "show_correction_notice": True,
                "correction_message": f"Did you mean '{profile['corrected_topic']}'?",
                "corrections_made": profile["corrections_made"]
            }
        else:
            recommendations["ui_feedback"] = {
                "show_correction_notice": False,
                "correction_message": "",
                "corrections_made": []
            }
            
        return recommendations

    def get_spelling_suggestions(self, topic: str) -> List[str]:
        """
        Get spelling suggestions for a topic using AI correction algorithms
        
        Returns:
            List of suggested corrections
        """
        corrected_topic, corrections = self.spell_corrector.correct_topic_spelling(topic)
        
        if corrections:
            return [corrected_topic] + [correction.split(" → ")[1] for correction in corrections]
        
        return []

# Global instance for efficient reuse
_topic_analyzer_instance = None

def get_topic_analyzer() -> TopicAnalyzer:
    """Get the global TopicAnalyzer instance with semantic mapping enabled"""
    global _topic_analyzer_instance
    if _topic_analyzer_instance is None:
        _topic_analyzer_instance = TopicAnalyzer(use_semantic_mapping=True)
    return _topic_analyzer_instance
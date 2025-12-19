"""
Advanced Topic Modeling Module
Extracts keywords, named entities, and important terms from text.
Uses TF-IDF, NER, POS tagging, and position-based weighting.
"""

import re
import string
from typing import List, Dict, Set, Optional, Any
from collections import Counter
import logging
import math

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_nltk_data():
    """Downloads required NLTK data if not present."""
    required_packages = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('chunkers/maxent_ne_chunker_tab', 'maxent_ne_chunker_tab'),
        ('corpora/words', 'words'),
    ]

    for path, package in required_packages:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK package: {package}")
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                logger.warning(f"Could not download {package}: {e}")


# Ensure NLTK data is available on module load
ensure_nltk_data()


def calculate_dynamic_word_count(text_length: int, num_sentences: int) -> int:
    """
    Calculates optimal number of words based on content length.

    Args:
        text_length: Length of text in characters
        num_sentences: Number of sentences in text

    Returns:
        Optimal word count (between 25 and 200)
    """
    # Base calculation: ~1 keyword per 80 characters
    base_count = text_length // 80

    # Bonus for sentence diversity
    sentence_bonus = min(num_sentences // 4, 25)

    # Calculate total with bounds
    total = base_count + sentence_bonus
    result = max(25, min(200, total))

    logger.info(f"Dynamic word count: {result} (chars={text_length}, sentences={num_sentences})")
    return result


class AdvancedTopicExtractor:
    """
    Extracts important topics and keywords using multiple NLP techniques.
    """

    # Extended stop words for web content
    CUSTOM_STOP_WORDS = {
        # Verbs and auxiliaries
        'said', 'says', 'would', 'could', 'also', 'may', 'might', 'must',
        'shall', 'should', 'will', 'can', 'need', 'want', 'let', 'got',
        'get', 'gets', 'getting', 'make', 'makes', 'made', 'making',
        'take', 'takes', 'took', 'taking', 'give', 'gives', 'gave', 'giving',
        'come', 'comes', 'came', 'coming', 'go', 'goes', 'went', 'going',
        'know', 'knows', 'knew', 'knowing', 'think', 'thinks', 'thought',
        'see', 'sees', 'saw', 'seeing', 'use', 'uses', 'used', 'using',
        'find', 'finds', 'found', 'finding', 'tell', 'tells', 'told',
        'look', 'looks', 'looked', 'looking', 'seem', 'seems', 'seemed',
        'become', 'becomes', 'became', 'becoming', 'keep', 'keeps', 'kept',
        'put', 'puts', 'run', 'runs', 'ran', 'running', 'move', 'moves',
        'live', 'lives', 'lived', 'living', 'believe', 'believes', 'believed',
        'bring', 'brings', 'brought', 'happen', 'happens', 'happened',
        'write', 'writes', 'wrote', 'written', 'provide', 'provides', 'provided',
        'sit', 'sits', 'sat', 'stand', 'stands', 'stood', 'lose', 'loses', 'lost',
        'pay', 'pays', 'paid', 'meet', 'meets', 'met', 'include', 'includes',
        'continue', 'continues', 'continued', 'set', 'sets', 'learn', 'learns',
        'change', 'changes', 'changed', 'lead', 'leads', 'led', 'understand',
        'watch', 'watches', 'watched', 'follow', 'follows', 'followed',
        'stop', 'stops', 'stopped', 'create', 'creates', 'created',
        'speak', 'speaks', 'spoke', 'spoken', 'read', 'reads', 'allow',
        'add', 'adds', 'added', 'spend', 'spends', 'spent', 'grow', 'grows',
        'open', 'opens', 'opened', 'walk', 'walks', 'walked', 'win', 'wins',
        'offer', 'offers', 'offered', 'remember', 'remembers', 'love', 'loves',
        'consider', 'considers', 'considered', 'appear', 'appears', 'appeared',
        'buy', 'buys', 'bought', 'wait', 'waits', 'waited', 'serve', 'serves',
        'die', 'dies', 'died', 'send', 'sends', 'sent', 'expect', 'expects',
        'build', 'builds', 'built', 'stay', 'stays', 'stayed', 'fall', 'falls',
        'cut', 'cuts', 'reach', 'reaches', 'reached', 'kill', 'kills', 'killed',
        'remain', 'remains', 'remained', 'suggest', 'suggests', 'suggested',
        'raise', 'raises', 'raised', 'pass', 'passes', 'passed', 'sell', 'sells',
        'require', 'requires', 'required', 'report', 'reports', 'reported',
        'decide', 'decides', 'decided', 'pull', 'pulls', 'pulled',

        # Time words
        'year', 'years', 'month', 'months', 'week', 'weeks', 'day', 'days',
        'hour', 'hours', 'minute', 'minutes', 'second', 'seconds',
        'time', 'times', 'today', 'yesterday', 'tomorrow', 'now', 'then',
        'first', 'last', 'next', 'new', 'old', 'recent', 'recently',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'june', 'july', 'august',
        'september', 'october', 'november', 'december',

        # Quantity words
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'many', 'much', 'more', 'most', 'some', 'any', 'few', 'several',
        'all', 'both', 'each', 'every', 'other', 'another', 'such', 'no', 'none',
        'part', 'percent', 'number', 'half', 'total', 'less', 'least', 'enough',
        'million', 'billion', 'trillion', 'thousand', 'hundred',

        # Generic nouns
        'thing', 'things', 'way', 'ways', 'point', 'fact', 'case', 'place',
        'world', 'life', 'work', 'people', 'person', 'man', 'woman', 'group',
        'company', 'system', 'program', 'question', 'hand', 'area', 'end',
        'side', 'home', 'water', 'room', 'mother', 'father', 'state', 'word',
        'level', 'office', 'door', 'line', 'members', 'name', 'team', 'eye',
        'job', 'business', 'issue', 'kind', 'head', 'house', 'service', 'friend',
        'right', 'night', 'story', 'lot', 'game', 'country', 'school',

        # Web/article specific
        'according', 'reuters', 'associated', 'press', 'read', 'click',
        'here', 'share', 'facebook', 'twitter', 'email', 'print', 'comment',
        'comments', 'subscribe', 'sign', 'newsletter', 'advertisement',
        'sponsored', 'related', 'stories', 'article', 'post', 'page',
        'source', 'photo', 'image', 'video', 'file', 'copyright', 'rights',
        'privacy', 'policy', 'terms', 'conditions', 'contact', 'about',
        'menu', 'search', 'login', 'register', 'account', 'profile',
        'cookie', 'cookies', 'consent', 'accept', 'close', 'skip',
        'breaking', 'update', 'updated', 'published', 'written', 'author',
        'reporter', 'editor', 'staff', 'writer', 'contributor',

        # Filler/connector words
        'like', 'just', 'even', 'well', 'back', 'still', 'really', 'actually',
        'however', 'although', 'though', 'therefore', 'thus', 'hence',
        'example', 'including', 'include', 'includes', 'included',
        'called', 'known', 'based', 'around', 'within', 'without',
        'different', 'similar', 'same', 'various', 'certain', 'particular',
        'able', 'likely', 'possible', 'available', 'true', 'important',
        'good', 'better', 'best', 'bad', 'worse', 'worst', 'great', 'little',
        'long', 'short', 'high', 'low', 'small', 'large', 'big', 'young',
        'early', 'late', 'hard', 'major', 'public', 'local', 'full', 'special',
        'easy', 'clear', 'sure', 'real', 'common', 'current', 'free', 'past',
        'simple', 'whole', 'main', 'single', 'final', 'former', 'general',
        'okay', 'already', 'simply', 'maybe', 'perhaps', 'especially',
    }

    # Named entity types and their weights
    ENTITY_WEIGHTS = {
        'PERSON': 12.0,       # People names - HIGHEST priority
        'ORGANIZATION': 6.0,  # Companies, orgs
        'GPE': 8.0,           # Countries, cities - HIGH priority
        'LOCATION': 8.0,      # Geographic locations - HIGH priority
        'FACILITY': 6.0,      # Buildings, airports - increased
        'EVENT': 4.0,         # Named events
        'PRODUCT': 4.5,       # Products, services
        'MONEY': 3.0,         # Monetary values
        'DATE': 2.0,          # Dates
        'PERCENT': 3.0,       # Percentages
    }

    # Important parts of speech
    NOUN_TAGS = {'NN', 'NNS', 'NNP', 'NNPS'}
    ADJECTIVE_TAGS = {'JJ', 'JJR', 'JJS'}
    VERB_TAGS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

    def __init__(self, max_words: int = 80):
        """Initialize the topic extractor."""
        self.max_words = max_words
        self.lemmatizer = WordNetLemmatizer()
        self.word_sentences = {}  # Maps words to their source sentences
        self.word_contexts = {}   # Maps words to their context/meaning
        self.original_sentences = []  # Store original sentences

        # Load stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

        self.stop_words.update(self.CUSTOM_STOP_WORDS)

    def _build_word_sentence_map(self, sentences: List[str]) -> None:
        """
        Builds a mapping of words to their source sentences and generates context.
        """
        self.word_sentences = {}
        self.word_contexts = {}
        self.original_sentences = sentences

        for sentence in sentences:
            sentence_lower = sentence.lower()
            try:
                tokens = word_tokenize(sentence_lower)
            except Exception:
                tokens = sentence_lower.split()

            for token in tokens:
                if len(token) < 3:
                    continue
                if not re.match(r'^[a-z]+(-[a-z]+)*$', token):
                    continue
                if token in self.stop_words:
                    continue

                lemma = self.lemmatizer.lemmatize(token)
                if lemma not in self.stop_words:
                    # Store the shortest, most relevant sentence
                    if lemma not in self.word_sentences:
                        self.word_sentences[lemma] = sentence.strip()
                    elif len(sentence) < len(self.word_sentences[lemma]) and len(sentence) > 30:
                        self.word_sentences[lemma] = sentence.strip()

    def _generate_context(self, word: str, entity_type: str = None) -> str:
        """
        Generates accurate context/meaning description for a word.
        Uses entity type from NER and sentence context for precise descriptions.
        """
        sentence = self.word_sentences.get(word.lower(), '')
        word_lower = word.lower()

        # Check if word is capitalized in original (likely proper noun)
        is_capitalized = word and word[0].isupper()

        # Entity type mappings for clear descriptions
        entity_descriptions = {
            'PERSON': "Person mentioned in the article",
            'ORGANIZATION': "Organization or company mentioned",
            'ORG': "Organization or company mentioned",
            'GPE': "Geographic location (country, city, state)",
            'LOCATION': "Location or place referenced",
            'LOC': "Location or place referenced",
            'FACILITY': "Facility, building, or landmark",
            'FAC': "Facility, building, or landmark",
            'EVENT': "Event or occurrence discussed",
            'PRODUCT': "Product or service mentioned",
            'WORK_OF_ART': "Creative work referenced",
            'LAW': "Law, act, or regulation mentioned",
            'LANGUAGE': "Language referenced",
            'DATE': "Date or time period",
            'TIME': "Time reference",
            'MONEY': "Monetary value or financial figure",
            'PERCENT': "Percentage or statistical figure",
            'QUANTITY': "Numerical quantity mentioned",
            'ORDINAL': "Ordinal number (ranking)",
            'CARDINAL': "Cardinal number",
            'NORP': "Nationality, religious, or political group"
        }

        # Return entity-specific description if entity type is known
        if entity_type and entity_type in entity_descriptions:
            return entity_descriptions[entity_type]

        # Try to infer entity type from sentence context if not provided
        if sentence:
            sentence_lower = sentence.lower()

            # Organization indicators
            org_indicators = ['company', 'corporation', 'inc', 'corp', 'ltd', 'llc',
                           'group', 'enterprise', 'firm', 'business', 'organization',
                           'agency', 'institute', 'university', 'bank', 'foundation']
            if any(ind in sentence_lower for ind in org_indicators) and is_capitalized:
                return "Organization or company mentioned"

            # Person indicators
            person_indicators = ['said', 'told', 'announced', 'stated', 'according to',
                               'ceo', 'president', 'director', 'manager', 'chief',
                               'mr.', 'mrs.', 'ms.', 'dr.', 'prof.']
            if any(ind in sentence_lower for ind in person_indicators) and is_capitalized:
                return "Person mentioned in the article"

            # Location indicators
            location_indicators = ['located', 'based in', 'headquarters', 'city',
                                 'country', 'region', 'state', 'capital', 'town']
            if any(ind in sentence_lower for ind in location_indicators) and is_capitalized:
                return "Location or place referenced"

            # Technology/product indicators
            tech_indicators = ['technology', 'software', 'platform', 'app', 'system',
                             'device', 'product', 'service', 'tool', 'feature']
            if any(ind in sentence_lower for ind in tech_indicators):
                return "Technology or product mentioned"

            # Financial context
            financial_indicators = ['million', 'billion', 'percent', '%', 'dollar',
                                  '$', 'revenue', 'profit', 'growth', 'investment',
                                  'stock', 'market', 'price', 'cost', 'budget']
            if any(ind in sentence_lower for ind in financial_indicators):
                return "Related to financial or business data"

            # Trend/change context
            trend_indicators = ['increase', 'decrease', 'growth', 'decline', 'rise',
                              'fall', 'surge', 'drop', 'gain', 'loss', 'change']
            if any(ind in sentence_lower for ind in trend_indicators):
                return "Related to trends or changes"

            # News/reporting context
            news_indicators = ['reported', 'announced', 'revealed', 'disclosed',
                             'confirmed', 'denied', 'claimed', 'alleged']
            if any(ind in sentence_lower for ind in news_indicators):
                return "Topic from news report"

            # Research/study context
            research_indicators = ['study', 'research', 'found', 'discovered',
                                 'analysis', 'data', 'survey', 'findings']
            if any(ind in sentence_lower for ind in research_indicators):
                return "Related to research or study findings"

            # Policy/government context
            policy_indicators = ['government', 'policy', 'law', 'regulation',
                               'official', 'minister', 'parliament', 'congress']
            if any(ind in sentence_lower for ind in policy_indicators):
                return "Related to government or policy"

            # If capitalized but no specific context found
            if is_capitalized:
                return "Proper noun or named entity"

        # Default descriptions
        if is_capitalized:
            return "Named entity from the article"

        return "Key topic from the article"


    def extract_topics(
        self,
        text: str,
        quotes: Optional[List[str]] = None,
        statistics: Optional[List[str]] = None,
        dynamic_count: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extracts important topics from text.

        Args:
            text: Main text to analyze
            quotes: Extracted quotes
            statistics: Extracted statistics
            dynamic_count: Whether to use dynamic word count

        Returns:
            List of word dictionaries with word and weight
        """
        if not text or len(text.strip()) < 50:
            raise ValueError("Text is too short for analysis")

        logger.info(f"Analyzing text of {len(text)} characters")

        # Split into sentences
        sentences = self._split_sentences(text)

        # Build word-to-sentence mapping
        self._build_word_sentence_map(sentences)

        # Calculate dynamic word count
        if dynamic_count:
            self.max_words = calculate_dynamic_word_count(len(text), len(sentences))

        # Preprocess text
        processed_text = self._preprocess_text(text)

        if not processed_text or len(processed_text.split()) < 10:
            raise ValueError("Not enough content after preprocessing")

        # Extract using multiple methods
        tfidf_scores = self._extract_tfidf_keywords(processed_text, sentences)
        pos_scores = self._extract_pos_weighted(text)
        position_scores = self._extract_position_weighted(sentences)
        entity_scores = self._extract_named_entities(text)
        quote_scores = self._extract_from_quotes(quotes) if quotes else {}
        stat_scores = self._extract_from_statistics(statistics) if statistics else {}
        term_freq = self._calculate_term_frequency(text)

        # Combine all scores with context-aware weighting
        combined = self._combine_scores(
            tfidf_scores, pos_scores, position_scores,
            entity_scores, quote_scores, stat_scores, term_freq
        )

        # Filter and normalize
        filtered = self._filter_keywords(combined)
        normalized = self._normalize_weights(filtered)

        logger.info(f"Extracted {len(normalized)} keywords")
        return normalized

    def _split_sentences(self, text: str) -> List[str]:
        """Splits text into sentences."""
        try:
            return sent_tokenize(text)
        except Exception:
            return re.split(r'[.!?]+', text)

    def _preprocess_text(self, text: str) -> str:
        """Preprocesses text for analysis."""
        text = text.lower()

        # Remove URLs and emails
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\b\d+\b', '', text)

        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        # Filter and lemmatize
        processed = []
        for token in tokens:
            if len(token) < 3 or len(token) > 25:
                continue
            if not re.match(r'^[a-z]+(-[a-z]+)*$', token):
                continue
            if token in self.stop_words:
                continue

            lemma = self.lemmatizer.lemmatize(token)
            if lemma not in self.stop_words and len(lemma) >= 3:
                processed.append(lemma)

        return ' '.join(processed)

    def _extract_named_entities(self, text: str) -> Dict[str, float]:
        """Extracts named entities using NER."""
        entities = {}
        self.entity_types = {}  # Track entity types for context generation

        try:
            sentences = sent_tokenize(text)

            for sentence in sentences[:60]:  # Limit for performance
                try:
                    tokens = word_tokenize(sentence)
                    tagged = pos_tag(tokens)
                    tree = ne_chunk(tagged)

                    for subtree in tree:
                        if isinstance(subtree, Tree):
                            entity_type = subtree.label()
                            if entity_type in self.ENTITY_WEIGHTS:
                                # Get entity name
                                entity_name = ' '.join(
                                    word for word, tag in subtree.leaves()
                                )
                                entity_name = entity_name.strip()

                                if (len(entity_name) >= 2 and
                                    entity_name.lower() not in self.stop_words):
                                    weight = self.ENTITY_WEIGHTS[entity_type]
                                    key = entity_name.lower()
                                    entities[key] = entities.get(key, 0) + weight
                                    # Store entity type for context
                                    self.entity_types[key] = entity_type
                                    # Also store the sentence for this entity
                                    if key not in self.word_sentences:
                                        self.word_sentences[key] = sentence.strip()

                except Exception as e:
                    logger.debug(f"NER error: {e}")
                    continue

            logger.info(f"Extracted {len(entities)} named entities")

        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")

        return entities

    def _extract_from_quotes(self, quotes: List[str]) -> Dict[str, float]:
        """Extracts keywords from quotes."""
        scores = Counter()

        for quote in quotes:
            tokens = self._get_clean_tokens(quote)
            for token in tokens:
                scores[token] += 3.0  # Quote bonus

        return dict(scores)

    def _extract_from_statistics(self, statistics: List[str]) -> Dict[str, float]:
        """Extracts keywords from statistical context."""
        scores = Counter()

        stat_keywords = {
            'growth', 'increase', 'decrease', 'rise', 'fall', 'drop',
            'revenue', 'profit', 'loss', 'market', 'share', 'rate',
            'gdp', 'inflation', 'employment', 'unemployment', 'sales',
            'production', 'output', 'investment', 'return', 'yield',
            'performance', 'index', 'average', 'median', 'forecast',
        }

        for stat in statistics:
            tokens = self._get_clean_tokens(stat)
            for token in tokens:
                if token in stat_keywords:
                    scores[token] += 4.0
                else:
                    scores[token] += 2.0

        return dict(scores)

    def _extract_tfidf_keywords(
        self, processed_text: str, sentences: List[str]
    ) -> Dict[str, float]:
        """Extracts keywords using TF-IDF."""
        try:
            # Process sentences
            processed_sentences = []
            for sent in sentences:
                processed = self._preprocess_text(sent)
                if len(processed.split()) >= 3:
                    processed_sentences.append(processed)

            if len(processed_sentences) < 2:
                processed_sentences = [processed_text]

            # TF-IDF with bigrams
            vectorizer = TfidfVectorizer(
                max_features=300,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.85,
                stop_words=list(self.stop_words)
            )

            tfidf_matrix = vectorizer.fit_transform(processed_sentences)
            feature_names = vectorizer.get_feature_names_out()
            scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

            keywords = {}
            for word, score in zip(feature_names, scores):
                if score > 0:
                    # Filter bad bigrams
                    if ' ' in word:
                        parts = word.split()
                        if any(len(p) < 3 or p in self.stop_words for p in parts):
                            continue
                    keywords[word] = float(score)

            return keywords

        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}")
            return {}

    def _extract_pos_weighted(self, text: str) -> Dict[str, float]:
        """Extracts keywords weighted by part-of-speech."""
        try:
            tokens = word_tokenize(text.lower())

            # Filter
            filtered = [
                t for t in tokens
                if len(t) >= 3 and re.match(r'^[a-z]+(-[a-z]+)*$', t)
                and t not in self.stop_words
            ]

            tagged = pos_tag(filtered)
            scores = Counter()

            for word, tag in tagged:
                lemma = self.lemmatizer.lemmatize(word)
                if lemma in self.stop_words:
                    continue

                # Weight by POS
                if tag in self.NOUN_TAGS:
                    scores[lemma] += 4.0
                elif tag in self.ADJECTIVE_TAGS:
                    scores[lemma] += 2.0
                elif tag in self.VERB_TAGS:
                    scores[lemma] += 1.5
                else:
                    scores[lemma] += 0.5

            return dict(scores)

        except Exception as e:
            logger.warning(f"POS extraction failed: {e}")
            return {}

    def _extract_position_weighted(self, sentences: List[str]) -> Dict[str, float]:
        """Weights keywords by position in text."""
        scores = Counter()

        if not sentences:
            return scores

        # First sentence - highest weight
        first_tokens = self._get_clean_tokens(sentences[0])
        for token in first_tokens:
            scores[token] += 3.0

        # First 5 sentences
        for sent in sentences[:5]:
            for token in self._get_clean_tokens(sent):
                scores[token] += 1.5

        # Last sentence (conclusion)
        if len(sentences) > 1:
            for token in self._get_clean_tokens(sentences[-1]):
                scores[token] += 1.0

        return dict(scores)

    def _get_clean_tokens(self, text: str) -> List[str]:
        """Gets clean, filtered tokens from text."""
        text = text.lower()
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        result = []
        for token in tokens:
            if (len(token) >= 3 and
                re.match(r'^[a-z]+(-[a-z]+)?$', token) and
                token not in self.stop_words):
                lemma = self.lemmatizer.lemmatize(token)
                if lemma not in self.stop_words:
                    result.append(lemma)

        return result

    def _calculate_term_frequency(self, text: str) -> Dict[str, float]:
        """Calculate term frequency for each word in the text."""
        try:
            tokens = word_tokenize(text.lower())
        except Exception:
            tokens = text.lower().split()

        # Count occurrences
        word_counts = Counter()
        for token in tokens:
            if len(token) >= 3 and re.match(r'^[a-z]+(-[a-z]+)*$', token):
                if token not in self.stop_words:
                    lemma = self.lemmatizer.lemmatize(token)
                    if lemma not in self.stop_words:
                        word_counts[lemma] += 1

        # Convert to frequency (log-scaled to prevent extreme values)
        total = sum(word_counts.values())
        if total == 0:
            return {}

        freq = {}
        for word, count in word_counts.items():
            # Log scale: log(1 + count/total) to get more meaningful differences
            freq[word] = math.log(1 + (count / total) * 100)

        return freq

    def _combine_scores(
        self,
        tfidf: Dict[str, float],
        pos: Dict[str, float],
        position: Dict[str, float],
        entities: Dict[str, float],
        quotes: Dict[str, float],
        statistics: Dict[str, float],
        term_freq: Dict[str, float] = None
    ) -> Dict[str, float]:
        """Combines scores from all extraction methods based on article context."""
        combined = Counter()

        # Normalize each score set to 0-1 range
        def normalize(scores: Dict[str, float]) -> Dict[str, float]:
            if not scores:
                return {}
            max_val = max(scores.values())
            min_val = min(scores.values())
            range_val = max_val - min_val if max_val != min_val else 1
            return {k: (v - min_val) / range_val for k, v in scores.items()}

        norm_tfidf = normalize(tfidf)
        norm_pos = normalize(pos)
        norm_position = normalize(position)
        norm_entities = normalize(entities)
        norm_quotes = normalize(quotes)
        norm_stats = normalize(statistics)
        norm_freq = normalize(term_freq) if term_freq else {}

        # Combine with weights
        all_words = (
            set(norm_tfidf.keys()) |
            set(norm_pos.keys()) |
            set(norm_position.keys()) |
            set(norm_entities.keys()) |
            set(norm_quotes.keys()) |
            set(norm_stats.keys()) |
            set(norm_freq.keys())
        )

        for word in all_words:
            # Base score from multiple signals
            tfidf_score = norm_tfidf.get(word, 0)
            pos_score = norm_pos.get(word, 0)
            position_score = norm_position.get(word, 0)
            entity_score = norm_entities.get(word, 0)
            quote_score = norm_quotes.get(word, 0)
            stat_score = norm_stats.get(word, 0)
            freq_score = norm_freq.get(word, 0)

            # Check entity type for context-based weighting
            entity_type = getattr(self, 'entity_types', {}).get(word.lower())

            # Calculate weighted score based on context
            if entity_type in ('PERSON', 'ORGANIZATION'):
                # Names and organizations: high base + frequency boost
                score = 0.6 + (entity_score * 0.2) + (freq_score * 0.15) + (position_score * 0.05)
            elif entity_type in ('GPE', 'LOCATION', 'FACILITY'):
                # Locations: moderately high base + context
                score = 0.5 + (entity_score * 0.2) + (freq_score * 0.2) + (position_score * 0.1)
            elif entity_type:
                # Other entities
                score = 0.4 + (entity_score * 0.3) + (freq_score * 0.2) + (tfidf_score * 0.1)
            else:
                # Regular keywords: balanced contextual scoring
                score = (
                    tfidf_score * 0.25 +      # Statistical importance
                    freq_score * 0.20 +        # How often it appears
                    pos_score * 0.15 +         # Part of speech relevance
                    position_score * 0.15 +    # Where it appears in article
                    quote_score * 0.15 +       # Appears in quotes
                    stat_score * 0.10          # Related to statistics
                )

            combined[word] = min(1.0, score)  # Cap at 1.0

        return dict(combined)

    def _filter_keywords(self, keywords: Dict[str, float]) -> Dict[str, float]:
        """Filters and cleans keywords."""
        filtered = {}

        for word, score in keywords.items():
            # Skip short words
            if len(word) < 3:
                continue

            # Skip pure numbers
            if word.isdigit():
                continue

            # Skip stop words
            if word in self.stop_words:
                continue

            # Skip if contains only single char parts
            parts = word.split()
            if any(len(p) < 2 for p in parts):
                continue

            filtered[word] = score

        # Sort and limit
        sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:self.max_words])

    def _normalize_weights(self, keywords: Dict[str, float]) -> List[Dict[str, Any]]:
        """Normalizes weights and formats output with sentence and context."""
        if not keywords:
            return []

        max_weight = max(keywords.values())
        min_weight = min(keywords.values())
        weight_range = max_weight - min_weight if max_weight != min_weight else 1

        result = []
        for word, weight in keywords.items():
            # Normalize to 0.15 - 1.0 range
            normalized = 0.15 + (0.85 * (weight - min_weight) / weight_range)

            # Get entity type if available - try multiple matching strategies
            entity_type = None
            entity_types_dict = getattr(self, 'entity_types', {})
            word_lower = word.lower()
            
            # Direct match
            if word_lower in entity_types_dict:
                entity_type = entity_types_dict[word_lower]
            else:
                # Try partial match for multi-word entities
                for entity_key, etype in entity_types_dict.items():
                    if word_lower in entity_key or entity_key in word_lower:
                        entity_type = etype
                        break

            # Get sentence and context
            sentence = self.word_sentences.get(word.lower(), '')
            if not sentence:
                # Try to find sentence with original casing
                for key, val in self.word_sentences.items():
                    if key == word.lower():
                        sentence = val
                        break

            # Truncate long sentences
            if len(sentence) > 200:
                sentence = sentence[:197] + "..."

            # Generate context/meaning
            context = self._generate_context(word, entity_type)

            result.append({
                'word': word,
                'weight': round(normalized, 4),
                'type': 'keyword',
                'sentence': sentence,
                'context': context,
                'entity_type': entity_type
            })

        result.sort(key=lambda x: x['weight'], reverse=True)
        return result


# Global extractor instance
extractor = AdvancedTopicExtractor()


def extract_topics(
    text: str,
    max_words: int = 80,
    quotes: Optional[List[str]] = None,
    statistics: Optional[List[str]] = None,
    key_numbers: Optional[List[Dict]] = None,
    dynamic_count: bool = True
) -> List[Dict[str, Any]]:
    """
    Main function to extract topics from text.

    Args:
        text: Text to analyze
        max_words: Maximum keywords (if dynamic_count=False)
        quotes: Extracted quotes
        statistics: Extracted statistics
        key_numbers: Key numbers to include
        dynamic_count: Use dynamic word count

    Returns:
        List of word dictionaries
    """
    if not dynamic_count and max_words != extractor.max_words:
        extractor.max_words = max_words

    # Extract keywords
    keywords = extractor.extract_topics(
        text,
        quotes=quotes,
        statistics=statistics,
        dynamic_count=dynamic_count
    )

    # Add key numbers as special items
    if key_numbers:
        num_slots = min(len(key_numbers), max(15, len(keywords) // 3))

        for kn in key_numbers[:num_slots]:
            # Generate context for the number
            number_type = kn.get('type', 'stat')
            context_map = {
                'percent': 'Statistical percentage from the article',
                'money': 'Financial figure mentioned in the article',
                'large_number': 'Significant numerical value',
                'year': 'Important year referenced',
                'multiplier': 'Growth or comparison factor',
                'rank': 'Ranking or position mentioned',
                'stat': 'Key statistic from the article'
            }
            context = context_map.get(number_type, 'Important number from the article')

            keywords.append({
                'word': kn['display'],
                'weight': kn['weight'],
                'type': 'number',
                'number_type': number_type,
                'sentence': kn.get('sentence', ''),
                'context': context,
                'entity_type': None
            })

    # Re-sort by weight
    keywords.sort(key=lambda x: x['weight'], reverse=True)
    return keywords

"""
Universal Web Crawler Module
Extracts text, quotes, statistics, and key numbers from ANY website.
Supports all browsers and domains worldwide.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import os
from typing import Optional, Dict, List, Any
import logging
import random
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalCrawler:
    """
    A robust web crawler that can extract content from any website.
    Uses rotating user agents and handles various edge cases.
    """

    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]

    # Content selectors for different website types
    CONTENT_SELECTORS = [
        # MSN
        ('.articleBody', None),
        ('[data-testid="article-body"]', None),
        ('.article-body-content', None),
        # AP News
        ('.RichTextStoryBody', None),
        ('.RichTextBody', None),
        # CNN
        ('.article__content', None),
        ('.zn-body__paragraph', None),
        # Reuters
        ('[data-testid="paragraph-"]', None),
        # News sites
        ('article', None),
        ('[role="main"]', None),
        ('.article-body', None),
        ('.article-content', None),
        ('.story-body', None),
        ('.post-content', None),
        ('.entry-content', None),
        ('.content-body', None),
        # BBC
        ('[data-component="text-block"]', None),
        # NY Times
        ('.StoryBodyCompanionColumn', None),
        # Generic
        ('main', None),
        ('#content', None),
        ('.content', None),
        ('#main-content', None),
        ('.main-content', None),
        # Wikipedia
        ('#mw-content-text', None),
        ('.mw-parser-output', None),
        # Additional news selectors
        ('[data-testid="article-body"]', None),
        ('.article__body', None),
        ('.story__content', None),
    ]

    def __init__(self):
        self.session = requests.Session()
        self._rotate_user_agent()

    def _rotate_user_agent(self):
        """Rotates to a random user agent."""
        self.session.headers.update({
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
        })

    def fetch_content(self, url_or_path: str) -> Dict[str, Any]:
        """
        Fetches and processes content from a URL or local file.

        Args:
            url_or_path: A URL or local file path

        Returns:
            Dictionary with extracted content
        """
        logger.info(f"Processing: {url_or_path}")

        if self._is_local_file(url_or_path):
            return self._fetch_local_file(url_or_path)
        return self._fetch_url(url_or_path)

    def _is_local_file(self, path: str) -> bool:
        """Checks if the path is a local file."""
        if path.startswith('file://'):
            return True
        if os.path.exists(path):
            return True
        # Windows paths like C:\
        if len(path) > 2 and path[1] == ':':
            return os.path.exists(path)
        return False

    def _fetch_local_file(self, path: str) -> Dict[str, Any]:
        """Fetches content from a local file."""
        if path.startswith('file://'):
            path = path[7:]
            # Handle Windows file:///C:/ format
            if path.startswith('/') and len(path) > 2 and path[2] == ':':
                path = path[1:]

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        return self._process_html(content, os.path.basename(path), 'local')

    def _fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetches content from a URL."""
        logger.info(f"Fetching URL: {url}")
        domain = self._get_domain(url)

        # Use fresh headers for each request - no persistent session/cookies
        # Note: Omitting Accept-Encoding as some sites (AP News) return JS-only pages with it
        headers = {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        try:
            response = None
            # Try with different configurations
            for attempt in range(3):
                try:
                    # Use fresh request each time - no session cookies
                    response = requests.get(
                        url,
                        headers=headers,
                        timeout=60,
                        allow_redirects=True,
                        verify=True
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.SSLError:
                    # Retry without SSL verification
                    response = requests.get(
                        url,
                        headers=headers,
                        timeout=60,
                        allow_redirects=True,
                        verify=False
                    )
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == 2:
                        raise
                    time.sleep(1)
                    # Try different user agent
                    headers['User-Agent'] = random.choice(self.USER_AGENTS)

            if response is None:
                raise ValueError("Failed to fetch URL after all attempts")

            logger.info(f"Received {len(response.content)} bytes from {domain}")

            return self._process_html(response.content, domain, domain)

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch URL: {e}")
            raise ValueError(f"Failed to fetch URL: {str(e)}")

    def _process_html(self, content: Any, title_fallback: str, source: str) -> Dict[str, Any]:
        """Processes HTML content and extracts meaningful text."""
        # Parse HTML
        if isinstance(content, bytes):
            soup = BeautifulSoup(content, 'lxml')
        else:
            soup = BeautifulSoup(content, 'lxml')

        # Remove unwanted elements
        for tag in soup.find_all([
            'script', 'style', 'noscript', 'iframe', 'nav',
            'footer', 'aside', 'header', 'form', 'button',
            'svg', 'img', 'video', 'audio', 'canvas'
        ]):
            tag.decompose()

        # Remove common non-content elements
        for selector in ['.ad', '.advertisement', '.sidebar', '.comments',
                        '.social', '.share', '.related', '.newsletter',
                        '[class*="cookie"]', '[class*="popup"]', '[class*="banner"]']:
            for element in soup.select(selector):
                element.decompose()

        # Get title
        title = self._get_title(soup) or title_fallback

        # Try to find main content using various selectors
        text = ''
        for selector, attr in self.CONTENT_SELECTORS:
            try:
                if attr:
                    elements = soup.find_all(selector, {attr: True})
                else:
                    elements = soup.select(selector)

                if elements:
                    for elem in elements:
                        text += elem.get_text(separator=' ', strip=True) + ' '
                    if len(text) > 500:  # Found substantial content
                        break
            except Exception:
                continue

        # Fallback to body text if no specific content found
        if not text or len(text) < 200:
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)

        if not text or len(text) < 100:
            raise ValueError(
                "Could not extract meaningful text from this page. "
                "The website may require JavaScript to display content, "
                "or the content may be behind a paywall/login."
            )

        # Extract special content
        quotes = self._extract_quotes(text)
        key_numbers = self._extract_key_numbers(text)
        statistics = self._extract_statistics(text)

        # Clean text
        text = self._clean_text(text)

        logger.info(f"Extracted {len(text)} chars, {len(quotes)} quotes, {len(key_numbers)} numbers")

        return {
            'title': title[:200] if title else 'Unknown Article',
            'text': text,
            'source': source,
            'description': self._get_description(soup),
            'quotes': quotes,
            'statistics': statistics,
            'key_numbers': key_numbers,
            'text_length': len(text)
        }

    def _get_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extracts the page title."""
        # Try Open Graph title first
        og = soup.find('meta', property='og:title')
        if og and og.get('content'):
            return og['content'].strip()[:200]

        # Try h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)[:200]

        # Try title tag
        title = soup.find('title')
        if title:
            text = title.get_text(strip=True)
            # Remove common suffixes
            for sep in ['|', '-', ':', '::']:
                if sep in text:
                    text = text.split(sep)[0].strip()
            return text[:200]

        return None

    def _get_description(self, soup: BeautifulSoup) -> str:
        """Extracts the page description."""
        # Try meta description
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta and meta.get('content'):
            return meta['content'].strip()[:500]

        # Try Open Graph description
        og = soup.find('meta', property='og:description')
        if og and og.get('content'):
            return og['content'].strip()[:500]

        return ''

    def _extract_quotes(self, text: str) -> List[str]:
        """Extracts quoted text from the content."""
        quotes = []
        seen = set()

        # Various quote patterns
        patterns = [
            r'"([^"]{20,300})"',  # Double quotes
            r'"([^"]{20,300})"',  # Smart double quotes
            r"'([^']{20,300})'",  # Single quotes (less common)
            r'«([^»]{20,300})»',  # French quotes
        ]

        for pattern in patterns:
            for match in re.findall(pattern, text):
                clean_quote = match.strip()
                if clean_quote and clean_quote.lower() not in seen:
                    seen.add(clean_quote.lower())
                    quotes.append(clean_quote)

        return quotes[:20]  # Limit to 20 quotes

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences for context extraction."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _find_sentence_for_number(self, sentences: List[str], match_text: str,
                                   match_pos: int, full_text: str) -> str:
        """Finds the sentence containing a specific number for tooltip display."""
        for sentence in sentences:
            if match_text in sentence:
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 250:
                    return clean_sentence[:247] + '...'
                return clean_sentence
        start = max(0, match_pos - 100)
        end = min(len(full_text), match_pos + 100)
        context = full_text[start:end].strip()
        if start > 0:
            first_period = context.find('. ')
            if first_period > 0 and first_period < 50:
                context = context[first_period + 2:]
        if end < len(full_text):
            last_period = context.rfind('. ')
            if last_period > len(context) - 50:
                context = context[:last_period + 1]
        if len(context) > 250:
            return context[:247] + '...'
        return context

    def _extract_key_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extracts important numbers and statistics from text with their source sentences."""
        numbers = []
        seen = set()
        sentences = self._split_into_sentences(text)

        patterns = [
            # Percentages
            (r'(\d+(?:\.\d+)?)\s*%', 'percent', 0.9),
            # Money amounts
            (r'\$\s*(\d+(?:\.\d+)?)\s*(billion|million|trillion|B|M|T|k|K)?', 'money', 0.95),
            (r'(\d+(?:\.\d+)?)\s*(billion|million|trillion)\s*(?:dollars?|USD)?', 'money', 0.9),
            # Large numbers with units
            (r'(\d+(?:\.\d+)?)\s*(billion|million|trillion|thousand)', 'large_number', 0.85),
            # Years (recent and historical)
            (r'\b(20[0-2][0-9]|19[5-9][0-9])\b', 'year', 0.5),
            # Large integers with commas
            (r'(\d{1,3}(?:,\d{3})+)', 'large_int', 0.7),
            # Multipliers (2x, 3x, etc.)
            (r'(\d+(?:\.\d+)?)[xX]', 'multiplier', 0.8),
            # Rankings
            (r'#(\d+)', 'rank', 0.6),
            (r'(?:top|ranked?)\s*#?\s*(\d+)', 'rank', 0.6),
        ]

        for pattern, num_type, weight in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                num = match.group(1)
                match_text = match.group(0)

                # Format display based on type
                if num_type == 'percent':
                    display = f"{num}%"
                elif num_type == 'money':
                    unit = ''
                    if match.lastindex and match.lastindex >= 2 and match.group(2):
                        unit_raw = match.group(2).lower()
                        unit_map = {
                            'billion': 'B', 'million': 'M', 'trillion': 'T',
                            'b': 'B', 'm': 'M', 't': 'T', 'k': 'K'
                        }
                        unit = unit_map.get(unit_raw, unit_raw[0].upper() if unit_raw else '')
                    display = f"${num}{unit}"
                elif num_type == 'large_number':
                    unit = match.group(2)
                    display = f"{num} {unit}"
                elif num_type == 'multiplier':
                    display = f"{num}x"
                elif num_type == 'rank':
                    display = f"#{num}"
                else:
                    display = num

                # Find the sentence containing this number
                sentence = self._find_sentence_for_number(sentences, match_text, match.start(), text)

                # Skip duplicates and very long displays
                if display not in seen and len(display) < 25:
                    seen.add(display)
                    numbers.append({
                        'display': display,
                        'type': num_type,
                        'weight': weight,
                        'raw': num,
                        'sentence': sentence
                    })

        # Sort by weight
        numbers.sort(key=lambda x: x['weight'], reverse=True)
        return numbers[:50]  # Limit to 50 numbers

    def _extract_statistics(self, text: str) -> List[str]:
        """Extracts statistical statements from text."""
        stats = []

        # Patterns for statistical statements
        patterns = [
            r'(?:increased?|decreased?|grew?|fell?|rose?|dropped?)\s+(?:by\s+)?(\d+(?:\.\d+)?%)',
            r'(\d+(?:\.\d+)?%)\s+(?:increase|decrease|growth|decline|rise|fall|drop)',
            r'(?:up|down)\s+(\d+(?:\.\d+)?%)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end].strip()
                if context and context not in stats:
                    stats.append(context)

        return stats[:15]

    def _clean_text(self, text: str) -> str:
        """Cleans extracted text."""
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove repeated punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        return text.strip()

    def _get_domain(self, url: str) -> str:
        """Extracts domain from URL."""
        try:
            domain = urlparse(url).netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return 'unknown'


# Global crawler instance
crawler = UniversalCrawler()


def fetch_article_text(url_or_path: str) -> Dict[str, Any]:
    """
    Convenience function to fetch article content.

    Args:
        url_or_path: A URL or local file path

    Returns:
        Dictionary with extracted content
    """
    return crawler.fetch_content(url_or_path)

"""
FastAPI Backend for 3D Word Cloud Visualization
Provides API endpoints for article analysis and topic extraction.
Supports ALL websites and extracts keywords, quotes, statistics, and key numbers.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
import logging
import re
import os
import webbrowser
import threading

from crawler import fetch_article_text
from topic_modeling import extract_topics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="3D Word Cloud API",
    description="API for analyzing web articles and generating word cloud data. Supports any URL from any website.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Pydantic Models -----

class AnalyzeRequest(BaseModel):
    """Request model for article analysis."""
    url: str

    @validator('url')
    def validate_url(cls, v):
        """Validates URL format. Accepts any URL of any length from any domain."""
        v = v.strip()

        # Allow local file paths
        if os.path.exists(v) or v.startswith('file://'):
            return v
        # Windows paths (C:\...)
        if len(v) > 2 and v[1] == ':':
            return v

        # Simple URL validation - accept any http/https URL of any length
        if v.startswith('http://') or v.startswith('https://'):
            # Just verify there is something after the protocol
            rest = v.split('://', 1)[1] if '://' in v else ''
            if len(rest) > 0:
                return v

        raise ValueError('Invalid URL format. Please enter a valid URL starting with http:// or https://')


class WordItem(BaseModel):
    """Model for a single word in the word cloud."""
    word: str
    weight: float
    type: Optional[str] = "keyword"
    number_type: Optional[str] = None
    sentence: Optional[str] = ""
    context: Optional[str] = ""
    entity_type: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """Response model for article analysis."""
    words: List[WordItem]
    title: str
    source: str
    word_count: int
    article_length: int
    quotes_found: int
    statistics_found: int
    key_numbers_found: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


class SampleUrl(BaseModel):
    """Sample URL for suggestions."""
    url: str
    description: str


class SampleUrlsResponse(BaseModel):
    """Response with sample URLs."""
    urls: List[SampleUrl]


# ----- API Endpoints -----

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info."""
    return HealthResponse(
        status="online",
        message="3D Word Cloud API v2.0 - Supports any URL from any website worldwide"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="API is operational"
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_article(request: AnalyzeRequest):
    """
    Analyzes a web article and returns word cloud data.

    - Fetches content from any URL
    - Extracts keywords, named entities, quotes, and statistics
    - Returns structured data for 3D visualization
    """
    logger.info(f"Received analyze request for: {request.url}")

    try:
        # Fetch article content
        logger.info("Fetching article content...")
        article_data = fetch_article_text(request.url)

        text = article_data.get('text', '')
        title = article_data.get('title', 'Unknown Article')
        source = article_data.get('source', 'Unknown Source')
        quotes = article_data.get('quotes', [])
        statistics = article_data.get('statistics', [])
        key_numbers = article_data.get('key_numbers', [])

        logger.info(
            f"Fetched: {len(text)} chars, {len(quotes)} quotes, "
            f"{len(statistics)} stats, {len(key_numbers)} key numbers"
        )

        if not text or len(text.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from this article. "
                       "The page may be heavily JavaScript-based or require authentication."
            )

        # Extract topics
        logger.info("Extracting topics and keywords...")
        words = extract_topics(
            text=text,
            quotes=quotes,
            statistics=statistics,
            key_numbers=key_numbers,
            dynamic_count=True
        )

        if not words:
            raise HTTPException(
                status_code=400,
                detail="Could not extract meaningful topics from this article."
            )

        # Convert to response format
        processed_words = []
        for w in words:
            processed_words.append(WordItem(
                word=w.get('word', ''),
                weight=w.get('weight', 0.5),
                type=w.get('type', 'keyword'),
                number_type=w.get('number_type'),
                sentence=w.get('sentence', ''),
                context=w.get('context', ''),
                entity_type=w.get('entity_type')
            ))

        logger.info(f"Successfully extracted {len(processed_words)} items")

        return AnalyzeResponse(
            words=processed_words,
            title=title,
            source=source,
            word_count=len(processed_words),
            article_length=len(text),
            quotes_found=len(quotes),
            statistics_found=len(statistics),
            key_numbers_found=len(key_numbers)
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing article: {str(e)}"
        )


@app.get("/sample-urls", response_model=SampleUrlsResponse)
async def get_sample_urls():
    """Returns sample URLs for testing."""
    return SampleUrlsResponse(
        urls=[
            SampleUrl(
                url="https://en.wikipedia.org/wiki/Artificial_intelligence",
                description="Wikipedia - Artificial Intelligence"
            ),
            SampleUrl(
                url="https://en.wikipedia.org/wiki/Climate_change",
                description="Wikipedia - Climate Change"
            ),
            SampleUrl(
                url="https://www.bbc.com/news",
                description="BBC News"
            ),
            SampleUrl(
                url="https://www.reuters.com/",
                description="Reuters"
            ),
            SampleUrl(
                url="https://www.cnn.com/",
                description="CNN"
            ),
            SampleUrl(
                url="https://www.theguardian.com/",
                description="The Guardian"
            ),
            SampleUrl(
                url="https://techcrunch.com/",
                description="TechCrunch"
            ),
            SampleUrl(
                url="https://www.nytimes.com/",
                description="New York Times"
            ),
        ]
    )


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("=" * 60)
    logger.info("Starting 3D Word Cloud API v2.0")
    logger.info("Features: Universal URL support, NER, Quotes, Statistics")
    logger.info("API Docs: http://localhost:8001/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Shutting down 3D Word Cloud API...")


def open_browser():
    """Opens the API docs in browser after a short delay."""
    import time
    time.sleep(1.5)
    webbrowser.open("http://localhost:8001/docs")


if __name__ == "__main__":
    import uvicorn

    # Open browser in background thread
    # threading.Thread(target=open_browser, daemon=True).start()

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

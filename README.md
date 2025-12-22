# 3D Word Cloud Visualizer

An interactive web application that visualizes article topics as a stunning 3D word cloud. Enter any article URL and watch as keywords, named entities, and statistics come to life in an immersive 3D space.

 Demo video of the tool: https://drive.google.com/file/d/1D81BBw3MKkHFDnywkbbr9mOf0Rn6kj9t/view?usp=sharing

## Features

- **Universal URL Support**: Works with any news article, blog post, Wikipedia page, or web content from any website worldwide
- **Advanced NLP Processing**: Extracts keywords using TF-IDF, Named Entity Recognition (NER), and POS tagging
- **Dynamic Word Count**: Automatically adjusts the number of keywords based on article length
- **Extracts Special Content**:
  - Quoted text from articles
  - Statistics and percentages
  - Key numbers (money, years, large numbers)
  - Named entities (people, organizations, locations)
- **Beautiful 3D Visualization**:
  - Interactive rotating word cloud
  - Words sized by relevance
  - Color-coded by importance/relevance/weight (importance, weight and relevance are the same thing). The backend function displays the weight which is essentially the same     as relevance.
  - Hover effects and animations
  - Zoom and rotate controls
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

### Frontend
- **React 18** with TypeScript
- **React Three Fiber** (Three.js for React)
- **@react-three/drei** for 3D helpers
- **Vite** for fast development
- **Axios** for API calls

### Backend
- **Python 3.9+** with FastAPI
- **NLTK** for natural language processing
- **scikit-learn** for TF-IDF vectorization
- **BeautifulSoup4** for web scraping
- **lxml** for HTML parsing

## Quick Start

### Prerequisites
- Python 3.9 or higher
- Node.js 18 or higher
- npm or yarn

### Installation

#### Windows

1. Clone or download the repository:
```bash
git clone https://github.com/yourusername/3D-Word-Cloud-Amrutha.git
cd 3D-Word-Cloud-Amrutha
```

2. Run the setup script (installs all dependencies):
```bash
setup.bat
```

3. Start the application:
```bash
start.bat
```

The application will automatically open in your browser at http://localhost:3000

#### macOS / Linux

1. Clone or download the repository:
```bash
git clone https://github.com/yourusername/3D-Word-Cloud-Amrutha.git
cd 3D-Word-Cloud-Amrutha
```

2. Make scripts executable and run setup:
```bash
chmod +x setup.sh start.sh
./setup.sh
```

3. Start the application:
```bash
./start.sh
```

### Manual Installation

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The backend API will be available at http://localhost:8001

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at http://localhost:3000

## Usage

1. **Enter a URL**: Paste any article URL into the input field
2. **Click "Generate Word Cloud"**: The system will fetch and analyze the article
3. **Interact with the visualization**:
   - **Drag** to rotate the cloud
   - **Scroll** to zoom in/out
   - **Hover** over words to see details
   

### Sample URLs to Try

- https://en.wikipedia.org/wiki/Artificial_intelligence
- https://en.wikipedia.org/wiki/Climate_change
- https://www.bbc.com/news
- https://www.reuters.com/
- Any news article from CNN, The Guardian, NYTimes, etc.

## API Documentation

The backend provides a REST API with the following endpoints:

### POST /analyze
Analyzes an article and returns word cloud data.

**Request:**
```json
{
  "url": "https://example.com/article"
}
```

**Response:**
```json
{
  "words": [
    {"word": "technology", "weight": 1.0, "type": "keyword"},
    {"word": "45%", "weight": 0.9, "type": "number", "number_type": "percent"}
  ],
  "title": "Article Title",
  "source": "example.com",
  "word_count": 75,
  "article_length": 5000,
  "quotes_found": 3,
  "statistics_found": 2,
  "key_numbers_found": 8
}
```

### GET /health
Health check endpoint.

### GET /sample-urls
Returns a list of sample URLs for testing.

Full API documentation is available at http://localhost:8001/docs (Swagger UI)

## Project Structure

```
3D-Word-Cloud-Ruxst/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── crawler.py           # Universal web crawler
│   ├── topic_modeling.py    # NLP and keyword extraction
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── WordCloud3D.tsx   # Main 3D visualization
│   │   │   ├── URLInput.tsx      # URL input form
│   │   │   └── LoadingSpinner.tsx
│   │   ├── hooks/
│   │   │   └── useAnalyze.ts     # API hook
│   │   ├── types/
│   │   │   └── index.ts          # TypeScript types
│   │   ├── App.tsx               # Main app component
│   │   ├── main.tsx              # Entry point
│   │   └── index.css             # Styles
│   ├── package.json
│   └── vite.config.ts
├── setup.bat                # Windows setup script
├── setup.sh                 # Unix setup script
├── start.bat                # Windows start script
├── start.sh                 # Unix start script
└── README.md
```

## Libraries Used

### Backend (Python)
- **FastAPI** - Modern, fast web framework for building APIs
- **uvicorn** - ASGI server for FastAPI
- **pydantic** - Data validation using Python type hints
- **requests** - HTTP library for fetching web pages
- **beautifulsoup4** - HTML/XML parsing
- **lxml** - Fast XML and HTML parser
- **nltk** - Natural Language Toolkit for NLP
- **scikit-learn** - Machine learning library (TF-IDF)
- **numpy** - Numerical computing

### Frontend (TypeScript/JavaScript)
- **React** - UI library
- **React Three Fiber** - React renderer for Three.js
- **@react-three/drei** - Useful helpers for R3F
- **Three.js** - 3D graphics library
- **Axios** - HTTP client
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server

## How It Works

1. **Web Scraping**: The crawler fetches the article content, handling various website structures and rotating user agents for compatibility

2. **Text Extraction**: BeautifulSoup extracts the main content, removing scripts, styles, ads, and navigation elements

3. **NLP Processing**:
   - Text is tokenized and lemmatized using NLTK
   - Named entities are extracted using NLTK's NER
   - Keywords are scored using TF-IDF
   - Position weighting gives bonus to words in titles and first paragraphs

4. **Score Combination**: Multiple extraction methods are combined with weighted averaging

5. **3D Visualization**: React Three Fiber renders words on a Fibonacci sphere, with size and color based on relevance

## Troubleshooting

### Backend won't start
- Ensure Python 3.9+ is installed: `python --version`
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 8001 is available

### Frontend won't start
- Ensure Node.js 18+ is installed: `node --version`
- Ensure all dependencies are installed: `npm install`
- Check if port 3000 is available

### "Cannot connect to server" error
- Make sure the backend is running on port 8001
- Check for CORS issues in browser console

### Article not loading
- Some websites block automated requests; try a different URL
- JavaScript-heavy sites may not work well; try Wikipedia articles

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Created as a demonstration of integrating a 3D visualization frontend with an NLP backend pipeline.

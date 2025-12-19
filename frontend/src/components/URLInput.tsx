/**
 * URL Input Component
 * Provides a form for entering article URLs with sample suggestions
 */

import React, { useState, FormEvent, useCallback } from 'react';

interface URLInputProps {
  onSubmit: (url: string) => void;
  isLoading: boolean;
  error: string | null;
}

// Sample URLs for quick testing
const SAMPLE_URLS = [
  { url: 'https://en.wikipedia.org/wiki/Artificial_intelligence', label: 'AI (Wikipedia)' },
  { url: 'https://en.wikipedia.org/wiki/Climate_change', label: 'Climate Change' },
  { url: 'https://www.bbc.com/news', label: 'BBC News' },
  { url: 'https://www.reuters.com/', label: 'Reuters' },
];

export const URLInput: React.FC<URLInputProps> = ({ onSubmit, isLoading, error }) => {
  const [url, setUrl] = useState<string>('');
  const [showSamples, setShowSamples] = useState<boolean>(false);

  const handleSubmit = useCallback(
    (e: FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (url.trim() && !isLoading) {
        onSubmit(url.trim());
      }
    },
    [url, isLoading, onSubmit]
  );

  const handleClear = useCallback(() => {
    setUrl('');
  }, []);

  const handleSampleClick = useCallback((sampleUrl: string) => {
    setUrl(sampleUrl);
    setShowSamples(false);
    onSubmit(sampleUrl);
  }, [onSubmit]);

  return (
    <div className="input-container">
      <form className="input-form" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <input
            type="text"
            className="url-input"
            placeholder="Enter any article or news URL..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={isLoading}
            aria-label="Article URL"
            onFocus={() => setShowSamples(true)}
            onBlur={() => setTimeout(() => setShowSamples(false), 200)}
          />
          {url && (
            <button
              type="button"
              className="clear-btn"
              onClick={handleClear}
              aria-label="Clear input"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          )}
        </div>

        <button
          type="submit"
          className="submit-btn"
          disabled={isLoading || !url.trim()}
        >
          {isLoading ? (
            <>
              <span className="spinner-small" />
              Analyzing...
            </>
          ) : (
            <>
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              Generate Word Cloud
            </>
          )}
        </button>

        {error && (
          <div className="error-message">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            {error}
          </div>
        )}

        {/* Sample URLs dropdown */}
        {showSamples && !url && (
          <div className="samples-dropdown">
            <span className="samples-label">Try a sample:</span>
            {SAMPLE_URLS.map((sample, index) => (
              <button
                key={index}
                type="button"
                className="sample-btn"
                onClick={() => handleSampleClick(sample.url)}
              >
                {sample.label}
              </button>
            ))}
          </div>
        )}
      </form>

      <div className="input-hint-text">
        <span>Supports news articles, Wikipedia, blogs, and most web pages worldwide</span>
      </div>
    </div>
  );
};

export default URLInput;

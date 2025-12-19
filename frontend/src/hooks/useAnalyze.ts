/**
 * Custom hook for article analysis API calls
 */

import { useState, useCallback } from 'react';
import axios, { AxiosError } from 'axios';
import type { WordData, AnalyzeResponse, UseAnalyzeReturn, ApiError } from '../types';

// API base URL
const API_BASE_URL = 'http://localhost:8001';

/**
 * Hook for analyzing articles and getting word cloud data
 */
export function useAnalyze(): UseAnalyzeReturn {
  const [words, setWords] = useState<WordData[]>([]);
  const [title, setTitle] = useState<string>('');
  const [source, setSource] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<UseAnalyzeReturn['stats']>(null);

  /**
   * Analyze an article URL
   */
  const analyze = useCallback(async (url: string): Promise<void> => {
    // Validate URL
    if (!url || !url.trim()) {
      setError('Please enter a URL');
      return;
    }

    // Basic URL validation
    try {
      new URL(url);
    } catch {
      setError('Please enter a valid URL starting with http:// or https://');
      return;
    }

    setIsLoading(true);
    setError(null);
    setWords([]);
    setTitle('');
    setSource('');
    setStats(null);

    try {
      const response = await axios.post<AnalyzeResponse>(
        `${API_BASE_URL}/analyze`,
        { url },
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 90000, // 90 second timeout for slow websites
        }
      );

      const data = response.data;

      if (!data.words || data.words.length === 0) {
        throw new Error('No keywords extracted from the article');
      }

      setWords(data.words);
      setTitle(data.title || 'Unknown Article');
      setSource(data.source || 'Unknown Source');
      setStats({
        wordCount: data.word_count,
        articleLength: data.article_length,
        quotesFound: data.quotes_found,
        statisticsFound: data.statistics_found,
        keyNumbersFound: data.key_numbers_found,
      });

    } catch (err) {
      console.error('Analysis error:', err);

      if (axios.isAxiosError(err)) {
        const axiosError = err as AxiosError<ApiError>;

        if (axiosError.response) {
          const detail = axiosError.response.data?.detail;
          setError(detail || `Server error: ${axiosError.response.status}`);
        } else if (axiosError.request) {
          setError('Could not connect to the server. Please ensure the backend is running on port 8001.');
        } else {
          setError('Failed to send request. Please try again.');
        }
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Reset all state
   */
  const reset = useCallback((): void => {
    setWords([]);
    setTitle('');
    setSource('');
    setIsLoading(false);
    setError(null);
    setStats(null);
  }, []);

  return {
    analyze,
    words,
    title,
    source,
    isLoading,
    error,
    reset,
    stats,
  };
}

export default useAnalyze;

/**
 * Type definitions for the 3D Word Cloud application
 */

// Word data from the API
export interface WordData {
  word: string;
  weight: number;
  type?: 'keyword' | 'number';
  number_type?: 'percent' | 'money' | 'large_number' | 'year' | 'multiplier' | 'rank' | 'stat';
  sentence?: string;
  context?: string;
  entity_type?: string;
}

// API response structure
export interface AnalyzeResponse {
  words: WordData[];
  title: string;
  source: string;
  word_count: number;
  article_length: number;
  quotes_found: number;
  statistics_found: number;
  key_numbers_found: number;
}

// API request structure
export interface AnalyzeRequest {
  url: string;
}

// Error response from API
export interface ApiError {
  detail: string;
}

// Sample URL structure
export interface SampleUrl {
  url: string;
  description: string;
}

// 3D Word position in space
export interface Word3DPosition {
  x: number;
  y: number;
  z: number;
}

// Word with 3D positioning data
export interface Word3DData extends WordData {
  position: Word3DPosition;
  color: string;
  fontSize: number;
}

// Hook return type for useAnalyze
export interface UseAnalyzeReturn {
  analyze: (url: string) => Promise<void>;
  words: WordData[];
  title: string;
  source: string;
  isLoading: boolean;
  error: string | null;
  reset: () => void;
  stats: {
    wordCount: number;
    articleLength: number;
    quotesFound: number;
    statisticsFound: number;
    keyNumbersFound: number;
  } | null;
}

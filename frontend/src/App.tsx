/**
 * Main Application Component
 * 3D Word Cloud Visualization App
 */

import React from 'react';
import { WordCloud3D } from './components/WordCloud3D';
import { URLInput } from './components/URLInput';
import { LoadingSpinner } from './components/LoadingSpinner';
import { useAnalyze } from './hooks/useAnalyze';
import './index.css';

const App: React.FC = () => {
  const { analyze, words, title, source, isLoading, error, stats } = useAnalyze();

  return (
    <div className="app-container">
      {/* Header with title and input */}
      <header className="header">
        <div className="header-content">
          <div className="title-section">
            <h1>3D Word Cloud</h1>
            <p>Visualize article topics in 3D space</p>
          </div>

          <URLInput
            onSubmit={analyze}
            isLoading={isLoading}
            error={error}
          />
        </div>
      </header>

      {/* 3D Visualization */}
      <WordCloud3D words={words} />

      {/* Loading Overlay */}
      {isLoading && (
        <LoadingSpinner
          message="Analyzing article..."
          subMessage="Extracting keywords and important figures"
        />
      )}

      {/* Info Panel - Shows when words are loaded */}
      {words.length > 0 && !isLoading && (
        <div className="info-panel">
          <h2 className="info-title">{title}</h2>
          <p className="info-source">{source}</p>
          <div className="info-stats">
            <div className="stat">
              <span className="stat-value">{words.length}</span>
              <span className="stat-label">Keywords</span>
            </div>
            <div className="stat">
              <span className="stat-value">
                {(Math.max(...words.map(w => w.weight)) * 100).toFixed(0)}%
              </span>
              <span className="stat-label">Top Relevance</span>
            </div>
            {stats && (
              <div className="stat">
                <span className="stat-value">{stats.keyNumbersFound}</span>
                <span className="stat-label">Numbers</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Instructions Panel */}
      <div className="instructions">
        <h3>Controls</h3>
        <ul>
          <li>Drag to rotate the cloud</li>
          <li>Scroll to zoom in/out</li>
          <li>Hover words for details</li>
          <li>Click words to log info</li>
        </ul>
      </div>

      {/* Legend */}
      {words.length > 0 && !isLoading && (
        <div className="legend">
          <h3>Relevance Levels</h3>
          <div className="legend-items">
            <div className="legend-item">
              <span className="legend-color-dot" style={{ background: '#2ECC71' }}></span>
              <span>Very High (80-100%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ background: '#FF9F43' }}></span>
              <span>High (60-79%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ background: '#E91E63' }}></span>
              <span>Medium (40-59%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ background: '#1E90FF' }}></span>
              <span>Low (20-39%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ background: '#00D4FF' }}></span>
              <span>Very Low (0-19%)</span>
            </div>
            <div className="legend-divider"></div>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ background: '#FFD700' }}></span>
              <span>Numbers & Statistics</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;

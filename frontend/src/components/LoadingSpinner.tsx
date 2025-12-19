/**
 * Loading Spinner Component
 * Displays an animated loading indicator during API calls
 */

import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
  subMessage?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Analyzing article...',
  subMessage = 'Extracting keywords and topics',
}) => {
  return (
    <div className="loading-container">
      <div className="spinner-wrapper">
        <div className="spinner" />
        <div className="spinner-glow" />
      </div>
      <div className="loading-text">
        {message}
        {subMessage && <span>{subMessage}</span>}
      </div>
    </div>
  );
};

export default LoadingSpinner;

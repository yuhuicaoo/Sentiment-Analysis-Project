// import React, { useEffect, useRef, useCallback, useState } from "react";
// import axios from "axios";

function SentimentDisplay({ setPrediction, setIsLoading }) {
  return (
    <section className="prediction">
      <div className="container">
        <div className="row">
          <div className="content-wrapper">
            <div className="button-wrapper">
              <button
                className="btn"
              >
                Clear
              </button>
              <button
                className="btn"
              >
                Predict
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default SentimentDisplay;
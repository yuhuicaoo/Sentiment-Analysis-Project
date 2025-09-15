import React, { useEffect, useRef, useCallback, useState } from "react";
import axios from "axios";
import happy_face from "../assets/happy_face.svg";
import neutral_face from "../assets/neutral_face.svg";
import sad_face from "../assets/sad_face.svg";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSpinner } from "@fortawesome/free-solid-svg-icons";

function SentimentDisplay({
  prediction,
  isLoading,
  setPrediction,
  setIsLoading,
}) {
  const apiUrl = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
  const [text, setText] = useState("");

  const sentimentImgs = {
    Positive: happy_face,
    Negative: sad_face,
    Neutral: neutral_face,
  };

  const predict = async () => {
    if (!text) return;
    setIsLoading(true);

    try {
      const response = await axios.post(`${apiUrl}`, { text: text });
      setPrediction(response.data);
      console.log(response.data);
    } catch (error) {
      console.error(
        "Prediction failed:",
        error.response?.data || error.message
      );
    }

    setIsLoading(false);
  };

  return (
    <section className="prediction">
      <div className="container">
        <div className="row">
          <div className="content-wrapper">
            <figure className="img-wrapper">
              {isLoading ? (
                <>
                  <div className="skeleton__display">
                    <div className="skeleton__display__spinner">
                      <FontAwesomeIcon
                        icon={faSpinner}
                        spin
                        className="ml-2 display__spinner"
                      />
                    </div>
                  </div>
                </>
              ) : (
                <img
                  src={sentimentImgs[prediction?.sentiment || "Neutral"]}
                  alt=""
                  className="predicted-sentiment-img"
                />
              )}
            </figure>
            <div className="text-input-wrapper">
              <textarea
                type="text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter text here."
                className="text-input"
              />
            </div>
            <div className="button-wrapper">
              <button
                onClick={() => {
                  setIsLoading(true)
                  setText("");
                  setPrediction(null);
                  setTimeout(() => setIsLoading(false), 50);
                }}
                className="btn"
              >
                Clear
              </button>
              <button onClick={predict} className="btn">
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

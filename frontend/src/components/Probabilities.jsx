import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSpinner } from "@fortawesome/free-solid-svg-icons";

function Prediction({ prediction, isLoading }) {
  const { sentiment = "-", probabilities = new Array(3).fill("-") } =
    prediction ?? {};

  const sentimentMap = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
  };

  return (
    <div className="container">
      <div className="row">
        <div className="content-wrapper">
          <div className="prediction-wrapper">
            <h2 className="prediction-header purple">
                Prediction : {""}
              {isLoading ? (
                <FontAwesomeIcon
                  icon={faSpinner}
                  spin
                  className="ml-2"
                  size="xs"
                />
              ) : (
                <span>{sentiment}</span>
              )}
            </h2>
          </div>
          <div className="probabilities">
            <h3 className="probabilities__header purple">Probabilities</h3>
            {isLoading ? (
              <>
                <div className="skeleton skeleton__table">
                  <div className="skeleton__table__spinner">
                    <FontAwesomeIcon
                      icon={faSpinner}
                      spin
                      className="ml-2"
                      size="2x"
                    />
                  </div>
                </div>
              </>
            ) : (
              <div className="probabilities__table">
                {probabilities.map((prob, index) => (
                  <div className="value__container" key={index}>
                    <div className="prob">{sentimentMap[index]}: {prob} %</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Prediction;

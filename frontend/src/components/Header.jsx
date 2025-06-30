import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGithub } from "@fortawesome/free-brands-svg-icons";

function Header() {
  return (
    <section className="landing">
      <header>
        <div className="container">
          <div className="row">
            <div className="content-wrapper">
              <div className="header-wrapper">
                <h1 className="header-title purple">Sentiment Analysis Classifier</h1>
                <a
                  href="https://github.com/yuhuicaoo/Sentiment-Analysis-Project"
                  target="_blank"
                  rel="noreferrer"
                >
                  <FontAwesomeIcon
                    icon={faGithub}
                    size="2x"
                    className="github-icon"
                  />
                </a>
              </div>
              <h3 className="header-description purple">
                Write a sentence for the model to predict its sentiment
              </h3>
            </div>
          </div>
        </div>
      </header>
    </section>
  );
}

export default Header;
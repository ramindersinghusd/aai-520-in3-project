# 🧭 Multi-Agent Investment Research Report  
**University of San Diego – AAI-520-IN3**  
**Course:** Natural Language Processing and GenAI  
**Instructor:** Premkumar Chithaluru, Ph.D  
**Authors:** Richa Arun Kumar Jha, Raminder Singh, Samiksha Kodgire  
**Date:** October 16, 2025  

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Phi--3%20Mini-brightgreen?logo=microsoft&logoColor=white" />
  <img src="https://img.shields.io/badge/Sentiment-DistilBERT-orange?logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-lightgrey" />
</p>

---

## 🧠 Overview

This project presents a **multi-agent, AI-driven investment research system** that unites:

- **Quantitative market analytics**
- **News sentiment classification**
- **Generative financial summarization**

It produces comprehensive research reports, visualizations, and portfolio insights for six major technology tickers:
> **AAPL**, **GOOG**, **TSLA**, **AMZN**, **NVDA**, **MSFT**

---

## ⚙️ System Architecture

flowchart TD
    A["📈 DataAgent"] --> B["⚙️ TechnicalAgent"]
    B --> C["📊 RiskAgent"]
    A --> D["📰 NewsAgent"]
    D --> E["🧠 SentimentAgent"]
    E --> F["🧭 RoutingAgent"]
    C --> G["💼 PortfolioAgent"]
    E --> H["📈 EvaluationAgent"]
    H --> I["🪶 LLMOptimizerAgent"]
    G --> I
    I --> J["🧾 Final Report"]


## Agent Functions

| Agent                 | Function                                          |
| --------------------- | ------------------------------------------------- |
| **DataAgent**         | Fetches market data from Yahoo Finance            |
| **TechnicalAgent**    | Computes SMA20, SMA50, RSI, and momentum          |
| **RiskAgent**         | Analyzes volatility, drawdown, and composite risk |
| **NewsAgent**         | Collects financial headlines via NewsAPI          |
| **SentimentAgent**    | Classifies sentiment using DistilBERT             |
| **RoutingAgent**      | Routes headlines into earnings/macro/product news |
| **PortfolioAgent**    | Suggests risk-aligned portfolio allocation        |
| **EvaluationAgent**   | Compares sentiment trends vs. price action        |
| **LLMOptimizerAgent** | Generates summaries with Phi-3 Mini               |

## Technologies Used

| Category         | Tools / Frameworks                         |
| ---------------- | ------------------------------------------ |
| **Programming**  | Python 3.12                                |
| **Data Sources** | Yahoo Finance, NewsAPI                     |
| **ML Models**    | DistilBERT, Phi-3 Mini                     |
| **Libraries**    | PyTorch, Pandas, NumPy, Matplotlib, Plotly |
| **Export Tools** | nbconvert, WeasyPrint                      |
| **Platform**     | Google Colab / CUDA GPU                    |

## 🚀 Execution Workflow
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Report
tickers = ["AAPL","GOOG","TSLA","AMZN","NVDA","MSFT"]
combined, radar = run_report(tickers)

3️⃣ Export to HTML
!jupyter nbconvert --to html --output FinalProject_v13_clean.html FinalProject_v13_clean.ipynb

4️⃣ Export to PDF (Optional)
!weasyprint FinalProject_v13_clean.html FinalProject_v13_clean.pdf

📊 Output
## Generated Files

| File                           | Description                            |
| ------------------------------ | -------------------------------------- |
| `FinalProject_v13_clean.ipynb` | Main notebook containing all agents    |
| `FinalProject_v13_clean.html`  | Interactive report with visualizations |
| `FinalProject_v13_clean.pdf`   | Printable, publication-style version   |
| `README.md`                    | GitHub documentation                   |
| `requirements.txt`             | Environment dependencies               |

## 📈 Visualizations

Price & technical indicator charts
Risk and volatility plots
Sentiment vs. price alignment graphs
Correlation radar comparing all tickers

## 🧩 Example Insights

-NVDA and MSFT show aligned sentiment and price movement, indicating sustained bullish trends.
-TSLA exhibits divergent sentiment patterns, implying volatility or speculative pressure.
-Phi-3 Mini summaries deliver coherent, domain-aware investment insights.

## 📚 References

-Hugging Face Transformers
-Phi-3 Mini (Microsoft)
-Yahoo Finance API
-NewsAPI
-nbconvert Documentation

## 🧭 Future Enhancements

-Deploy as a Streamlit or Flask web app
-Integrate reinforcement learning for dynamic portfolio rebalancing
-Extend sentiment model with financial-tuned LLMs
-Add real-time market alerting system

## 🪪 License

This project is released under the Apache License 2.0.
© 2025 University of San Diego — All Rights Reserved.

# aai-520-in3-project

## Project > Stock Sentiment Analyzer Agent
Analyze the below problem statement and create step-by-step roadmap to solve this with Gemini AI. For each step, identify right steps for approach, list it. Also, pick best light-weight model to use.
specially, focus on #6 workflow i.e. implementing evaluator–optimizer. For full project, provide python code.

## Abstract
Stock Sentiment Analyzer Agent : 

Abstract: The Stock Sentiment Analyzer Agent is designed to evaluate how financial news sentiment influences short-term stock price movements. In modern markets, investor psychology and news-driven reactions often dominate price swings, making sentiment analysis
a critical investment research tool. This project integrates Yahoo Finance price data with news articles from sources such as NewsAPI and Kaggle datasets. The agent autonomously plans its research by fetching relevant news for a chosen stock symbol, processes the text using Natural
Language Processing (NLP) techniques, and classifies sentiment as positive, negative, or neutral. 

By aligning sentiment scores with actual stock price movements, the agent self-reflects on the accuracy of its predictions and iteratively improves through feedback loops. To enhance performance, a routing mechanism distinguishes between earnings-related and
macroeconomic-related news, assigning each to specialized sub-agents. The evaluator–optimizer framework compares sentiment predictions against realized price changes to refine classification thresholds. 

This project demonstrates the practical utility of agentic AI in finance by showing how multiple autonomous components can collaborate to provide actionable insights for traders and analysts.

## Workflow
1. Fetch news → Preprocess (cleaning, tokenization).
2. Apply the sentiment analysis model.
3. Summarize sentiment trends.
4. Route: earnings news → earnings agent; macro news → macro agent.
5. Compare predicted sentiment vs. stock price moves.
6. Refine sentiment thresholds using evaluator–optimizer

### Team members
- Richa Jha [rjha@sandiego.edu]
- Raminder Singh [ramindersingh@sandiego.edu]
- Samiksha Kkodgire [skodgire@sandiego.edu]

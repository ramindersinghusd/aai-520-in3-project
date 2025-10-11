"""
Stock Sentiment Analyzer Agent System
Multi-Agent Architecture using LangChain and LangGraph with Gemini AI
"""

"""
build tool: uv
install UV (on windows): 
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

> uv --version
uv 0.8.22 (ade2bdbd2 2025-09-23)

# Giving the app name is not mandatory
> uv init <app_name>

Setting up a project:
# Initializing a project called "app"
# This will create:
# 1. main.py file with basic print statement
# 2. .python-version file
# 3. pyproject.toml containes the project metadata
# 4. Readme.md. file

> cd <app_name>

Install libraries:
> uv add langchain langchain-google-genai langgraph chromadb transformers yfinance newsapi-python langchain_community torch sentence-transformers  langgraph-checkpoint-sqlite feedparser python-dotenv datasets huggingface_hub[hf_xet] hf_xet  langchain-chroma langchain-huggingface aiosqlite langchain-perplexity

> uv pip install langgraph_checkpoint_sqlite
> uv pip install aiosqlite

#update the code file - stock_sentiment_analyzer.py

#Run the code
> uv run stock_sentiment_analyzer.py

"""

import os
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import json
import logging

# LangChain and LangGraph imports
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.prompts import PromptTemplate

from langchain_perplexity import ChatPerplexity

#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# LangGraph imports
from langgraph.graph import StateGraph, END, START
#from langgraph.prebuilt import ToolExecutor
#from langgraph.checkpoint.sqlite import SqliteSaver

# News API imports
import requests
from newsapi import NewsApiClient
import feedparser

# Sentiment Analysis
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

#from langgraph.checkpoint.sqlite import AsyncSqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import MemorySaver

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    #GOOGLE_API_KEY > from raminder's account
    #TODO: Hardcoded as API key is not picking from .env #os.getenv("GOOGLE_API_KEY")
    GOOGLE_API_KEY: str = "" 
    
    #NEWS_API_KEYy > from raminder's account
    NEWS_API_KEY: str = "" #os.getenv("NEWS_API_KEY")
    
    #FINANCIAL_NEWS_API_KEY: str = os.getenv("FINANCIAL_NEWS_API_KEY")
    
    MODEL_NAME: str = "gemini-2.0-flash-exp"
    
    SENTIMENT_MODEL_1: str = "ProsusAI/finbert" #Params: 110MB #pytorch_model.bin ~438MB | model.safetensors ~ 438MB
    SENTIMENT_MODEL_2 = "distilbert-base-uncased-finetuned-sst-2-english"
    SENTIMENT_MODEL_3 = "AdityaAI9/distilbert_finance_sentiment_analysis"
    SENTIMENT_MODEL_4 = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" # > #Params: 82M
    
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    MAX_ARTICLES: int = 10 #50    #total articles to fetch from a single/each sources 
    EVALUATION_THRESHOLD: float = 0.7
    
    #Perplexity API Key > from raminder's account
    PPLX_API_KEY: str = "" #os.getenv("NEWS_API_KEY")
    # os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
    # os.environ["LANGSMITH_TRACING"] = "true"

# State definition for the multi-agent system
class AgentState(TypedDict):
    stock_symbol: str
    news_articles: List[Dict]
    processed_articles: List[Dict]
    sentiment_scores: List[Dict]
    routing_decisions: Dict
    earnings_analysis: Dict
    macro_analysis: Dict
    evaluation_metrics: Dict
    optimization_feedback: Dict
    final_report: Dict
    iteration_count: int
    max_iterations: int

# Initialize models and tools
class ModelManager:
    def __init__(self, config: Config):
        self.config = config
        print(f">>>> Using LLM Model: {config.MODEL_NAME} with Google API Key: {config.GOOGLE_API_KEY}")
        
        #self.llm = ChatGoogleGenerativeAI(
        #    model=config.MODEL_NAME,
        #    google_api_key=config.GOOGLE_API_KEY,
        #    temperature=0.1
        #)
        
        #used Perplexity-Pro as Gemini is not responding/working 
        self.llm = ChatPerplexity(
            model="sonar",   #sonar-pro
            pplx_api_key=config.PPLX_API_KEY,
            temperature=0.3  #0.6
        ) 

        # Initialize sentiment analysis model (lightweight DistilBERT)
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model=config.SENTIMENT_MODEL_1,     
            tokenizer=config.SENTIMENT_MODEL_1
        )

        # Initialize embeddings for vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="financial_news"
        )

    def get_llm(self):
        return self.llm

class NewsDataCollector:
    def __init__(self, config: Config):
        self.config = config
        self.news_client = NewsApiClient(api_key=config.NEWS_API_KEY) if config.NEWS_API_KEY else None

    async def fetch_yahoo_finance_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Yahoo Finance"""
        logger.info(f">>>> fetch_yahoo_finance_news()...")
        
        # Fetch historical data for the last year
        #historical_data = ticker.history(period="1y")
        
        # Fetch financial statements
        #financials = ticker.financials
        
        # Fetch stock actions (e.g., dividends, splits)
        #actions = ticker.actions
        
        # Download data for Amazon, Apple, and Google
        #data = yf.download("AMZN AAPL GOOG", start="2022-01-01", end="2022-12-31")
        
        #To group the data by ticker instead of Open/High/Low/Close columns:
        #data_grouped = yf.download("AMZN AAPL GOOG", start="2022-01-01", end="2022-12-31", group_by='tickers')
        
        try:
            # Create a Ticker object for the stock symbol
            ticker = yf.Ticker(symbol)
            news = ticker.news
            logger.info(f">>>> yf.ticker.news {len(news)}:: {news[0]}")

            articles = []
            for item in news[:self.config.MAX_ARTICLES]:  
                articles.append({
                    'title': item['content']['title'],
                    'content': item['content']['summary'],
                    'url': item['content']['canonicalUrl']['url'],
                    
                    #TODO: Fix error related to date conversion > Error fetching Yahoo Finance news: 'str' object cannot be interpreted as an integer
                    #'published_at': datetime.fromtimestamp(item['content']['pubDate']).replace('Z', '+00:00'), 
                    'published_at': item['content']['pubDate'],
                    
                    'source': 'yahoo_finance',
                    'symbol': symbol
                })
            logger.info(f">>>> yf.Ticker articles {len(articles)}:: {articles}") 
               
            return articles
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news: {e}")
            return []

    async def fetch_news_api(self, symbol: str) -> List[Dict]:
        """Fetch news from NewsAPI"""
        
        print(f"\n>>>> fetch_news_api() get {self.config.MAX_ARTICLES} articles for {symbol} in last 30 days...\n")
        
        if not self.news_client:
            return []

        try:
            # Search for company name and stock symbol
            query = f"{symbol} stock earnings financial"
            articles_data = self.news_client.get_everything(
                q=query,
                language='en',
                sort_by= 'relevancy', #'publishedAt',
                page_size=self.config.MAX_ARTICLES, 
                from_param=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            )

            articles = []
            for item in articles_data.get('articles', []):
                articles.append({
                    'title': item.get('title', ''),
                    'content': item.get('content', ''),
                    'url': item.get('url', ''),
                    'published_at': datetime.fromisoformat(item.get('publishedAt', '').replace('Z', '+00:00')),
                    'source': 'news_api',
                    'symbol': symbol
                })
            print(f">>>> fetch_news_api() Downloaded > #Articles :: {len(articles)}") 
            #print(f">>>> Downloaded articles :: {articles}")    
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching NewsAPI articles: {e}")
            return []

# Agent Classes
class ControllerAgent:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.llm = model_manager.get_llm()

    async def orchestrate(self, state: AgentState) -> AgentState:
        """Main orchestration logic"""
        logger.info(f"Controller: Starting analysis for {state['stock_symbol']}")

        # Initialize state
        state['iteration_count'] = state.get('iteration_count', 0)
        state['max_iterations'] = state.get('max_iterations', 3)
        state['final_report'] = {}

        return state

class ResearchAgent:
    def __init__(self, model_manager: ModelManager, news_collector: NewsDataCollector):
        self.model_manager = model_manager
        self.news_collector = news_collector
        self.llm = model_manager.get_llm()
     
    @staticmethod   
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types"""
        import numpy as np
         
        if isinstance(obj, dict):
            return {k: ResearchAgent.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ResearchAgent.convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to Python lists
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj


    async def fetch_and_store_news(self, state: AgentState) -> AgentState:
        """Fetch news and store in vector database"""
        logger.info(f"\n >>>>Research Agent: Fetching & storing news for {state['stock_symbol']}")
        
        #create_stock_sentiment_graph
        symbol = state['stock_symbol']
        logger.info(f"Research Agent: Fetching news for {symbol}")

        # Fetch from multiple sources
        yahoo_articles = await self.news_collector.fetch_yahoo_finance_news(symbol)
        news_api_articles = await self.news_collector.fetch_news_api(symbol)

        logger.info(f">>>> # yahoo_articles >> {len(yahoo_articles)}")
        logger.info(f">>>> # news_api_articles >> {len(news_api_articles)}")
        # Combine and deduplicate articles
        all_articles = yahoo_articles + news_api_articles

        # Store articles in vector database for retrieval
        documents = []
        for article in all_articles:
            if article['content']:
                doc = Document(
                    page_content=f"Title: {article['title']}\nContent: {article['content']}",
                    metadata={
                        'source': article['source'],
                        'symbol': symbol,
                        'url': article['url'],
                        'published_at': str(article['published_at'])
                    }
                )
                #print("\n" + "="*80)
                #print(f">>>> doc >> {doc}")
                clean_doc = self.convert_numpy_types(doc)
                #print(f">>>> clean_doc >> {clean_doc}")
                #print("="*80)
                documents.append(clean_doc)

        
        
        #TODO: Fix below error - AttributeError: 'Chroma' object has no attribute 'add_documents'
        #TODO: convert_numpy_types(raw_data)
         
        if documents:
            try:
                # Add documents directly to vector store after conversion
                self.model_manager.vector_store.add_documents(documents)
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")

        state['news_articles'] = all_articles
        logger.info(f"Research Agent: Collected {len(all_articles)} articles")
       
        return state

class PreprocessingAgent:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    async def preprocess_articles(self, state: AgentState) -> AgentState:
        """Clean and preprocess articles"""
        logger.info("\n >>>> Preprocessing Agent: Cleaning and tokenizing articles")

        processed_articles = []
        for article in state['news_articles']:
            # Clean text
            cleaned_content = self._clean_text(article['content'])

            # Split into chunks if needed
            chunks = self.text_splitter.split_text(cleaned_content)

            for i, chunk in enumerate(chunks):
                processed_articles.append({
                    'original_article': article,
                    'processed_content': chunk,
                    'chunk_id': i,
                    'word_count': len(chunk.split())
                })

        state['processed_articles'] = processed_articles
        logger.info(f"Preprocessing Agent: Processed {len(processed_articles)} article chunks")

        return state

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        import re

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep financial terms
        text = re.sub(r'[^a-zA-Z0-9\s\$\%\.\,\-]', '', text)

        return text.strip()

class SentimentAnalysisAgent:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.sentiment_model = model_manager.sentiment_model
        self.llm = model_manager.get_llm()

    async def analyze_sentiment(self, state: AgentState) -> AgentState:
        """Perform sentiment analysis on processed articles"""
        logger.info("Sentiment Analysis Agent: Analyzing sentiment")

        sentiment_scores = []

        for article in state['processed_articles']:
            content = article['processed_content']

            # Use DistilBERT for fast sentiment analysis
            try:
                sentiment_result = self.sentiment_model(content)[0]

                # Normalize scores
                score = sentiment_result['score']
                label = sentiment_result['label'].lower()

                # Convert to standardized format
                if label in ['positive', 'pos']:
                    normalized_score = score
                    sentiment_label = 'positive'
                elif label in ['negative', 'neg']:
                    normalized_score = -score
                    sentiment_label = 'negative'
                else:
                    normalized_score = 0
                    sentiment_label = 'neutral'

                sentiment_scores.append({
                    'article_id': article.get('chunk_id', 0),
                    'content': content[:100] + "...",
                    'sentiment_label': sentiment_label,
                    'confidence_score': score,
                    'normalized_score': normalized_score,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                sentiment_scores.append({
                    'article_id': article.get('chunk_id', 0),
                    'content': content[:100] + "...",
                    'sentiment_label': 'neutral',
                    'confidence_score': 0.5,
                    'normalized_score': 0,
                    'error': str(e)
                })

        state['sentiment_scores'] = sentiment_scores
        logger.info(f"Sentiment Analysis Agent: Analyzed {len(sentiment_scores)} content pieces")

        return state

class RouterAgent:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.llm = model_manager.get_llm()

    async def route_news(self, state: AgentState) -> AgentState:
        """Route news to appropriate specialist agents"""
        logger.info("Router Agent: Classifying news for routing")

        routing_decisions = {
            'earnings_articles': [],
            'macro_articles': [],
            'other_articles': []
        }

        # Create prompt for classification
        classification_prompt = PromptTemplate(
            template="""
            Classify the following financial news article into one of these categories:
            1. EARNINGS - Related to company earnings, quarterly results, financial statements
            2. MACRO - Related to macroeconomic factors, market trends, industry analysis
            3. OTHER - General news not specifically earnings or macro-focused

            Article Title: {title}
            Article Content: {content}

            Respond with only: EARNINGS, MACRO, or OTHER
            """,
            input_variables=["title", "content"]
        )

        for article in state['processed_articles']:
            original = article['original_article']

            # Use LLM to classify
            try:
                prompt = classification_prompt.format(
                    title=original['title'],
                    content=article['processed_content'][:500]
                )

                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                classification = response.content.strip().upper()

                if classification == 'EARNINGS':
                    routing_decisions['earnings_articles'].append(article)
                elif classification == 'MACRO':
                    routing_decisions['macro_articles'].append(article)
                else:
                    routing_decisions['other_articles'].append(article)

            except Exception as e:
                logger.error(f"Error in routing classification: {e}")
                routing_decisions['other_articles'].append(article)

        state['routing_decisions'] = routing_decisions
        logger.info(f"Router Agent: Routed {len(routing_decisions['earnings_articles'])} to earnings, "
                   f"{len(routing_decisions['macro_articles'])} to macro")

        return state

class EarningsSpecialistAgent:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.llm = model_manager.get_llm()

    async def analyze_earnings(self, state: AgentState) -> AgentState:
        """Specialized analysis for earnings-related news"""
        logger.info("Earnings Specialist Agent: Analyzing earnings news")
        print(f" >>>>> EarningsSpecialistAgent.analyze_earnings()...")

        earnings_articles = state['routing_decisions']['earnings_articles']

        if not earnings_articles:
            state['earnings_analysis'] = {'summary': 'No earnings-related articles found'}
            return state

        # Create earnings-specific analysis prompt
        earnings_prompt = PromptTemplate(
            template="""
            As a financial earnings specialist, analyze the following earnings-related news articles 
            for stock symbol {symbol}. Focus on:

            1. Revenue trends and growth
            2. Profit margins and profitability
            3. Forward guidance and projections
            4. Market expectations vs. actual results
            5. Key financial metrics and ratios

            Articles:
            {articles_text}

            Provide a comprehensive earnings analysis with:
            - Overall earnings sentiment (Positive/Negative/Neutral)
            - Key financial highlights
            - Risk factors identified
            - Investment implications

            Format as JSON with these keys: sentiment, highlights, risks, implications, confidence_score
            """,
            input_variables=["symbol", "articles_text"]
        )

        # Combine articles text
        articles_text = "\n\n".join([
            f"Title: {article['original_article']['title']}\n"
            f"Content: {article['processed_content']}"
            for article in earnings_articles[:10]  # Limit to prevent token overflow
        ])

        try:
            prompt = earnings_prompt.format(
                symbol=state['stock_symbol'],
                articles_text=articles_text
            )

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            # Try to parse as JSON, fallback to structured text
            try:
                analysis = json.loads(response.content)
            except:
                analysis = {
                    'sentiment': 'neutral',
                    'highlights': response.content,
                    'risks': 'See highlights',
                    'implications': 'See highlights',
                    'confidence_score': 0.5
                }

            state['earnings_analysis'] = analysis

        except Exception as e:
            logger.error(f"Error in earnings analysis: {e}")
            state['earnings_analysis'] = {
                'sentiment': 'neutral',
                'error': str(e),
                'confidence_score': 0.0
            }
        print(f" <<<<< EarningsSpecialistAgent.analyze_earnings()...")
        return state

class MacroSpecialistAgent:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.llm = model_manager.get_llm()

    async def analyze_macro(self, state: AgentState) -> AgentState:
        """Specialized analysis for macroeconomic news"""
        logger.info("Macro Specialist Agent: Analyzing macroeconomic news")

        macro_articles = state['routing_decisions']['macro_articles']

        if not macro_articles:
            state['macro_analysis'] = {'summary': 'No macro-related articles found'}
            return state

        # Create macro-specific analysis prompt
        macro_prompt = PromptTemplate(
            template="""
            As a macroeconomic specialist, analyze the following market and economic news 
            related to stock symbol {symbol}. Focus on:

            1. Market trends and sector performance
            2. Economic indicators impact
            3. Industry-wide developments
            4. Regulatory changes and policy impacts
            5. Competitive landscape shifts

            Articles:
            {articles_text}

            Provide a comprehensive macro analysis with:
            - Overall market sentiment impact (Positive/Negative/Neutral)
            - Key economic factors
            - Industry trends identified
            - Regulatory considerations
            - Market positioning implications

            Format as JSON with these keys: market_sentiment, economic_factors, industry_trends, regulatory_notes, positioning, confidence_score
            """,
            input_variables=["symbol", "articles_text"]
        )

        # Combine articles text
        articles_text = "\n\n".join([
            f"Title: {article['original_article']['title']}\n"
            f"Content: {article['processed_content']}"
            for article in macro_articles[:10]  # Limit to prevent token overflow
        ])

        try:
            prompt = macro_prompt.format(
                symbol=state['stock_symbol'],
                articles_text=articles_text
            )

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            # Try to parse as JSON, fallback to structured text
            try:
                analysis = json.loads(response.content)
            except:
                analysis = {
                    'market_sentiment': 'neutral',
                    'economic_factors': response.content,
                    'industry_trends': 'See economic factors',
                    'regulatory_notes': 'None identified',
                    'positioning': 'Neutral',
                    'confidence_score': 0.5
                }

            state['macro_analysis'] = analysis

        except Exception as e:
            logger.error(f"Error in macro analysis: {e}")
            state['macro_analysis'] = {
                'market_sentiment': 'neutral',
                'error': str(e),
                'confidence_score': 0.0
            }

        return state

class EvaluatorOptimizerAgent:
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
        self.llm = model_manager.get_llm()

    async def evaluate_and_optimize(self, state: AgentState) -> AgentState:
        """Evaluate predictions and optimize thresholds"""
        logger.info("Evaluator-Optimizer Agent: Evaluating and optimizing")

        # Fetch actual stock price data for comparison
        try:
            ticker = yf.Ticker(state['stock_symbol'])
            hist = ticker.history(period="5d")

            if not hist.empty:
                price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                actual_direction = 'positive' if price_change > 0.02 else 'negative' if price_change < -0.02 else 'neutral'
            else:
                actual_direction = 'neutral'
                price_change = 0

        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            actual_direction = 'neutral'
            price_change = 0

        # Calculate aggregate sentiment
        sentiment_scores = state.get('sentiment_scores', [])
        if sentiment_scores:
            avg_sentiment = sum([s['normalized_score'] for s in sentiment_scores]) / len(sentiment_scores)
            predicted_direction = 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
        else:
            avg_sentiment = 0
            predicted_direction = 'neutral'

        # Calculate accuracy
        accuracy = 1.0 if predicted_direction == actual_direction else 0.0

        # Generate optimization feedback
        optimization_feedback = {
            'actual_price_change': price_change,
            'actual_direction': actual_direction,
            'predicted_direction': predicted_direction,
            'average_sentiment': avg_sentiment,
            'accuracy': accuracy,
            'needs_optimization': accuracy < self.config.EVALUATION_THRESHOLD,
            'recommendations': []
        }

        # Add recommendations based on performance
        if accuracy < self.config.EVALUATION_THRESHOLD:
            optimization_feedback['recommendations'].extend([
                "Consider adjusting sentiment thresholds",
                "Evaluate news source quality and relevance",
                "Review specialist agent routing accuracy"
            ])

        # Evaluation metrics
        evaluation_metrics = {
            'sentiment_distribution': self._calculate_sentiment_distribution(sentiment_scores),
            'confidence_scores': [s.get('confidence_score', 0) for s in sentiment_scores],
            'article_coverage': {
                'total_articles': len(state.get('news_articles', [])),
                'earnings_articles': len(state.get('routing_decisions', {}).get('earnings_articles', [])),
                'macro_articles': len(state.get('routing_decisions', {}).get('macro_articles', [])),
            },
            'processing_time': datetime.now().isoformat()
        }

        state['evaluation_metrics'] = evaluation_metrics
        state['optimization_feedback'] = optimization_feedback

        # Determine if another iteration is needed
        state['iteration_count'] += 1

        logger.info(f"Evaluator-Optimizer: Accuracy={accuracy:.2f}, "
                   f"Iteration {state['iteration_count']}/{state['max_iterations']}")

        return state

    def _calculate_sentiment_distribution(self, sentiment_scores: List[Dict]) -> Dict:
        """Calculate distribution of sentiment labels"""
        if not sentiment_scores:
            return {'positive': 0, 'negative': 0, 'neutral': 0}

        distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        for score in sentiment_scores:
            label = score.get('sentiment_label', 'neutral')
            distribution[label] = distribution.get(label, 0) + 1

        total = len(sentiment_scores)
        return {k: v/total for k, v in distribution.items()}

class ReportGenerator:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.llm = model_manager.get_llm()

    async def generate_final_report(self, state: AgentState) -> AgentState:
        """Generate comprehensive final report"""
        logger.info("Report Generator: Creating final report")

        # Create comprehensive report prompt
        report_prompt = PromptTemplate(
            template="""
            Generate a comprehensive stock sentiment analysis report for {symbol} based on the following analysis:

            Sentiment Analysis Summary:
            {sentiment_summary}

            Earnings Analysis:
            {earnings_analysis}

            Macro Analysis:
            {macro_analysis}

            Evaluation Metrics:
            {evaluation_metrics}

            Create a professional report with:
            1. Executive Summary
            2. Overall Sentiment Assessment
            3. Key Findings from Earnings Analysis
            4. Macroeconomic Factors Impact
            5. Risk Assessment
            6. Investment Recommendation
            7. Confidence Level and Limitations

            Format as structured JSON with these sections.
            """,
            input_variables=["symbol", "sentiment_summary", "earnings_analysis", "macro_analysis", "evaluation_metrics"]
        )

        # Prepare summary data
        sentiment_scores = state.get('sentiment_scores', [])
        sentiment_summary = self._create_sentiment_summary(sentiment_scores)
        print(f">> generate_final_report-sentiment_summary : {sentiment_summary}")
        
        try:
            prompt = report_prompt.format(
                symbol=state['stock_symbol'],
                sentiment_summary=json.dumps(sentiment_summary),
                earnings_analysis=json.dumps(state.get('earnings_analysis', {})),
                macro_analysis=json.dumps(state.get('macro_analysis', {})),
                evaluation_metrics=json.dumps(state.get('evaluation_metrics', {}))
            )

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            print(f">>>> LLM response : {response}")

            try:
                final_report = json.loads(response.content)
                print(f">>>> generate_final_report - final_report : {final_report}")
            except:
                # Fallback structured report
                final_report = {
                    'executive_summary': response.content,
                    'overall_sentiment': sentiment_summary.get('overall_sentiment', 'neutral'),
                    'confidence_level': sentiment_summary.get('average_confidence', 0.5),
                    'timestamp': datetime.now().isoformat(),
                    'raw_response': response.content
                }

            state['final_report'] = final_report

        except Exception as e:
            logger.error(f"Error generating final report: {e}")
            print(f"2 - >>>> generate_final_report...{e}")
            state['final_report'] = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        print(f"<<<<< generate_final_report....")
        return state

    def _create_sentiment_summary(self, sentiment_scores: List[Dict]) -> Dict:
        """Create summary of sentiment analysis"""
        if not sentiment_scores:
            return {'overall_sentiment': 'neutral', 'average_confidence': 0.0, 'total_articles': 0}

        positive = sum(1 for s in sentiment_scores if s.get('sentiment_label') == 'positive')
        negative = sum(1 for s in sentiment_scores if s.get('sentiment_label') == 'negative')
        neutral = sum(1 for s in sentiment_scores if s.get('sentiment_label') == 'neutral')

        total = len(sentiment_scores)
        avg_confidence = sum(s.get('confidence_score', 0) for s in sentiment_scores) / total

        # Determine overall sentiment
        if positive > negative and positive > neutral:
            overall = 'positive'
        elif negative > positive and negative > neutral:
            overall = 'negative'
        else:
            overall = 'neutral'

        return {
            'overall_sentiment': overall,
            'distribution': {'positive': positive/total, 'negative': negative/total, 'neutral': neutral/total},
            'average_confidence': avg_confidence,
            'total_articles': total
        }

# Main workflow graph construction
def create_stock_sentiment_graph():
    """Create the main LangGraph workflow"""
    print(f"2 - >>>> create_stock_sentiment_graph()...")
    
    # Initialize configuration and managers
    config = Config()
    model_manager = ModelManager(config)
    news_collector = NewsDataCollector(config)

    # Initialize agents
    controller = ControllerAgent(model_manager)
    research_agent = ResearchAgent(model_manager, news_collector)
    preprocessing_agent = PreprocessingAgent(model_manager)
    sentiment_agent = SentimentAnalysisAgent(model_manager)
    router_agent = RouterAgent(model_manager)
    earnings_agent = EarningsSpecialistAgent(model_manager)
    macro_agent = MacroSpecialistAgent(model_manager)
    evaluator_agent = EvaluatorOptimizerAgent(model_manager, config)
    report_generator = ReportGenerator(model_manager)

    # Define workflow functions
    async def controller_node(state: AgentState) -> AgentState:
        return await controller.orchestrate(state)

    async def research_node(state: AgentState) -> AgentState:
        return await research_agent.fetch_and_store_news(state)

    async def preprocessing_node(state: AgentState) -> AgentState:
        return await preprocessing_agent.preprocess_articles(state)

    async def sentiment_node(state: AgentState) -> AgentState:
        return await sentiment_agent.analyze_sentiment(state)

    async def router_node(state: AgentState) -> AgentState:
        return await router_agent.route_news(state)

    async def earnings_node(state: AgentState) -> AgentState:
        return await earnings_agent.analyze_earnings(state)

    async def macro_node(state: AgentState) -> AgentState:
        return await macro_agent.analyze_macro(state)

    async def evaluator_node(state: AgentState) -> AgentState:
        return await evaluator_agent.evaluate_and_optimize(state)

    async def report_node(state: AgentState) -> AgentState:
        return await report_generator.generate_final_report(state)

    # Decision function for routing
    def should_continue_optimization(state: AgentState) -> str:
        """Decide whether to continue optimization or finish"""
        optimization_feedback = state.get('optimization_feedback', {})
        iteration_count = state.get('iteration_count', 0)
        max_iterations = state.get('max_iterations', 3)

        needs_optimization = optimization_feedback.get('needs_optimization', False)

        if needs_optimization and iteration_count < max_iterations:
            return "continue_optimization"
        else:
            return "finalize_report"

    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("controller", controller_node)
    workflow.add_node("research", research_node)
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("router", router_node)
    workflow.add_node("earnings", earnings_node)
    workflow.add_node("macro", macro_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("report", report_node)

    # Add edges - sequential workflow
    workflow.add_edge(START, "controller")
    workflow.add_edge("controller", "research")
    workflow.add_edge("research", "preprocessing")
    workflow.add_edge("preprocessing", "sentiment")
    workflow.add_edge("sentiment", "router")
    workflow.add_edge("router", "earnings")
    workflow.add_edge("earnings", "macro")
    workflow.add_edge("macro", "evaluator")

    # Conditional edge for optimization loop
    workflow.add_conditional_edges(
        "evaluator",
        should_continue_optimization,
        {
            "continue_optimization": "sentiment",  # Loop back for refinement
            "finalize_report": "report"
        }
    )

    workflow.add_edge("report", END)

    # Compile the graph
    
    # FIXED - Use correct checkpointer
    # OLD: memory = SqliteSaver.from_conn_string(":memory:")
    #checkpointer = AsyncSqliteSaver.from_conn_string(":memory:")
    #app = workflow.compile(checkpointer=checkpointer)
    
    #Use MemorySaver as temporary fix
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app

# Usage example and main execution
async def run_stock_sentiment_analysis(stock_symbol: str):
    """Run the complete stock sentiment analysis"""
    print(f"1 - >>>> run_stock_sentiment_analysis for {stock_symbol}")
    
    # Create the workflow
    app = create_stock_sentiment_graph()

    # Initial state
    initial_state = AgentState(
        stock_symbol=stock_symbol,
        news_articles=[],
        processed_articles=[],
        sentiment_scores=[],
        routing_decisions={},
        earnings_analysis={},
        macro_analysis={},
        evaluation_metrics={},
        optimization_feedback={},
        final_report={},
        iteration_count=0,
        max_iterations=3
    )

    # Configuration for the run
    config = {"configurable": {"thread_id": f"stock_analysis_{stock_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}

    try:
        # Stream the execution
        logger.info(f"Starting stock sentiment analysis for {stock_symbol}")

        final_state = None
        async for output in app.astream(initial_state, config=config):
            print(f"------- output :: {output}")
            for key, value in output.items():
                logger.info(f"Completed node: {key}")
                final_state = value  # Keep track of the latest state

        # Get final state
        #final_state = await app.aget_state(config)
        return final_state.values

    except Exception as e:
        logger.error(f"Error in stock sentiment analysis: {e}")
        return {"error": str(e)}

# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python stock_sentiment_analyzer.py <STOCK_SYMBOL>") #NVDA, AAPL, MSFT, | TCS, HDB, INFY, 
        sys.exit(1)

    stock_symbol = sys.argv[1].upper()

    # Run the analysis
    result = asyncio.run(run_stock_sentiment_analysis(stock_symbol))

    # Print results
    print("\n" + "="*80)
    print(f"STOCK SENTIMENT ANALYSIS REPORT FOR {stock_symbol}")
    print("="*80)

    final_report = result.get('final_report', {})
    if 'error' not in final_report:
        print("\nEXECUTIVE SUMMARY:")
        print(final_report.get('executive_summary', 'No summary available'))

        print("\nOVERALL SENTIMENT:")
        print(final_report.get('overall_sentiment', 'neutral').upper())

        print("\nCONFIDENCE LEVEL:")
        print(f"{final_report.get('confidence_level', 0) * 100:.1f}%")

        print("\nEVALUATION METRICS:")
        eval_metrics = result.get('evaluation_metrics', {})
        print(f"Total Articles Analyzed: {eval_metrics.get('article_coverage', {}).get('total_articles', 0)}")
        print(f"Earnings Articles: {eval_metrics.get('article_coverage', {}).get('earnings_articles', 0)}")
        print(f"Macro Articles: {eval_metrics.get('article_coverage', {}).get('macro_articles', 0)}")

    else:
        print(f"\nERROR: {final_report['error']}")

    print("\n" + "="*80)

# src/ingestion/news_fetcher.py

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class NewsFetcher:
    """Fetch news from NewsAPI with caching to save API calls"""
    
    def __init__(self):
        self.api_key = os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            raise ValueError("NEWSAPI_KEY not found in .env")
        
        self.base_url = "https://newsapi.org/v2/everything"
        self.request_count = 0  # Track requests to avoid hitting limit
    
    def fetch_by_ticker(self, ticker, days_back=7, max_articles=10):
        """
        Fetch news for a specific stock ticker
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
            days_back: How many days back to search
            max_articles: Maximum articles to return
        
        Returns:
            List of article dictionaries
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        params = {
            "q": ticker,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            self.request_count += 1
            
            data = response.json()
            
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                
                # Clean and format articles
                cleaned_articles = []
                for article in articles:
                    cleaned_articles.append({
                        "ticker": ticker,
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "author": article.get("author", "")
                    })
                
                print(f"✅ Fetched {len(cleaned_articles)} articles for {ticker}")
                print(f"   API requests used: {self.request_count}/100 today")
                
                return cleaned_articles
            else:
                error = data.get("message", "Unknown error")
                print(f"❌ NewsAPI error: {error}")
                return []
                
        except Exception as e:
            print(f"❌ Failed to fetch news for {ticker}: {str(e)}")
            return []
    
    def fetch_market_news(self, query="stock market", max_articles=10):
        """
        Fetch general market news
        
        Args:
            query: Search query
            max_articles: Maximum articles to return
        
        Returns:
            List of article dictionaries
        """
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            self.request_count += 1
            
            data = response.json()
            
            if data.get("status") == "ok":
                articles = data.get("articles", [])
                print(f"✅ Fetched {len(articles)} market news articles")
                print(f"   API requests used: {self.request_count}/100 today")
                return articles
            else:
                print(f"❌ NewsAPI error: {data.get('message', 'Unknown')}")
                return []
                
        except Exception as e:
            print(f"❌ Failed to fetch market news: {str(e)}")
            return []
    
    def get_request_count(self):
        """Get number of API requests made in this session"""
        return self.request_count

# Example usage
if __name__ == "__main__":
    fetcher = NewsFetcher()
    
    # Fetch Apple news
    apple_news = fetcher.fetch_by_ticker("AAPL", days_back=3, max_articles=5)
    
    for i, article in enumerate(apple_news, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   Published: {article['published_at']}")
import os
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import google.generativeai as genai
import re
import json

# Load environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)

def get_stock_or_crypto_news(query, days=7):
    """
    Fetch news articles about a specific stock or cryptocurrency from NewsAPI
    
    Args:
        query: Stock symbol, company name, or cryptocurrency name
        days: Number of days to look back for news
    
    Returns:
        List of news articles or error message
    """
    # Calculate date for query (last N days)
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Construct the API URL with parameters
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWSAPI_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        if data["status"] == "ok":
            return data["articles"]
        else:
            return f"Error: {data.get('message', 'Unknown error')}"
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching news: {str(e)}"

def get_topic_news(topic, limit=5, days=None, today_only=False):
    """
    Fetch news articles about a general topic from NewsAPI
    
    Args:
        topic: Topic to search for
        limit: Maximum number of articles to return
        days: Number of days to look back for news (None for default API behavior)
        today_only: If True, only get news from today
    
    Returns:
        List of news articles or error message
    """
    # Determine date parameters based on today_only flag
    if today_only:
        # For today's news, we don't set the 'from' parameter in the API call
        # Instead, we'll filter results after receiving them
        from_date = None
    elif days:
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    else:
        from_date = None
    
    # Construct the API URL with parameters
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWSAPI_KEY,
        "pageSize": 100  # Request more articles to ensure we have enough after filtering
    }
    
    if from_date:
        params["from"] = from_date
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        
        if data["status"] == "ok":
            articles = data["articles"]
            
            # Filter for today's news if requested
            if today_only:
                today_date = datetime.now().strftime('%Y-%m-%d')
                articles = [
                    article for article in articles 
                    if article['publishedAt'].startswith(today_date)
                ]
            
            return articles[:limit]  # Return only the requested number of articles
        else:
            return f"Error: {data.get('message', 'Unknown error')}"
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching news: {str(e)}"

def summarize_news_with_gemini(topic, articles, topic_type="general"):
    """
    Use Gemini to generate a summary of news
    
    Args:
        topic: Topic, stock, or cryptocurrency
        articles: List of news articles
        topic_type: "stock", "crypto", or "general"
    
    Returns:
        Summary of the news articles
    """
    if not articles or isinstance(articles, str):
        return articles if isinstance(articles, str) else f"No news found for {topic}."
    
    if len(articles) == 0:
        return f"No news articles found for '{topic}' with the specified filters."
    
    # Format the articles for the prompt
    news_digest = "\n\n".join([
        f"TITLE: {article['title']}\n"
        f"DATE: {article['publishedAt']}\n"
        f"SOURCE: {article['source']['name']}\n"
        f"DESCRIPTION: {article.get('description', 'No description available')}\n"
        f"URL: {article['url']}"
        for article in articles
    ])
    
    # Create appropriate prompt based on the topic type
    if topic_type == "stock":
        prompt = f"""
        Here are recent news articles about {topic} stock:
        
        {news_digest}
        
        Please provide a concise summary of the latest developments for {topic} based on these articles.
        Include the most important updates, trends, and their potential impact on the stock.
        Format your response as a well-structured summary with bullet points for key insights.
        Include emojis where appropriate to make the summary more engaging.
        Structure your response with clear sections and make it easy to read.
        """
    elif topic_type == "crypto":
        prompt = f"""
        Here are recent news articles about {topic} cryptocurrency:
        
        {news_digest}
        
        Please provide a concise summary of the latest developments for {topic} cryptocurrency based on these articles.
        Include the most important updates, trends, price movements, and their potential impact.
        Highlight any regulatory news, technological developments, or market sentiment changes.
        Format your response as a well-structured summary with bullet points for key insights.
        Include emojis where appropriate to make the summary more engaging.
        Structure your response with clear sections and make it easy to read.
        """
    else:
        prompt = f"""
        Here are recent news articles about "{topic}":
        
        {news_digest}
        
        Please provide a concise summary of these {len(articles)} articles.
        Focus on the most important information, key developments, and significant updates.
        For each article, provide 2-3 key takeaways in bullet points.
        Include article titles and sources.
        Format your response with clear headings and make it easy to read.
        Include emojis where appropriate to make the summary more engaging.
        """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def analyze_query(user_query):
    """
    Analyze the user query to determine intent and extract parameters
    
    Args:
        user_query: User's question or request
    
    Returns:
        Dictionary with query type and parameters
    """
    # Create a comprehensive prompt to analyze the user's query
    analysis_prompt = f"""
    Analyze this user query: "{user_query}"
    
    Determine if the user is asking about:
    1. A specific stock/company news
    2. A specific cryptocurrency news (Bitcoin, Ethereum, Dogecoin, etc.)
    3. General news about a topic
    
    Extract these parameters:
    - topic: the main subject (stock symbol, company name, cryptocurrency, or general topic)
    - count: number of news items requested (default to 5 if not specified)
    - timeframe: "today" if explicitly asking for today's news or mentions "published today", "recent" for recent news (default)
    
    Format your response as JSON with these fields:
    {{
      "query_type": "stock" or "crypto" or "general",
      "topic": "extracted topic",
      "count": number,
      "timeframe": "today" or "recent"
    }}
    
    Examples:
    - "Tell me about Bitcoin news" should return query_type "crypto"
    - "What's happening with AAPL?" should return query_type "stock"
    - "Latest news on climate change" should return query_type "general"
    
    Return only valid JSON with no additional text.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(analysis_prompt)
        
        # Clean the response to ensure it's valid JSON (remove any markdown code block markers)
        json_str = response.text.strip()
        json_str = re.sub(r'^```json', '', json_str)
        json_str = re.sub(r'```$', '', json_str)
        json_str = json_str.strip()
        
        # Parse the JSON response
        query_params = json.loads(json_str)
        
        # Extra checking for today keyword
        if "today" in user_query.lower() or "published today" in user_query.lower():
            query_params["timeframe"] = "today"
            
        return query_params
    except Exception as e:
        # Default fallback if analysis fails
        # Check for "today" explicitly
        timeframe = "today" if "today" in user_query.lower() else "recent"
        
        # Basic check for crypto-related keywords
        query_lower = user_query.lower()
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "dogecoin", "doge", 
                          "blockchain", "binance", "coinbase", "altcoin", "token"]
        
        query_type = "general"
        for keyword in crypto_keywords:
            if keyword in query_lower:
                query_type = "crypto"
                break
                
        return {
            "query_type": query_type,
            "topic": user_query,
            "count": 5,
            "timeframe": timeframe
        }

def get_news_response(user_query):
    """
    Process user query and return news response
    
    Args:
        user_query: User's question or request related to news
    
    Returns:
        Formatted news response
    """
    try:
        # Analyze the query to determine intent and parameters
        query_params = analyze_query(user_query)
        
        query_type = query_params.get("query_type", "general")
        topic = query_params.get("topic", "")
        count = query_params.get("count", 5)
        timeframe = query_params.get("timeframe", "recent")
        
        if not topic:
            return "I couldn't identify a specific topic in your query. Please specify what you'd like to know about."
        
        # Handle stock or crypto news queries
        if query_type in ["stock", "crypto"]:
            days = 1 if timeframe == "today" else 7
            articles = get_stock_or_crypto_news(topic, days=days)
            
            # For cryptocurrencies, add the coin name to the search if not already specified
            if query_type == "crypto" and topic.lower() not in ["cryptocurrency", "crypto"]:
                # For topics like "bitcoin", enhance search to include "bitcoin cryptocurrency"
                enhanced_articles = get_stock_or_crypto_news(f"{topic} cryptocurrency", days=days)
                # Combine and deduplicate articles by URL
                seen_urls = set()
                combined_articles = []
                for article in articles + enhanced_articles:
                    if article['url'] not in seen_urls:
                        seen_urls.add(article['url'])
                        combined_articles.append(article)
                articles = combined_articles[:count]
            
            response = summarize_news_with_gemini(topic, articles, topic_type=query_type)
            time_period = "today" if timeframe == "today" else f"the past {days} days"
            
            # Add a header to make the response more structured
            header = f"📰 NEWS UPDATE: {topic.upper()} {'STOCK' if query_type == 'stock' else 'CRYPTOCURRENCY'} - {time_period.upper()} 📰\n\n"
            return header + response
        
        # Handle general topic news queries
        else:
            today_only = (timeframe == "today")
            articles = get_topic_news(topic, limit=count, today_only=today_only)
            response = summarize_news_with_gemini(topic, articles, topic_type="general")
            
            time_period = "TODAY" if today_only else "RECENT"
            header = f"📰 {time_period} NEWS: {topic.upper()} 📰\n\n"
            return header + response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing your news query: {str(e)}"

# Function to check if a query is news-related
def is_news_query(query):
    """
    Determine if a user query is related to news
    
    Args:
        query: User's query string
    
    Returns:
        Boolean indicating if the query is about news
    """
    query = query.lower()
    news_keywords = [
        "news", "latest", "update", "headlines", "today's news", "recent developments",
        "what's happening", "current events", "breaking news", "report",
        "published today", "article", "coverage", "press", "media coverage",
        "stock news", "crypto news", "financial news", "market news"
    ]
    
    for keyword in news_keywords:
        if keyword in query:
            return True
    
    # Additional checks for queries that might be news-related
    # Check for stock ticker patterns (e.g., AAPL, MSFT)
    if re.search(r'\b[A-Z]{2,5}\b', query) and any(word in query for word in ["stock", "price", "company", "shares"]):
        return True
        
    # Check for crypto-related patterns
    crypto_terms = ["bitcoin", "btc", "ethereum", "eth", "crypto", "dogecoin", "blockchain", "coin"]
    if any(term in query for term in crypto_terms):
        return True
        
    return False

# Handle main execution for testing
if __name__ == "__main__":
    print("News Bot (type 'exit' to quit)")
    print("Examples:")
    print("- 'Tell me today's updates on Tesla stock'")
    print("- 'Show me top 5 news related to tax published today'")
    print("- 'What's the latest news about Bitcoin?'")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye!")
            break
        
        response = get_news_response(user_input)
        print(f"\nBot: {response}")

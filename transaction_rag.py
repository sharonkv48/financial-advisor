

import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from datetime import datetime
import re
import json

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define the index name
index_name = "transactions-index"
index = pc.Index(index_name)

# Get index dimension (alternatively, hardcode to 768 based on the error message)
INDEX_DIMENSION = 768  # As per the error message

def extract_json_from_text(text):
    """
    Extract JSON from text that might contain markdown code blocks or other non-JSON content
    """
    # Try to find JSON pattern with markdown code blocks
    json_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
    match = re.search(json_pattern, text)
    
    if match:
        return match.group(1)
    
    # If no markdown block, try to find a JSON object directly
    json_pattern = r'({[\s\S]*?})'
    match = re.search(json_pattern, text)
    
    if match:
        return match.group(1)
    
    # Return the original text if no JSON pattern is found
    return text

def preprocess_query(query_text):
    """
    Use Gemini API to preprocess and correct the query
    """
    try:
        prompt = f"""
        I need you to correct any spelling mistakes in this query about financial transactions.
        Extract and return the key category or merchant the user is asking about.
        
        Original query: "{query_text}"
        
        Format your response as a simple JSON object with the following fields:
        - corrected_query: The query with spelling fixed
        - search_term: The main category or merchant being queried (just the term, not the whole query)
        - is_spending_query: true/false if this is asking about spending
        - is_advice_query: true/false if this is asking for financial advice about spending patterns
        - is_total_query: true/false if this is asking about total spending across all categories
        
        Only respond with the JSON object, no markdown formatting, no explanations, no code blocks.
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Process the response to extract the info
        json_text = extract_json_from_text(response.text)
        
        try:
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError as e:
            print(f"Could not parse Gemini response as JSON: {e}")
            print(f"Raw response: {response.text}")
            print(f"Extracted JSON text: {json_text}")
            
            # Check if this is likely an advice query
            advice_patterns = [
                r'how (can|to) (save|reduce|cut)',
                r'where (can|to) (save|reduce|cut)',
                r'(reduce|cut) (spending|expenses)',
                r'save money',
                r'spending too much',
                r'budget',
                r'financial advice'
            ]
            
            # Check if this is a total spending query
            total_patterns = [
                r'total spent',
                r'spent (in|across) total',
                r'spent till now',
                r'spent so far',
                r'total spending',
                r'overall spending',
                r'all spending',
                r'all transactions',
                r'total expense'
            ]
            
            is_advice = any(re.search(pattern, query_text.lower()) for pattern in advice_patterns)
            is_total = any(re.search(pattern, query_text.lower()) for pattern in total_patterns)
            
            # Extract search term with basic regex as fallback
            match = re.search(r'(spent on|for|on) ([a-zA-Z\s]+)', query_text.lower())
            if match and not is_total:
                return {
                    "corrected_query": query_text,
                    "search_term": match.group(2).strip(),
                    "is_spending_query": True,
                    "is_advice_query": is_advice,
                    "is_total_query": is_total
                }
            return {
                "corrected_query": query_text,
                "search_term": "" if is_total else (query_text.split()[-1] if query_text.split() else ""),
                "is_spending_query": True,
                "is_advice_query": is_advice,
                "is_total_query": is_total
            }
    except Exception as e:
        print(f"Error in query preprocessing: {e}")
        # Check if this is likely an advice query
        advice_patterns = [
            r'how (can|to) (save|reduce|cut)',
            r'where (can|to) (save|reduce|cut)',
            r'(reduce|cut) (spending|expenses)',
            r'save money',
            r'spending too much',
            r'budget',
            r'financial advice'
        ]
        
        # Check if this is a total spending query
        total_patterns = [
            r'total spent',
            r'spent (in|across) total',
            r'spent till now',
            r'spent so far',
            r'total spending',
            r'overall spending',
            r'all spending',
            r'all transactions',
            r'total expense'
        ]
        
        is_advice = any(re.search(pattern, query_text.lower()) for pattern in advice_patterns)
        is_total = any(re.search(pattern, query_text.lower()) for pattern in total_patterns)
        
        # Fallback to basic pattern matching
        match = re.search(r'(spent on|for|on) ([a-zA-Z\s]+)', query_text.lower())
        if match and not is_total:
            return {
                "corrected_query": query_text,
                "search_term": match.group(2).strip(),
                "is_spending_query": True,
                "is_advice_query": is_advice,
                "is_total_query": is_total
            }
        return {
            "corrected_query": query_text,
            "search_term": "" if is_total else (query_text.split()[-1] if query_text.split() else ""),
            "is_spending_query": True,
            "is_advice_query": is_advice,
            "is_total_query": is_total
        }

def is_financial_advice_query(query_text):
    """
    Determine if the query is asking for financial advice about spending
    """
    advice_patterns = [
        r'how (can|to) (save|reduce|cut)',
        r'where (can|to) (save|reduce|cut)',
        r'(reduce|cut) (spending|expenses)',
        r'save money',
        r'spending too much',
        r'budget',
        r'financial advice'
    ]
    
    return any(re.search(pattern, query_text.lower()) for pattern in advice_patterns)

def is_total_spending_query(query_text):
    """
    Determine if the query is asking about total spending across all categories
    """
    total_patterns = [
        r'total spent',
        r'spent (in|across) total',
        r'spent till now',
        r'spent so far',
        r'total spending',
        r'overall spending',
        r'all spending',
        r'all transactions',
        r'total expense',
        r'how much (have|did) I spent',
        r'how much money (have|did) I spent'
    ]
    
    return any(re.search(pattern, query_text.lower()) for pattern in total_patterns)

def get_all_transactions(limit=1000):
    """
    Retrieve all transactions from Pinecone up to a limit
    """
    try:
        results = index.query(
            vector=[0] * INDEX_DIMENSION,
            include_metadata=True,
            top_k=limit
        )
        return results.matches
    except Exception as e:
        print(f"Error retrieving all transactions: {e}")
        return []

def query_transactions(query_text):
    """
    Query the transactions from Pinecone based on user input
    """
    # Preprocess and correct query
    query_info = preprocess_query(query_text)
    search_term = query_info["search_term"]
    
    print(f"Searching for: '{search_term}' (from: '{query_info['corrected_query']}')")
    
    # If this is a total spending query, get all transactions
    if query_info.get("is_total_query", False) or is_total_spending_query(query_text):
        query_info["is_total_query"] = True
        all_transactions = get_all_transactions()
        return all_transactions, query_info
    
    # If this is an advice query, we might need all transactions
    if query_info.get("is_advice_query", False):
        all_transactions = get_all_transactions()
        return all_transactions, query_info
    
    if search_term:
        # Query Pinecone for transactions matching the category
        try:
            # First try exact match with category field
            results = index.query(
                vector=[0] * INDEX_DIMENSION,
                filter={"category": {"$eq": search_term.lower()}},
                include_metadata=True,
                top_k=100
            )
            
            # If no results, try fuzzy match with category
            if len(results.matches) == 0:
                results = index.query(
                    vector=[0] * INDEX_DIMENSION,
                    filter={"category": {"$contregexp": f".*{search_term.lower()}.*"}},
                    include_metadata=True,
                    top_k=100
                )
            
            # If still no results, try matching transaction_to field
            if len(results.matches) == 0:
                results = index.query(
                    vector=[0] * INDEX_DIMENSION,
                    filter={"transaction_to": {"$contregexp": f".*{search_term}.*"}},
                    include_metadata=True,
                    top_k=100
                )
            
            return results.matches, query_info
        except Exception as e:
            print(f"Query error: {e}")
            return [], query_info
    
    return [], query_info

def analyze_transactions(matches):
    """
    Analyze transaction data to provide useful summaries
    """
    if not matches:
        return "I couldn't find any transactions matching your query."
    
    total_amount = 0
    transactions = []
    categories = {}
    
    for match in matches:
        metadata = match.metadata
        amount = metadata.get("amount", 0)
        date = metadata.get("date", "")
        transaction_to = metadata.get("transaction_to", "")
        category = metadata.get("category", "")
        
        total_amount += amount
        transactions.append({
            "amount": amount,
            "date": date,
            "transaction_to": transaction_to,
            "category": category
        })
        
        # Aggregate by category for spending analysis
        if category:
            if category in categories:
                categories[category] += amount
            else:
                categories[category] = amount
    
    # Sort transactions by date (with error handling for date parsing)
    try:
        transactions.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d %I:%M %p") if x["date"] else datetime.min)
    except ValueError:
        # If date format is different, just keep original order
        pass
    
    # Sort categories by spend amount (descending)
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "total_amount": total_amount,
        "count": len(transactions),
        "transactions": transactions,
        "categories": sorted_categories
    }

def generate_spending_advice(analysis_result):
    """
    Generate financial advice based on spending patterns
    """
    if isinstance(analysis_result, str):
        return "I don't have enough transaction data to provide spending advice. Please try a more specific query."
    
    # Prepare category spending data for the prompt
    category_details = ""
    for category, amount in analysis_result["categories"][:10]:
        percentage = (amount / analysis_result["total_amount"]) * 100 if analysis_result["total_amount"] > 0 else 0
        category_details += f"- {category}: INR {amount} ({percentage:.1f}% of total)\n"
    
    prompt = f"""
    I need you to provide personalized financial advice based on this spending data:
    
    Total spending: INR {analysis_result['total_amount']}
    Number of transactions: {analysis_result['count']}
    
    Top spending categories:
    {category_details}
    
    Please provide specific, actionable financial advice for reducing expenses and improving savings. Your advice should:
    1. Identify the top areas where spending could be reduced
    2. Suggest specific strategies to cut costs in those areas
    3. Recommend budgeting approaches based on this spending pattern
    4. Provide concrete examples of how to save money in daily life
    
    Important: Be AUTHORITATIVE and DEFINITIVE. Speak as the ULTIMATE financial advisor.
    - NO suggestions of consulting external advisors
    - Give CONFIDENT, DIRECT, and ACTIONABLE advice
    - Use PROFESSIONAL financial language
    - Focus on CONCRETE strategies, not general platitudes
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error in advice generation: {e}")
        # Fallback response if Gemini API fails
        return f"Based on your spending patterns, the top categories are {', '.join([cat for cat, _ in analysis_result['categories'][:3]])}. Consider reducing non-essential expenses in these areas to improve your savings."

def generate_total_spending_response(analysis_result):
    """
    Generate a response for total spending queries
    """
    if isinstance(analysis_result, str):
        return "I couldn't find any transaction data to calculate your total spending."
    
    # Prepare category breakdown for the prompt
    top_categories = ""
    for category, amount in analysis_result["categories"][:5]:
        percentage = (amount / analysis_result["total_amount"]) * 100 if analysis_result["total_amount"] > 0 else 0
        top_categories += f"- {category}: INR {amount} ({percentage:.1f}% of total)\n"
    
    earliest_date = None
    latest_date = None
    
    # Find date range
    try:
        if analysis_result["transactions"]:
            dates = [datetime.strptime(t["date"], "%Y-%m-%d %I:%M %p") for t in analysis_result["transactions"] if t["date"]]
            if dates:
                earliest_date = min(dates).strftime("%B %d, %Y")
                latest_date = max(dates).strftime("%B %d, %Y")
    except ValueError:
        # If date parsing fails, just skip date range
        pass
    
    date_range = f"from {earliest_date} to {latest_date}" if earliest_date and latest_date else ""
    
    prompt = f"""
    The user asked about their total spending.
    
    I found {analysis_result['count']} transaction(s) with a total spending of INR {analysis_result['total_amount']} {date_range}.
    
    Top spending categories:
    {top_categories}
    
    Please provide a helpful, concise response that explains their total spending.
    Include the total amount in INR, the time period if available, the number of transactions, and a brief breakdown of top spending categories.
    Always use "INR" before the amount in your response.
    Make the response conversational and friendly.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error in total spending response: {e}")
        # Fallback response if Gemini API fails
        date_info = f" {date_range}" if date_range else ""
        return f"Your total spending{date_info} is INR {analysis_result['total_amount']} across {analysis_result['count']} transactions."

def generate_response(query, analysis_result, query_info):
    """
    Generate a natural language response using Gemini
    """
    if isinstance(analysis_result, str):
        return analysis_result
    
    # Check if this is a total spending query
    if query_info.get("is_total_query", False) or is_total_spending_query(query):
        return generate_total_spending_response(analysis_result)
    
    # Check if this is an advice query
    if query_info.get("is_advice_query", False) or is_financial_advice_query(query):
        return generate_spending_advice(analysis_result)
    
    # Prepare transaction details for the prompt
    transaction_details = ""
    for i, txn in enumerate(analysis_result["transactions"][:5]):
        transaction_details += f"- {txn['date']}: INR {txn['amount']} to {txn['transaction_to']} ({txn['category']})\n"
    
    if len(analysis_result["transactions"]) > 5:
        transaction_details += f"- ... and {len(analysis_result['transactions'])-5} more transaction(s)\n"
    
    prompt = f"""
    The user asked: "{query}"
    Which was interpreted as: "{query_info['corrected_query']}"
    Searching for transactions related to: "{query_info['search_term']}"
    
    I found {analysis_result['count']} relevant transaction(s) with a total amount of INR {analysis_result['total_amount']}.
    
    Recent transactions:
    {transaction_details}
    
    Please provide a helpful, concise response that answers the user's question about their spending.
    Focus on giving the total amount spent in Indian Rupees (INR), the time period, and any other relevant insights.
    Always use "INR" before the amount in your response.
    
    If the original query had spelling mistakes, subtly acknowledge you understood what they meant.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback response if Gemini API fails
        return f"You spent a total of INR {analysis_result['total_amount']} across {analysis_result['count']} transactions related to {query_info['search_term']}."

def process_user_query(user_query):
    """
    Main function to process user queries about transactions
    """
    # Query transactions from Pinecone (this now returns query_info as well)
    matches, query_info = query_transactions(user_query)
    
    # Analyze the transaction data
    analysis = analyze_transactions(matches)
    
    # Generate natural language response with Gemini
    response = generate_response(user_query, analysis, query_info)
    
    return response

# Example usage
if __name__ == "__main__":
    print("Transaction Analysis Chatbot initialized.")
    print("You can ask questions like 'How much did I spend on food?' or 'How much have I spent in total?'")
    
    while True:
        user_query = input("\nAsk about your transactions (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        
        try:
            response = process_user_query(user_query)
            print("\nChatbot:", response)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try a different query.")

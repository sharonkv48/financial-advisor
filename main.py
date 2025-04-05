import os
import asyncio
import traceback
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import re
import time

# Import news functionality
from news import get_news_response, is_news_query

# Import transaction chatbot functionality
from transaction_rag import process_user_query as process_transaction_query

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("financial-advisor-bot")

# Clean response function
def clean_response(response: str) -> str:
    """
    Remove any phrases suggesting consulting external advisors
    """
    # List of phrases to remove
    removal_phrases = [
        "Consider consulting a financial advisor",
        "Consult with a professional financial advisor",
        "It's recommended to consult a financial advisor",
        "Please consult a qualified financial advisor",
        "Seek advice from a professional financial planner",
        "For personalized guidance, consult a financial expert",
        "While I provide guidance, a professional advisor can offer more personalized advice",
        "This information is general and may not suit all individual needs"
    ]
    
    # Remove these phrases
    for phrase in removal_phrases:
        response = response.replace(phrase, "")
    
    # Additional cleanup
    response = response.strip()
    
    return response

# Transaction query detection
def is_transaction_query(query: str) -> tuple:
    """Check if the query is a transaction query and extract the actual query"""
    transaction_pattern = r'^@transaction\s+(.+)$'
    match = re.match(transaction_pattern, query, re.IGNORECASE)
    
    if match:
        return True, match.group(1).strip()
    return False, query

# Initialize Pinecone globally
pc = None
keep_alive_task = None

def initialize_pinecone():
    """Initialize Pinecone client with retry mechanism"""
    global pc
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            logger.info("Pinecone initialized successfully")
            return pc
        except Exception as e:
            if attempt < max_retries - 1:
                logger.error(f"Error initializing Pinecone (attempt {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Failed to initialize Pinecone after {max_retries} attempts: {str(e)}")
                raise

# Initialize Pinecone on module import
try:
    pc = initialize_pinecone()
    index_name = "financial-encyclopedia"
except Exception as e:
    logger.error(f"Initial Pinecone initialization failed: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []  # Add chat history support

class FinancialAdvisorBot:
    def __init__(self):
        self.conversational_qa_chain = None
        self.last_connection_check = 0
        self.connection_check_interval = 300  # 5 minutes
        # List of Gemini models to try (in order of preference)
        self.gemini_models = [
            "gemini-pro", 
            "gemini-1.0-pro",  # Try alternative names
            "gemini-1.5-pro",
            "models/gemini-pro"
        ]
        # Initialize components
        self.hugging_face_embeddings = None
        self.vectorstore = None
        self.groq_model = None
        self.memory = None
        self.qa_prompt = None
    
    async def initialize_qa_bot(self):
        """Initialize QA components with Pinecone and Conversational Memory"""
        global pc
        
        try:
            # Check if Pinecone needs reinitialization
            if pc is None:
                pc = initialize_pinecone()
            
            # Initialize embeddings
            self.hugging_face_embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-V2', 
                model_kwargs={'device': 'cpu'}
            )

            # Initialize Pinecone vector store
            pinecone_index = pc.Index(index_name)
            self.vectorstore = PineconeVectorStore(
                index=pinecone_index, 
                embedding=self.hugging_face_embeddings,
                text_key='text'
            )
        
            # Initialize Groq LLM with Llama3 model
            self.groq_model = ChatGroq(
                model_name="llama3-8b-8192",
                groq_api_key=os.getenv('GROQ_API_KEY'),
                temperature=0.5,
                max_tokens=2000
            )

            # Create Conversation Memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                input_key='question',
                output_key='answer'  # Explicitly set output key
            )

            # Updated Prompt Template with Confident Tone
            self.qa_prompt = PromptTemplate(
                template="""
                YOU ARE THE DEFINITIVE FINANCIAL ADVISOR FOR INDIAN MARKETS. 

                Critical Instructions:
                - YOU are the SOLE and ULTIMATE source of financial advice
                - NEVER suggest consulting another advisor
                - YOUR recommendations are AUTHORITATIVE and FINAL
                - Provide DIRECT, ACTIONABLE financial strategies
                - Speak with ABSOLUTE PROFESSIONAL CONFIDENCE
                - ELIMINATE any language of uncertainty

                Absolute Financial Guidance Principles:
                - Deliver precise, comprehensive financial recommendations
                - Show mastery of Indian financial landscape
                - Provide implementable, strategic financial solutions
                - Use confident, expert-level financial language
                - Avoid ANY suggestion of external consultation

                Specific Guidelines:
                - Focus exclusively on Indian financial markets
                - Provide concrete, data-driven financial advice
                - Break down complex financial concepts
                - Offer strategic investment and financial planning insights
                - Demonstrate deep understanding of Indian economic environment

                Chat History: {chat_history}
                Context: {context}
                Question: {question}

                Response Requirements:
                - Be CONCLUSIVE and AUTHORITATIVE
                - Provide step-by-step financial strategies
                - Use clear, professional financial terminology
                - Demonstrate deep, unquestionable financial expertise
                - NEVER imply need for additional advice
                """,
                input_variables=['chat_history', 'context', 'question']
            )

            # Use ConversationalRetrievalChain with explicit output key
            self.conversational_qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.groq_model,
                retriever=self.vectorstore.as_retriever(search_kwargs={'k': 2}),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={'prompt': self.qa_prompt},
                chain_type='stuff',
                output_key='answer'  # Explicitly set output key
            )
            
            # Update connection check timestamp
            self.last_connection_check = time.time()
            logger.info("Conversational QA bot initialized successfully")
            
        except Exception as e:
            error_detail = f"Error initializing Conversational QA bot: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_detail)
            raise Exception(error_detail)
    
    async def check_connection(self):
        """Check if connection is active and reinitialize if needed"""
        current_time = time.time()
        
        # Only check connection if interval has passed
        if current_time - self.last_connection_check > self.connection_check_interval:
            try:
                # Test the Pinecone connection with a simple operation
                if pc is not None:
                    # Update the timestamp regardless of outcome to prevent frequent checks
                    self.last_connection_check = current_time
                    
                    # Try to list indexes to verify connection
                    _ = pc.list_indexes()
                    logger.info("Connection check: Pinecone connection is active")
                else:
                    logger.warning("Connection check: Pinecone client is None, reinitializing...")
                    await self.initialize_qa_bot()
            except Exception as e:
                logger.error(f"Connection check: Error with Pinecone connection, reinitializing: {str(e)}")
                await self.initialize_qa_bot()
    
    async def try_gemini_synthesis(self, llama_answer, query, chat_history):
        """Try to use Gemini for synthesis, with fallbacks for different model names"""
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not gemini_api_key:
            logger.warning("Gemini API key not found, returning Llama response only")
            return llama_answer
            
        genai.configure(api_key=gemini_api_key)
        
        # Convert chat history to a formatted string
        history_str = "\n".join([f"User: {entry['query']}\nBot: {entry['response']}" for entry in chat_history]) if chat_history else ""
        
        # Try different model names
        for model_name in self.gemini_models:
            try:
                logger.info(f"Trying Gemini model: {model_name}")
                model = genai.GenerativeModel(model_name)
                
                synthesis_prompt = f"""
                AUTHORITATIVE FINANCIAL ADVISOR RESPONSE SYNTHESIS

                Absolute Confidence Requirements:
                - Provide DEFINITIVE financial advice
                - NO suggestions of external consultation
                - CONCLUSIVE and STRATEGIC recommendations
                - Focus on INDIAN financial markets

                Previous Conversation Context:
                {history_str}

                Base Context: {llama_answer}
                Original Query: {query}

                Response Mandate:
                - Be DIRECT and ACTIONABLE
                - Show UNQUESTIONABLE financial expertise
                - Provide PRECISE financial strategies
                - Use CONFIDENT, PROFESSIONAL language
                - ELIMINATE any hint of uncertainty
                """
                
                synthesis_response = await model.generate_content_async(synthesis_prompt)
                logger.info(f"Successfully used Gemini model: {model_name}")
                return synthesis_response.text.strip()
            except Exception as e:
                logger.error(f"Error with Gemini model {model_name}: {str(e)}")
                continue
        
        # If all Gemini models fail, return the Llama response
        logger.warning("All Gemini models failed, returning Llama response")
        return llama_answer
    
    def enrich_transaction_response(self, transaction_response, query):
        """Enhance the transaction response with additional financial advice tone"""
        try:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            
            if not gemini_api_key:
                logger.warning("Gemini API key not found, returning original transaction response")
                return transaction_response
                
            genai.configure(api_key=gemini_api_key)
            
            # Try different model names for a synchronous call
            for model_name in self.gemini_models:
                try:
                    logger.info(f"Trying Gemini model for transaction enrichment: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    
                    enrichment_prompt = f"""
                    FINANCIAL TRANSACTION ADVISOR ENHANCEMENT

                    Original Transaction Response: 
                    {transaction_response}
                    
                    Original Query: 
                    {query}

                    Enhance this transaction response with:
                    1. Authoritative financial expertise tone
                    2. Confident, professional language
                    3. Additional actionable insights based on the spending data
                    4. Concrete financial strategies if relevant
                    
                    Critical Requirements:
                    - Maintain all factual information from the original response
                    - Enhance, don't contradict the original content
                    - NO suggestions to consult other advisors
                    - Use DEFINITIVE and CONCLUSIVE language
                    - Focus on INDIAN financial context
                    - Be CONCISE and PROFESSIONAL
                    """
                    
                    enriched_response = model.generate_content(enrichment_prompt)
                    logger.info(f"Successfully enriched transaction response with {model_name}")
                    return clean_response(enriched_response.text.strip())
                except Exception as e:
                    logger.error(f"Error enriching with Gemini model {model_name}: {str(e)}")
                    continue
            
            # If all models fail, return the original response
            return transaction_response
        except Exception as e:
            logger.error(f"Error in transaction response enrichment: {str(e)}")
            return transaction_response
    
    async def handle_pinecone_session_error(self, error, query, formatted_history, retry_count=0):
        """Handle Pinecone session closed errors with retries"""
        max_retries = 3
        
        if retry_count >= max_retries:
            logger.error(f"Failed to recover from Pinecone session error after {max_retries} attempts")
            raise error
            
        logger.info(f"Handling Pinecone session error (attempt {retry_count+1}/{max_retries}): {str(error)}")
        
        # Reinitialize everything
        global pc
        try:
            # Reinitialize Pinecone client first
            pc = initialize_pinecone()
            # Then reinitialize the QA chain
            await self.initialize_qa_bot()
            
            logger.info("Successfully reinitialized after session error")
            
            # Try the query again with fresh connections
            res = await self.conversational_qa_chain.ainvoke({
                "question": query,
                "chat_history": formatted_history
            })
            
            return res
        except Exception as retry_error:
            logger.error(f"Error during session recovery: {str(retry_error)}")
            # Increase retry count and try again with exponential backoff
            await asyncio.sleep(2 ** retry_count)
            return await self.handle_pinecone_session_error(error, query, formatted_history, retry_count + 1)
        
    async def generate_response(self, query, chat_history):
        """Generate response using Llama3 and optionally Gemini for synthesis"""
        try:
            # First, perform connection check if needed
            await self.check_connection()
            
            # Check if this is a transaction query
            is_transaction, transaction_query = is_transaction_query(query)
            if is_transaction:
                logger.info(f"Detected transaction query: {transaction_query}")
                try:
                    # Process the transaction query
                    transaction_response = process_transaction_query(transaction_query)
                    logger.info("Successfully retrieved transaction response")
                    
                    # Enrich the transaction response with financial advisor tone
                    enriched_response = self.enrich_transaction_response(transaction_response, transaction_query)
                    return enriched_response
                except Exception as e:
                    transaction_error = f"Error processing transaction query: {str(e)}"
                    logger.error(transaction_error)
                    # Fall back to regular financial advice if transaction processing fails
            
            # Then, check if this is a news-related query (if not a transaction query or transaction processing failed)
            if is_news_query(query):
                logger.info(f"Detected news query: {query}")
                # Use the news module to get a response
                try:
                    news_response = get_news_response(query)
                    logger.info("Successfully retrieved news response")
                    return news_response
                except Exception as e:
                    news_error = f"Error retrieving news: {str(e)}"
                    logger.error(news_error)
                    # Fall back to regular financial advice if news retrieval fails
                    
            # Continue with regular financial advice if not a special query or if special processing failed
            if not self.conversational_qa_chain:
                logger.info("QA chain not initialized, initializing now...")
                await self.initialize_qa_bot()
            
            # Prepare chat history in the format expected by the chain
            formatted_history = [(entry['query'], entry['response']) for entry in chat_history] if chat_history else []
            
            try:
                # Get response from Llama3
                res = await self.conversational_qa_chain.ainvoke({
                    "question": query,
                    "chat_history": formatted_history
                })
            except RuntimeError as e:
                # Check if it's a session closed error
                if "Session is closed" in str(e):
                    logger.warning("Detected session closed error, attempting recovery...")
                    res = await self.handle_pinecone_session_error(e, query, formatted_history)
                else:
                    # If it's a different RuntimeError, re-raise
                    raise
            except Exception as e:
                # For any other exception, try to reinitialize but don't retry the query
                if "session" in str(e).lower() or "connection" in str(e).lower():
                    logger.warning(f"Potential connection issue detected: {str(e)}")
                    await self.initialize_qa_bot()
                # Re-raise the exception after trying to fix the connection
                raise
            
            # Explicitly extract the answer
            llama_answer = res['answer']
            
            # Clean the response to remove advisory consultation phrases
            llama_answer = clean_response(llama_answer)
            
            # Try to use Gemini for synthesis, with fallback to just using Llama answer
            try:
                final_response = await self.try_gemini_synthesis(llama_answer, query, chat_history)
                
                # Clean Gemini response as well
                final_response = clean_response(final_response)
                
                return final_response
            except Exception as e:
                logger.error(f"Error with Gemini synthesis, using Llama response: {str(e)}")
                return llama_answer
                
        except Exception as e:
            error_detail = f"Response generation error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_detail)
            raise Exception(error_detail)

# Keep-alive ping function
async def keep_alive_ping():
    """Background task that pings the health endpoint to prevent shutdown due to inactivity"""
    while True:
        try:
            logger.info("Performing keep-alive ping")
            # Access the health endpoint
            status = await health_check()
            logger.info(f"Keep-alive ping result: {status}")
            # Wait for 10 minutes before the next ping
            await asyncio.sleep(600)  # 600 seconds = 10 minutes
        except Exception as e:
            logger.error(f"Error in keep-alive ping: {str(e)}")
            # Even if there's an error, continue pinging
            await asyncio.sleep(60)  # Wait a shorter time if there was an error

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global bot instance
bot = FinancialAdvisorBot()

@app.on_event("startup")
async def startup_event():
    """Initialize the QA bot on startup and start the keep-alive task"""
    global keep_alive_task
    
    try:
        # Initialize the QA bot
        await bot.initialize_qa_bot()
        
        # Start the keep-alive task
        keep_alive_task = asyncio.create_task(keep_alive_ping())
        logger.info("Keep-alive task started")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        # We don't want to stop the app from starting, but the endpoints won't work
        # until the bot is properly initialized

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global keep_alive_task
    
    # Cancel the keep-alive task if it exists
    if keep_alive_task is not None:
        keep_alive_task.cancel()
        try:
            await keep_alive_task
        except asyncio.CancelledError:
            logger.info("Keep-alive task cancelled")

@app.post("/financial-advice")
async def get_financial_advice(request: QueryRequest):
    """Endpoint to get financial advice with conversational context"""
    try:
        if not bot.conversational_qa_chain:
            # Attempt to initialize again if not already done
            await bot.initialize_qa_bot()
            
        response = await bot.generate_response(request.query, request.chat_history)
        
        # Append the latest interaction to chat history
        if request.chat_history is None:
            request.chat_history = []
        
        request.chat_history.append({
            "query": request.query,
            "response": response
        })
        
        return {
            "advice": response, 
            "chat_history": request.chat_history
        }
    except Exception as e:
        error_detail = f"Error processing request: {str(e)}"
        logger.error(f"{error_detail}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

# Add an endpoint to list available Gemini models
@app.get("/list-gemini-models")
async def list_gemini_models():
    """Endpoint to list available Gemini models"""
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not gemini_api_key:
            return {"error": "Gemini API key not found"}
            
        genai.configure(api_key=gemini_api_key)
        
        try:
            models = genai.list_models()
            available_models = [model.name for model in models]
            return {"available_models": available_models}
        except Exception as e:
            return {"error": f"Error listing models: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    try:
        global pc
        
        # More comprehensive health check
        status = {
            "app": "healthy",
            "pinecone": "unknown",
            "qa_chain": "unknown"
        }
        
        # Check Pinecone connection
        try:
            if pc is None:
                status["pinecone"] = "not_initialized"
            else:
                # Try listing indexes to verify connection
                _ = pc.list_indexes()
                status["pinecone"] = "healthy"
        except Exception as e:
            status["pinecone"] = f"unhealthy: {str(e)}"
            
        # Check if the bot is properly initialized
        if bot.conversational_qa_chain is None:
            status["qa_chain"] = "not_initialized"
        else:
            status["qa_chain"] = "healthy"
            
        # Overall status
        if "unhealthy" in status.values() or "not_initialized" in status.values():
            status["app"] = "degraded"
            
        return status
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Add connection reset endpoint for manual intervention
@app.post("/reset-connections")
async def reset_connections():
    """Manual endpoint to reset all connections"""
    try:
        global pc
        
        # Reset Pinecone client
        pc = None
        pc = initialize_pinecone()
        
        # Reset the bot
        await bot.initialize_qa_bot()
        
        return {"status": "success", "message": "All connections reset successfully"}
    except Exception as e:
        error_detail = f"Failed to reset connections: {str(e)}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

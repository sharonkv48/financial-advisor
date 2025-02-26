import os
import asyncio
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Pinecone Configuration
try:
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index_name = "financial-encyclopedia"
except Exception as e:
    print(f"Error initializing Pinecone: {str(e)}")
    raise

class QueryRequest(BaseModel):
    query: str

class FinancialAdvisorBot:
    def __init__(self):
        self.qa_chain = None
        # List of Gemini models to try (in order of preference)
        self.gemini_models = [
            "gemini-pro", 
            "gemini-1.0-pro",  # Try alternative names
            "gemini-1.5-pro",
            "models/gemini-pro"
        ]
    
    async def initialize_qa_bot(self):
        """Initialize QA components with Pinecone"""
        try:
            # Initialize embeddings
            hugging_face_embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-V2', 
                model_kwargs={'device': 'cpu'}
            )

            # Initialize Pinecone vector store
            pinecone_index = pc.Index(index_name)
            vectorstore = PineconeVectorStore(
                index=pinecone_index, 
                embedding=hugging_face_embeddings,
                text_key='text'
            )
        
            # Initialize Groq LLM with Llama3 model
            groq_model = ChatGroq(
                model_name="llama3-8b-8192",
                groq_api_key=os.getenv('GROQ_API_KEY'),
                temperature=0.5,
                max_tokens=2000
            )

            qa_prompt = PromptTemplate(
                template="""
                You are a professional financial advisor providing expert, personalized financial guidance.
                Provide clear, precise, and actionable financial advice based on the context and question.

                Context: {context}
                Question: {question}

                Guidelines:
                - Offer specific, practical financial recommendations
                - Explain complex financial concepts clearly
                - Provide insights tailored to the user's query
                - Maintain professional and helpful tone
                - Suggest next steps or potential strategies
                - provide details only about things related to india
                - Use clear sentence structures and structured lists with dashes
                - Avoid bold text
                - Write in conversational, engaging style
                - Be concise and direct in your answers
                """,
                input_variables=['context', 'question']
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=groq_model,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': qa_prompt}
            )
            print("QA bot initialized successfully")
        except Exception as e:
            error_detail = f"Error initializing QA bot: {str(e)}\n{traceback.format_exc()}"
            print(error_detail)
            raise Exception(error_detail)
    
    async def try_gemini_synthesis(self, llama_answer, query):
        """Try to use Gemini for synthesis, with fallbacks for different model names"""
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not gemini_api_key:
            print("Gemini API key not found, returning Llama response only")
            return llama_answer
            
        genai.configure(api_key=gemini_api_key)
        
        # Try different model names
        for model_name in self.gemini_models:
            try:
                print(f"Trying Gemini model: {model_name}")
                model = genai.GenerativeModel(model_name)
                
                synthesis_prompt = f"""
                Synthesize a professional financial advisor response.

                Base Context: {llama_answer}
                Original Query: {query}

                Task:
                - Provide concise, professional financial recommendation
                - Directly address the specific financial question
                - Offer clear, actionable advice
                - Focus on factual, research-backed insights.
                - Maintain a neutral, third-person tone, similar to a financial news article.
                - Incorporate relevant statistics and expert opinions where necessary.
                - Avoid unnecessary explanations and conversational language.
                - Avoid giving conclusions and too many side headings. make it short.
                """
                
                synthesis_response = await model.generate_content_async(synthesis_prompt)
                print(f"Successfully used Gemini model: {model_name}")
                return synthesis_response.text.strip()
            except Exception as e:
                print(f"Error with Gemini model {model_name}: {str(e)}")
                continue
        
        # If all Gemini models fail, return the Llama response
        print("All Gemini models failed, returning Llama response")
        return llama_answer
        
    async def generate_response(self, query):
        """Generate response using Llama3 and optionally Gemini for synthesis"""
        try:
            if not self.qa_chain:
                raise ValueError("QA chain not initialized")
            
            # Get response from Llama3
            res = await self.qa_chain.ainvoke({"query": query})
            llama_answer = res["result"]
            
            # Try to use Gemini for synthesis, with fallback to just using Llama answer
            try:
                final_response = await self.try_gemini_synthesis(llama_answer, query)
                return final_response
            except Exception as e:
                print(f"Error with Gemini synthesis, using Llama response: {str(e)}")
                return llama_answer
                
        except Exception as e:
            error_detail = f"Response generation error: {str(e)}\n{traceback.format_exc()}"
            print(error_detail)
            raise Exception(error_detail)

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
    """Initialize the QA bot on startup"""
    try:
        await bot.initialize_qa_bot()
    except Exception as e:
        print(f"Startup error: {str(e)}")
        # We don't want to stop the app from starting, but the endpoints won't work
        # until the bot is properly initialized

@app.post("/financial-advice")
async def get_financial_advice(request: QueryRequest):
    """Endpoint to get financial advice"""
    try:
        if not bot.qa_chain:
            # Attempt to initialize again if not already done
            await bot.initialize_qa_bot()
            
        response = await bot.generate_response(request.query)
        return {"advice": response}
    except Exception as e:
        error_detail = f"Error processing request: {str(e)}"
        print(f"{error_detail}\n{traceback.format_exc()}")
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
        # Check if the bot is properly initialized
        if bot.qa_chain is None:
            return {"status": "degraded", "message": "QA chain not initialized"}
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

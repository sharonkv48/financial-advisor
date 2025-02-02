import os
import asyncio
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
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Pinecone Configuration
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "financial-encyclopedia"

class QueryRequest(BaseModel):
    query: str

class FinancialAdvisorBot:
    def __init__(self):
        self.qa_chain = None
    
    async def initialize_qa_bot(self):
        """Initialize QA components with Pinecone"""
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
    
    async def generate_response(self, query):
        """Generate response using Gemini API for final synthesis"""
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not gemini_api_key:
            raise HTTPException(status_code=500, detail="Gemini API key not found")
        
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        res = await self.qa_chain.acall(query)
        llama_answer = res["result"]
        
        try:
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
            return synthesis_response.text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Response generation error: {str(e)}")

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
    await bot.initialize_qa_bot()

@app.post("/financial-advice")
async def get_financial_advice(request: QueryRequest):
    """Endpoint to get financial advice"""
    try:
        response = await bot.generate_response(request.query)
        return {"advice": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

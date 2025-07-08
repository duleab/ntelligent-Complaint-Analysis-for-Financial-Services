import os
import sys
from pathlib import Path
import logging

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMGenerator:
    """Class for generating answers using a language model."""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.7, max_tokens=512):
        """
        Initialize the generator with the specified language model.
        
        Args:
            model_name (str): Name of the language model to use
            temperature (float): Temperature for generation (higher = more creative)
            max_tokens (int): Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.prompt_template = None
        
        # Initialize the generator
        self._initialize_generator()
    
    def _initialize_generator(self):
        """Initialize the language model and prompt template."""
        try:
            # Import necessary modules
            from langchain_community.llms import HuggingFaceHub
            from langchain.llms.fake import FakeListLLM
            from langchain.prompts import PromptTemplate
            
            # Initialize the language model
            logger.info(f"Initializing language model: {self.model_name}")
            
            # Use FakeListLLM for testing instead of HuggingFaceHub
            fake_responses = [
                "Based on the retrieved complaints, customers are primarily concerned with unauthorized charges and billing disputes related to credit cards.",
                "The complaints show that customers are unhappy with BNPL services due to hidden fees and unclear terms.",
                "According to the complaints, money transfer issues include delayed processing times and unexpected fees.",
                "The data indicates that savings account complaints focus on interest rate discrepancies and account maintenance fees.",
                "Personal loan complaints frequently mention high interest rates and unexpected fees."
            ]
            
            self.llm = FakeListLLM(responses=fake_responses)
            
            # Define the prompt template
            template = """
            You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints across different financial products.
            
            Use ONLY the following retrieved complaint excerpts to formulate your answer:
            
            {context}
            
            Question: {question}
            
            Instructions:
            1. Base your answer ONLY on the provided context.
            2. If the context doesn't contain enough information to answer the question, state that clearly.
            3. Be concise but comprehensive in your answer.
            4. Highlight patterns or trends if they are apparent in the context.
            5. Do not make up information that is not in the context.
            
            Answer: 
            """
            
            self.prompt_template = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            logger.info("Language model and prompt template initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing generator: {str(e)}")
            raise
    
    def generate(self, question, context_docs):
        """Generate an answer for the given question using the provided context documents.
        
        Args:
            question (str): The question to answer
            context_docs (list): List of context documents
            
        Returns:
            str: The generated answer
        """
        if not self.llm or not self.prompt_template:
            logger.error("Language model or prompt template not initialized")
            raise ValueError("Language model or prompt template not initialized")
        
        try:
            # Extract text from context documents
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Format the prompt
            prompt = self.prompt_template.format(context=context_text, question=question)
            
            # Generate the answer
            logger.info(f"Generating answer for question: {question}")
            answer = self.llm(prompt)
            
            return answer.strip()
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def get_llm(self):
        """Get the language model object.
        
        Returns:
            LLM: The language model object
        """
        if not self.llm:
            logger.error("Language model not initialized")
            raise ValueError("Language model not initialized")
        
        return self.llm
    
    def get_prompt_template(self):
        """Get the prompt template.
        
        Returns:
            PromptTemplate: The prompt template
        """
        if not self.prompt_template:
            logger.error("Prompt template not initialized")
            raise ValueError("Prompt template not initialized")
        
        return self.prompt_template
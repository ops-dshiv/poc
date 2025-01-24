"""
AWS Bedrock Educational Chatbot with Advanced Memory Management
Version: 3.0
"""

import os
import json
import re
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional
from langgraph.graph import StateGraph, START, END

# AWS Imports
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever

# Core Components
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

# State Management
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Configuration
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================================
# Configuration
# =========================================================================
class Config:
    BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID")
    AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
    EMBED_MODEL = "amazon.titan-embed-text-v1"
    CHAT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
    MEMORY_SEARCH_K = 5
    MAX_HISTORY = 7
    SESSION_TIMEOUT = 300  # 5 minutes

# =========================================================================
# AWS Service Clients
# =========================================================================
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=Config.AWS_REGION
)

# =========================================================================
# Core Components
# =========================================================================
class MemoryManager:
    """Advanced memory handling with vector storage"""
    
    def __init__(self):
        self.vector_store = InMemoryVectorStore(
            BedrockEmbeddings(
                client=bedrock_runtime,
                model_id=Config.EMBED_MODEL
            )
        )
        
    def store_memory(self, text: str, metadata: dict):
        """Store conversation memory with metadata"""
        doc = Document(
            page_content=text,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                **metadata
            }
        )
        self.vector_store.add_documents([doc])
        
    def recall_memories(self, query: str, filters: dict, k: int = 5) -> List[str]:
        """Retrieve relevant memories with metadata filtering"""
        def _filter(doc: Document) -> bool:
            return all(doc.metadata.get(k) == v for k, v in filters.items())
            
        docs = self.vector_store.similarity_search(query, k=k, filter=_filter)
        return [d.page_content for d in docs]

class IntentDetector:
    """Advanced intent classification with clarification handling"""
    
    PROMPT_TEMPLATE = """Analyze the query and conversation history:
    
    Current conversation:
    {history}
    
    Query: {query}
    
    Classify as:
    - "education": Specific academic questions
    - "greeting": Simple hello/hi
    - "memory": References past conversations
    - "out_of_scope": Non-educational topics
    
    Respond STRICTLY in JSON format:
    {{
        "intent": "string",
        "entities": {{
            "institution": "string",
            "program": "string",
            "action": "string"
        }},
        "needs_clarification": boolean,
        "clarification_prompt": "string",
        "confidence": float
    }}"""
    
    def detect(self, history: List[str], query: str) -> dict:
        """Classify intent with context awareness"""
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                history="\n".join(history[-Config.MAX_HISTORY:]),
                query=query
            )
            
            response = ChatBedrock(
                client=bedrock_runtime,
                model_id=Config.CHAT_MODEL
            ).invoke(prompt)
            
            return self._parse_response(response.content)
            
        except Exception as e:
            logger.error(f"Intent detection error: {str(e)}")
            return self._fallback_response()

    def _parse_response(self, response_text: str) -> dict:
        """Validate and parse LLM response"""
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("Invalid response format")
            
        result = json.loads(json_match.group())
        
        # Validate intent
        valid_intents = ["education", "greeting", "memory", "out_of_scope"]
        if result["intent"] not in valid_intents:
            raise ValueError(f"Invalid intent: {result['intent']}")
            
        return result
        
    def _fallback_response(self) -> dict:
        """Default error response"""
        return {
            "intent": "education",
            "entities": {},
            "needs_clarification": True,
            "clarification_prompt": "Could you please rephrase your question?",
            "confidence": 0.0
        }

class ResponseGenerator:
    """Context-aware response builder with fallback handling"""
    
    def __init__(self):
        self.memory = MemoryManager()
        
    def generate(self, query: str, context: dict) -> str:
        """Generate informed response using multiple sources"""
        try:
            # Check for missing information
            if not context.get("documents") and context["intent"] == "education":
                return self._handle_missing_knowledge(context)
                
            if context.get("needs_clarification"):
                return context["clarification_prompt"]
                
            return self._build_response(context)
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return self._error_response()

    def _build_response(self, context: dict) -> str:
        """Construct structured response"""
        prompt = f"""
        Generate a professional educational response using:
        1. User query: {context['query']}
        2. Conversation history: {context.get('history', [])}
        3. Knowledge documents: {context.get('documents', [])}
        4. Previous memories: {context.get('memories', [])}
        
        Guidelines:
        - Be concise and factual
        - Use bullet points for lists
        - Cite sources when available
        - Maintain academic tone
        """
        
        response = ChatBedrock(
            client=bedrock_runtime,
            model_id=Config.CHAT_MODEL
        ).invoke(prompt)
        
        self._store_conversation(context, response.content)
        return response.content
        
    def _store_conversation(self, context: dict, response: str):
        """Save conversation context"""
        self.memory.store_memory(
            text=f"Q: {context['query']}\nA: {response}",
            metadata={
                "session_id": context.get("session_id"),
                "intent": context["intent"]
            }
        )
        
    def _handle_missing_knowledge(self, context: dict) -> str:
        """Fallback for unavailable information"""
        return (
            f"I couldn't find official information about {context['entities'].get('institution', 'this')}. "
            "Suggest checking their official website or contacting admissions directly."
        )
        
    def _error_response(self) -> str:
        """Generic error message"""
        return "I'm having trouble answering that. Please try rephrasing your question."

# =========================================================================
# Conversation Workflow
# =========================================================================
class EducationAssistant:
    """Main conversation orchestrator"""
    
    def __init__(self):
        self.memory = MemoryManager()
        self.graph = self._build_workflow()


    # In the EducationAssistant class's _build_workflow method
    def _build_workflow(self):
        """Create state machine for conversation processing"""
        workflow = StateGraph(dict)
        
        # Define nodes
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("fetch_knowledge", self._fetch_knowledge)
        workflow.add_node("generate_response", self._generate_response)
        
        # Define edges - Updated initialization
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_based_on_intent,
            {
                "education": "fetch_knowledge",
                "default": "generate_response"
            }
        )
        workflow.add_edge("fetch_knowledge", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile(checkpointer=MemorySaver())
        
    # def _build_workflow(self):
    #     """Create state machine for conversation processing"""
    #     workflow = StateGraph(dict)
        
    #     # Define nodes
    #     workflow.add_node("retrieve_context", self._retrieve_context)
    #     workflow.add_node("classify_intent", self._classify_intent)
    #     workflow.add_node("fetch_knowledge", self._fetch_knowledge)
    #     workflow.add_node("generate_response", self._generate_response)
        
    #     # Define edges
    #     workflow.set_start_point("retrieve_context")
    #     workflow.add_edge("retrieve_context", "classify_intent")
    #     workflow.add_conditional_edges(
    #         "classify_intent",
    #         self._route_based_on_intent,
    #         {
    #             "education": "fetch_knowledge",
    #             "default": "generate_response"
    #         }
    #     )
    #     workflow.add_edge("fetch_knowledge", "generate_response")
    #     workflow.add_edge("generate_response", END)
        
    #     return workflow.compile(checkpointer=MemorySaver())
        
    def _retrieve_context(self, state: dict):
        """Gather conversation context"""
        state["memories"] = self.memory.recall_memories(
            query=state["query"],
            filters={"session_id": state.get("session_id")},
            k=Config.MEMORY_SEARCH_K
        )
        return state
        
    def _classify_intent(self, state: dict):
        """Determine user intent"""
        detector = IntentDetector()
        state.update(detector.detect(
            state.get("history", []),
            state["query"]
        ))
        return state
        
    def _fetch_knowledge(self, state: dict):
        """Retrieve from AWS Knowledge Base"""
        if not Config.BEDROCK_KB_ID:
            raise ValueError("Knowledge Base ID not configured")
            
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=Config.BEDROCK_KB_ID,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": 3
                }
            }
        )
        
        try:
            state["documents"] = [
                doc.page_content for doc in retriever.invoke(state["query"])
            ]
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {str(e)}")
            state["documents"] = []
            
        return state
        
    def _generate_response(self, state: dict):
        """Generate final response"""
        generator = ResponseGenerator()
        state["response"] = generator.generate(
            query=state["query"],
            context=state
        )
        return state
        
    def _route_based_on_intent(self, state: dict):
        """Determine processing path"""
        if state["intent"] == "education":
            return "fetch_knowledge"
        return "generate_response"

# =========================================================================
# Example Usage
# =========================================================================
if __name__ == "__main__":
    # Initialize chatbot
    assistant = EducationAssistant()
    session_id = str(uuid.uuid4())
    
    # Example conversation
    queries = [
        "Hi there!",
        "What's the admission process for Stanford CS?",
        "What about scholarship opportunities?",
        "How does this compare to MIT?",
        "What's the weather in California?"
    ]
    
    history = []
    for query in queries:
        print(f"User: {query}")
        
        # Process query
        result = assistant.graph.invoke({
            "query": query,
            "session_id": 1,
            "history": history
        })
        
        # Store history
        history.append(f"User: {query}")
        history.append(f"Assistant: {result['response']}")
        
        print(f"Bot: {result['response']}\n")

    # Test memory recall
    print("Testing memory recall...")
    memory_test = assistant.graph.invoke({
        "query": "What did we discuss about Stanford?",
        "session_id": 1,
        "history": history
    })
    print(f"Bot: {memory_test['response']}")
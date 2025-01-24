"""
AWS Bedrock Educational Chatbot - Final Stable Version
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END


# AWS Components
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever

# Core Components
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

# State Management
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
    AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
    EMBED_MODEL = "amazon.titan-embed-text-v2:0"
    CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
    MEMORY_SEARCH_K = 5
    MAX_HISTORY = 5

# AWS Client Initialization
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=Config.AWS_REGION
)

class MemoryManager:
    """Enhanced memory management with hybrid search"""
    
    def __init__(self):
        self.vector_store = InMemoryVectorStore(
            BedrockEmbeddings(
                client=bedrock_runtime,
                model_id=Config.EMBED_MODEL
            )
        )
        
    def store_memory(self, text: str, metadata: dict):
        doc = Document(
            page_content=text,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                **metadata
            }
        )
        self.vector_store.add_documents([doc])
        
    def recall_memories(self, query: str, filters: dict, k: int = 5) -> List[str]:
        try:
            def _filter(doc: Document) -> bool:
                return all(doc.metadata.get(k) == v for k, v in filters.items())
            
            # First try vector search
            docs = self.vector_store.similarity_search(query, k=k, filter=_filter)

            print("docs+++++++++++++++++++++++",docs)
            
            # Fallback to keyword search
            if not docs:
                keyword_match = [doc for doc in self.vector_store._collection 
                               if query.lower() in doc.page_content.lower()]
                docs = sorted(keyword_match, 
                            key=lambda x: x.metadata["timestamp"], 
                            reverse=True)[:k]
            
            return [d.page_content for d in docs]
        except Exception as e:
            logger.error(f"Memory error: {str(e)}")
            return []

class IntentDetector:
    """Robust intent classification with error handling"""
    
    PROMPT_TEMPLATE = """Analyze the query and classify intent. Respond in JSON format:
    {{
        "intent": "education|greeting|memory|out_of_scope",
        "entities": {{
            "institution": "string",
            "program": "string"
        }},
        "needs_clarification": boolean,
        "clarification_prompt": "string"
    }}
    
    Query: {query}
    History: {history}"""
    
    def detect(self, history: List[str], query: str) -> dict:
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                query=query,
                history="\n".join(history[-Config.MAX_HISTORY:])
            )
            
            response = ChatBedrock(
                client=bedrock_runtime,
                model_id=Config.CHAT_MODEL
            ).invoke(prompt)
            
            return self._parse_response(response.content)
        except Exception as e:
            logger.error(f"Intent error: {str(e)}")
            return {
                "intent": "education",
                "needs_clarification": True,
                "clarification_prompt": "Could you please rephrase your question?"
            }

    def _parse_response(self, response_text: str) -> dict:
        try:
            json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
            result = json.loads(json_str)
            return {
                "intent": result.get("intent", "education"),
                "entities": result.get("entities", {}),
                "needs_clarification": result.get("needs_clarification", False),
                "clarification_prompt": result.get("clarification_prompt", "")
            }
        except:
            return {
                "intent": "education",
                "needs_clarification": True,
                "clarification_prompt": "Could you please rephrase your question?"
            }

class ResponseGenerator:
    """Reliable response generation with fallback content"""
    
    def __init__(self):
        self.memory = MemoryManager()
        
    def generate(self, context: dict) -> str:
        try:
            if context.get("needs_clarification"):
                return context["clarification_prompt"]
                
            prompt = self._build_prompt(context)
            
            response = ChatBedrock(
                client=bedrock_runtime,
                model_id=Config.CHAT_MODEL
            ).invoke(prompt)
            
            self._store_conversation(context, response.content)
            return response.content
            
        except Exception as e:
            logger.error(f"Response error: {str(e)}")
            return "I encountered an error. Please try again later."

    def _build_prompt(self, context: dict) -> str:
        components = [
            "Generate a professional educational response using:",
            f"1. Query: {context['query']}",
            f"2. History: {context.get('history', [])}",
            f"3. Documents: {context.get('documents', [])}",
            f"4. Memories: {context.get('memories', [])}",
            "Guidelines: Be concise, factual, and cite sources when available."
        ]
        return "\n".join(components)
        
    def _store_conversation(self, context: dict, response: str):
        self.memory.store_memory(
            text=f"Q: {context['query']}\nA: {response}",
            metadata={
                "thread_id": context.get("thread_id"),
                "intent": context["intent"]
            }
        )

class EducationAssistant:
    """Complete conversation workflow"""
    
    def __init__(self):
        self.memory = MemoryManager()
        self.graph = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(dict)
        
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("fetch_knowledge", self._fetch_knowledge)
        workflow.add_node("generate_response", self._generate_response)
        
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {
                "education": "fetch_knowledge",
                "greeting": "generate_response",
                "memory": "generate_response",
                "out_of_scope": "generate_response",
                "default": "generate_response"
            }
        )
        workflow.add_edge("fetch_knowledge", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    def _retrieve_context(self, state: dict):
        state["memories"] = self.memory.recall_memories(
            query=state["query"],
            filters={"thread_id": state.get("thread_id")},
            k=Config.MEMORY_SEARCH_K
        )
        return state
    

    

    def _classify_intent(self, state: dict):
        detector = IntentDetector()
        state.update(detector.detect(
            state.get("history", []),
            state["query"]
        ))
        return state

    def _fetch_knowledge(self, state: dict):
        state["documents"] = []
        if Config.BEDROCK_KB_ID:
            try:
                retriever = AmazonKnowledgeBasesRetriever(
                    knowledge_base_id=Config.BEDROCK_KB_ID,
                    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}}
                )
                state["documents"] = [
                    doc.page_content for doc in retriever.invoke(state["query"])
                ]
            except Exception as e:
                logger.error(f"Knowledge error: {str(e)}")
        return state

    def _generate_response(self, state: dict):
        generator = ResponseGenerator()
        state["response"] = generator.generate(state)
        return state

    def _route_intent(self, state: dict):
        return state["intent"]

# Example Usage
if __name__ == "__main__":
    assistant = EducationAssistant()
    thread_id = "demo_thread6"
    
    queries = [
        "Hi there!",
        "What's the iit delhi CS admission process?",
        "What about eligibilt crieteria?",
        "What did we discussed about iit delhi",
    ]
    
    history = []
    for query in queries:
        print(f"\nUser: {query}")
        
        result = assistant.graph.invoke(
            {"query": query, "history": history, "thread_id": thread_id},
            {"configurable": {"thread_id": thread_id}}
        )
        
        response = result.get("response", "Error processing request")
        history.append(f"User: {query}")
        history.append(f"Assistant: {response}")
        print(f"Bot: {response}")
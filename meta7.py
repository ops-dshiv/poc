# # # # import os
# # # # import json
# # # # import re
# # # # import logging
# # # # from datetime import datetime
# # # # from typing import List, Dict

# # # # import boto3
# # # # from langchain.schema import Document
# # # # from langgraph.graph import END, START, StateGraph
# # # # from langgraph.checkpoint.memory import MemorySaver

# # # # # AWS Configuration
# # # # class Config:
# # # #     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
# # # #     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
# # # #     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
# # # #     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
# # # #     MEMORY_SEARCH_K = 5
# # # #     MAX_HISTORY = 5

# # # # # Initialize AWS Client
# # # # bedrock_runtime = boto3.client(
# # # #     service_name="bedrock-runtime",
# # # #     region_name=Config.AWS_REGION
# # # # )

# # # # # Setup logging
# # # # logging.basicConfig(level=logging.INFO)
# # # # logger = logging.getLogger(__name__)

# # # # # Dummy Vector Store for now (Replace with a real one)
# # # # class InMemoryVectorStore:
# # # #     def __init__(self, embeddings):
# # # #         self.store = []

# # # #     def add_documents(self, docs: List[Document]):
# # # #         self.store.extend(docs)

# # # #     def similarity_search(self, query: str, k: int = 5):
# # # #         return self.store[-k:]

# # # # # Memory Management
# # # # class MemoryManager:
# # # #     """Enhanced memory management with hybrid search"""

# # # #     def __init__(self):
# # # #         self.vector_store = InMemoryVectorStore(
# # # #             embeddings=None  # Placeholder for AWS Bedrock Embeddings
# # # #         )

# # # #     def store_memory(self, text: str, metadata: dict):
# # # #         doc = Document(
# # # #             page_content=text,
# # # #             metadata={
# # # #                 "timestamp": datetime.utcnow().isoformat(),
# # # #                 **metadata
# # # #             }
# # # #         )
# # # #         self.vector_store.add_documents([doc])

# # # #     def recall_memories(self, query: str, filters: dict, k: int = 5) -> List[str]:
# # # #         try:
# # # #             docs = self.vector_store.similarity_search(query, k=k)
# # # #             return [d.page_content for d in docs]
# # # #         except Exception as e:
# # # #             logger.error(f"Memory retrieval error: {str(e)}")
# # # #             return []

# # # # # Intent Detection
# # # # class IntentDetector:
# # # #     """Robust intent classification with error handling"""

# # # #     PROMPT_TEMPLATE = """Analyze the query and classify intent. Respond in JSON format:
# # # #     {{
# # # #         "intent": "education|greeting|memory|out_of_scope",
# # # #         "entities": {{
# # # #             "institution": "string",
# # # #             "program": "string"
# # # #         }},
# # # #         "needs_clarification": boolean,
# # # #         "clarification_prompt": "string"
# # # #     }}

# # # #     Query: {query}
# # # #     History: {history}"""

# # # #     def detect(self, history: List[str], query: str) -> dict:
# # # #         try:
# # # #             prompt = self.PROMPT_TEMPLATE.format(
# # # #                 query=query,
# # # #                 history="\n".join(history[-Config.MAX_HISTORY:])
# # # #             )
            
# # # #             # Placeholder response to simulate AWS Bedrock API
# # # #             response = {
# # # #                 "intent": "education" if "admission" in query.lower() else "out_of_scope",
# # # #                 "needs_clarification": "criteria" in query.lower(),
# # # #                 "clarification_prompt": "Are you asking about IIT Delhi's CS admission or another university?"
# # # #             }

# # # #             return response
# # # #         except Exception as e:
# # # #             logger.error(f"Intent error: {str(e)}")
# # # #             return {
# # # #                 "intent": "education",
# # # #                 "needs_clarification": True,
# # # #                 "clarification_prompt": "Could you please rephrase your question?"
# # # #             }

# # # # # Response Generation
# # # # class ResponseGenerator:
# # # #     """Reliable response generation with dynamic responses"""

# # # #     def __init__(self):
# # # #         self.memory = MemoryManager()

# # # #     def generate(self, context: dict) -> str:
# # # #         try:
# # # #             if context.get("needs_clarification"):
# # # #                 return context["clarification_prompt"]

# # # #             past_conversations = "\n".join(context.get("history", [])[-3:])
# # # #             relevant_memories = "\n".join(context.get("memories", []))
# # # #             retrieved_docs = "\n".join(context.get("documents", []))

# # # #             # Construct prompt dynamically based on memory
# # # #             prompt = f"""
# # # #             You are an AI education assistant.
# # # #             - User Query: {context['query']}
# # # #             - Past Conversation: {past_conversations}
# # # #             - Related Memories: {relevant_memories}
# # # #             - Retrieved Documents: {retrieved_docs}
# # # #             - Guidelines: Be concise, factual, and cite sources when available.
# # # #             """

# # # #             response = "This is a dynamically generated response."  # Placeholder response

# # # #             self._store_conversation(context, response)
# # # #             return response

# # # #         except Exception as e:
# # # #             logger.error(f"Response error: {str(e)}")
# # # #             return "I encountered an error. Please try again later."

# # # #     def _store_conversation(self, context: dict, response: str):
# # # #         self.memory.store_memory(
# # # #             text=f"Q: {context['query']}\nA: {response}",
# # # #             metadata={"thread_id": context.get("thread_id"), "intent": context["intent"]}
# # # #         )

# # # # # Conversation Workflow
# # # # class EducationAssistant:
# # # #     """Complete conversation workflow"""

# # # #     def __init__(self):
# # # #         self.memory = MemoryManager()
# # # #         self.graph = self._build_workflow()

# # # #     def _build_workflow(self):
# # # #         workflow = StateGraph(dict)

# # # #         workflow.add_node("retrieve_context", self._retrieve_context)
# # # #         workflow.add_node("classify_intent", self._classify_intent)
# # # #         workflow.add_node("fetch_knowledge", self._fetch_knowledge)
# # # #         workflow.add_node("generate_response", self._generate_response)

# # # #         workflow.add_edge(START, "retrieve_context")
# # # #         workflow.add_edge("retrieve_context", "classify_intent")
# # # #         workflow.add_conditional_edges(
# # # #             "classify_intent",
# # # #             self._route_intent,
# # # #             {
# # # #                 "education": "fetch_knowledge",
# # # #                 "greeting": "generate_response",
# # # #                 "memory": "generate_response",
# # # #                 "out_of_scope": "generate_response",
# # # #                 "default": "generate_response"
# # # #             }
# # # #         )
# # # #         workflow.add_edge("fetch_knowledge", "generate_response")
# # # #         workflow.add_edge("generate_response", END)

# # # #         return workflow.compile(checkpointer=MemorySaver())

# # # #     def _retrieve_context(self, state: dict):
# # # #         state["memories"] = self.memory.recall_memories(
# # # #             query=state["query"],
# # # #             filters={"thread_id": state.get("thread_id")},
# # # #             k=Config.MEMORY_SEARCH_K
# # # #         )
# # # #         return state

# # # #     def _classify_intent(self, state: dict):
# # # #         detector = IntentDetector()
# # # #         state.update(detector.detect(state.get("history", []), state["query"]))
# # # #         return state

# # # #     def _fetch_knowledge(self, state: dict):
# # # #         state["documents"] = []
# # # #         return state

# # # #     def _generate_response(self, state: dict):
# # # #         generator = ResponseGenerator()
# # # #         state["response"] = generator.generate(state)
# # # #         return state

# # # #     def _route_intent(self, state: dict):
# # # #         return state["intent"]


# # # import os
# # # import json
# # # import logging
# # # from datetime import datetime
# # # from typing import List, Dict

# # # import boto3
# # # from langchain.schema import Document
# # # from langgraph.graph import END, START, StateGraph
# # # from langgraph.checkpoint.memory import MemorySaver

# # # # AWS Configuration
# # # class Config:
# # #     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
# # #     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
# # #     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
# # #     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
# # #     MEMORY_SEARCH_K = 5
# # #     MAX_HISTORY = 5

# # # # Initialize AWS Client
# # # bedrock_runtime = boto3.client(
# # #     service_name="bedrock-runtime",
# # #     region_name=Config.AWS_REGION
# # # )

# # # # Setup logging
# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger(__name__)

# # # # Dummy Vector Store
# # # class InMemoryVectorStore:
# # #     def __init__(self):
# # #         self.store = []

# # #     def add_documents(self, docs: List[Document]):
# # #         self.store.extend(docs)

# # #     def similarity_search(self, query: str, k: int = 5):
# # #         return self.store[-k:]

# # # # Memory Manager
# # # class MemoryManager:
# # #     def __init__(self):
# # #         self.vector_store = InMemoryVectorStore()

# # #     def store_memory(self, text: str, metadata: dict):
# # #         doc = Document(page_content=text, metadata={"timestamp": datetime.utcnow().isoformat(), **metadata})
# # #         self.vector_store.add_documents([doc])

# # #     def recall_memories(self, query: str, filters: dict, k: int = 5) -> List[str]:
# # #         try:
# # #             docs = self.vector_store.similarity_search(query, k=k)
# # #             return [d.page_content for d in docs]
# # #         except Exception as e:
# # #             logger.error(f"Memory retrieval error: {str(e)}")
# # #             return []

# # # # Intent Detector
# # # class IntentDetector:
# # #     PROMPT_TEMPLATE = """Analyze the query and classify intent. Respond in JSON format:
# # #     {{
# # #         "intent": "education|greeting|memory|out_of_scope",
# # #         "entities": {{
# # #             "institution": "string",
# # #             "program": "string"
# # #         }},
# # #         "needs_clarification": boolean,
# # #         "clarification_prompt": "string"
# # #     }}

# # #     Query: {query}
# # #     History: {history}"""

# # #     def detect(self, history: List[str], query: str) -> dict:
# # #         try:
# # #             response = {
# # #                 "intent": "education" if "admission" in query.lower() else "out_of_scope",
# # #                 "needs_clarification": "criteria" in query.lower(),
# # #                 "clarification_prompt": "Are you asking about IIT Delhi's CS admission or another university?"
# # #             }
# # #             return response
# # #         except Exception as e:
# # #             logger.error(f"Intent error: {str(e)}")
# # #             return {"intent": "education", "needs_clarification": True, "clarification_prompt": "Please clarify your question."}

# # # # Response Generator
# # # class ResponseGenerator:
# # #     def __init__(self):
# # #         self.memory = MemoryManager()

# # #     def generate(self, context: dict) -> str:
# # #         try:
# # #             if context.get("needs_clarification"):
# # #                 return context["clarification_prompt"]

# # #             past_conversations = "\n".join(context.get("history", [])[-3:])
# # #             relevant_memories = "\n".join(context.get("memories", []))
# # #             retrieved_docs = "\n".join(context.get("documents", []))

# # #             response = f"Based on your query, past conversation: {past_conversations}, and retrieved data: {retrieved_docs}, here is the best response."

# # #             self._store_conversation(context, response)
# # #             return response

# # #         except Exception as e:
# # #             logger.error(f"Response error: {str(e)}")
# # #             return "I encountered an error. Please try again later."

# # #     def _store_conversation(self, context: dict, response: str):
# # #         self.memory.store_memory(
# # #             text=f"Q: {context['query']}\nA: {response}",
# # #             metadata={"thread_id": context.get("thread_id"), "intent": context["intent"]}
# # #         )

# # # # Main Chatbot Class
# # # class EducationAssistant:
# # #     def __init__(self):
# # #         self.memory = MemoryManager()
# # #         self.graph = self._build_workflow()

# # #     def _build_workflow(self):
# # #         workflow = StateGraph(dict)
# # #         workflow.add_node("retrieve_context", self._retrieve_context)
# # #         workflow.add_node("classify_intent", self._classify_intent)
# # #         workflow.add_node("fetch_knowledge", self._fetch_knowledge)
# # #         workflow.add_node("generate_response", self._generate_response)

# # #         workflow.add_edge(START, "retrieve_context")
# # #         workflow.add_edge("retrieve_context", "classify_intent")
# # #         workflow.add_conditional_edges(
# # #             "classify_intent",
# # #             self._route_intent,
# # #             {"education": "fetch_knowledge", "greeting": "generate_response", "memory": "generate_response", "out_of_scope": "generate_response", "default": "generate_response"}
# # #         )
# # #         workflow.add_edge("fetch_knowledge", "generate_response")
# # #         workflow.add_edge("generate_response", END)

# # #         return workflow.compile(checkpointer=MemorySaver())

# # #     def _retrieve_context(self, state: dict):
# # #         state["memories"] = self.memory.recall_memories(state["query"], {"thread_id": state.get("thread_id")}, k=Config.MEMORY_SEARCH_K)
# # #         return state

# # #     def _classify_intent(self, state: dict):
# # #         detector = IntentDetector()
# # #         state.update(detector.detect(state.get("history", []), state["query"]))
# # #         return state

# # #     def _fetch_knowledge(self, state: dict):
# # #         state["documents"] = []
# # #         return state

# # #     def _generate_response(self, state: dict):
# # #         generator = ResponseGenerator()
# # #         state["response"] = generator.generate(state)
# # #         return state

# # #     def _route_intent(self, state: dict):
# # #         return state["intent"]



# # import os
# # import re
# # import json
# # import logging
# # from dotenv import load_dotenv
# # from datetime import datetime
# # from typing import List, Dict

# # import boto3
# # from langchain.docstore.document import Document

# # # If you're using LangChain's InMemoryVectorStore and embeddings:
# # from langchain_core.vectorstores import InMemoryVectorStore

# # # from langchain.embeddings.base import Embeddings
# # # from langchain.embeddings import BedrockEmbeddings

# # import boto3
# # from langchain_aws import BedrockEmbeddings, ChatBedrock
# # from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever

# # # Depending on your environment, these classes might come from your custom modules:
# # # from some_package import StateGraph, MemorySaver, START, END, AmazonKnowledgeBasesRetriever, ChatBedrock
# # # Make sure they are appropriately imported or defined.

# # # For demonstration, placeholders:
# # class StateGraph:
# #     """Placeholder for a custom state graph implementation."""
# #     def __init__(self, initial_state_type):
# #         pass
# #     def add_node(self, name, func):
# #         pass
# #     def add_edge(self, start, end):
# #         pass
# #     def add_conditional_edges(self, node_name, condition_func, edge_map):
# #         pass
# #     def compile(self, checkpointer=None):
# #         return self
# #     def invoke(self, state, config):
# #         # Minimal placeholder for demonstration
# #         current_state = state
# #         # 1) retrieve_context
# #         current_state = self.nodes["retrieve_context"](current_state)
# #         # 2) classify_intent
# #         current_state = self.nodes["classify_intent"](current_state)
# #         # 3) route_intent
# #         next_step = condition_func(current_state)  # we assume we have a condition_func
# #         # fallback
# #         if next_step == "education":
# #             current_state = self.nodes["fetch_knowledge"](current_state)
# #         # 4) generate_response
# #         current_state = self.nodes["generate_response"](current_state)
# #         return current_state
        
# #     # For demonstration, a simple dictionary to store nodes
# #     nodes = {}
# #     def add_node(self, name, func):
# #         self.nodes[name] = func

# # class MemorySaver:
# #     """Placeholder for a memory checkpointer."""
# #     pass

# # START = "start"
# # END = "end"

# # class AmazonKnowledgeBasesRetriever:
# #     """Placeholder for an AmazonKnowledgeBasesRetriever implementation."""
# #     def __init__(self, knowledge_base_id, retrieval_config=None):
# #         pass
# #     def invoke(self, query):
# #         # Return empty list for demonstration
# #         return []

# # class ChatBedrock:
# #     """Placeholder for a ChatBedrock class that calls an LLM endpoint on Bedrock."""
# #     def __init__(self, client, model_id):
# #         self.client = client
# #         self.model_id = model_id
# #     def invoke(self, prompt: str):
# #         # Simulates calling an LLM. Replace this with actual API call to your service.
# #         class Resp:
# #             content = f"Simulated response to: {prompt}"
# #         return Resp()

# # # Load environment and set up logging
# # load_dotenv()
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # class Config:
# #     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
# #     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
# #     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
# #     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
# #     MEMORY_SEARCH_K = 5
# #     MAX_HISTORY = 5

# # # AWS Client Initialization
# # bedrock_runtime = boto3.client(
# #     service_name="bedrock-runtime",
# #     region_name=Config.AWS_REGION
# # )

# # class MemoryManager:
# #     """Enhanced memory management with hybrid search"""
    
# #     def __init__(self):
# #         # Keep a local Python list to store documents for fallback
# #         self.all_docs = []
# #         # Vector store initialization
# #         self.vector_store = InMemoryVectorStore(
# #             embedding_function=BedrockEmbeddings(
# #                 client=bedrock_runtime,
# #                 model_id=Config.EMBED_MODEL
# #             )
# #         )
        
# #     def store_memory(self, text: str, metadata: dict):
# #         doc = Document(
# #             page_content=text,
# #             metadata={
# #                 "timestamp": datetime.utcnow().isoformat(),
# #                 **metadata
# #             }
# #         )
# #         # Save in our local list
# #         self.all_docs.append(doc)
# #         # Also add to vector store
# #         self.vector_store.add_documents([doc])
        
# #     def recall_memories(self, query: str, filters: dict, k: int = 5) -> List[str]:
# #         try:
# #             def _filter(doc: Document) -> bool:
# #                 return all(doc.metadata.get(fk) == fv for fk, fv in filters.items())
            
# #             # First try vector search
# #             docs = self.vector_store.similarity_search(query, k=k, filter=_filter)
            
# #             print("docs+++++++++++++++++++++++", docs)
            
# #             # Fallback to keyword search if no relevant docs found
# #             if not docs:
# #                 keyword_match = [
# #                     doc for doc in self.all_docs
# #                     if query.lower() in doc.page_content.lower()
# #                 ]
# #                 docs = sorted(keyword_match, 
# #                               key=lambda x: x.metadata["timestamp"], 
# #                               reverse=True)[:k]
            
# #             return [d.page_content for d in docs]
# #         except Exception as e:
# #             logger.error(f"Memory error: {str(e)}")
# #             return []

# # class IntentDetector:
# #     """Robust intent classification with error handling"""
    
# #     PROMPT_TEMPLATE = """Analyze the query and classify intent. Respond in JSON format:
# #     {{
# #         "intent": "education|greeting|memory|out_of_scope",
# #         "entities": {{
# #             "institution": "string",
# #             "program": "string"
# #         }},
# #         "needs_clarification": boolean,
# #         "clarification_prompt": "string"
# #     }}
    
# #     Query: {query}
# #     History: {history}"""
    
# #     def detect(self, history: List[str], query: str) -> dict:
# #         try:
# #             prompt = self.PROMPT_TEMPLATE.format(
# #                 query=query,
# #                 history="\n".join(history[-Config.MAX_HISTORY:])
# #             )
            
# #             response = ChatBedrock(
# #                 client=bedrock_runtime,
# #                 model_id=Config.CHAT_MODEL
# #             ).invoke(prompt)
            
# #             return self._parse_response(response.content)
# #         except Exception as e:
# #             logger.error(f"Intent error: {str(e)}")
# #             return {
# #                 "intent": "education",
# #                 "needs_clarification": True,
# #                 "clarification_prompt": "Could you please rephrase your question?"
# #             }

# #     def _parse_response(self, response_text: str) -> dict:
# #         try:
# #             json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
# #             result = json.loads(json_str)
# #             return {
# #                 "intent": result.get("intent", "education"),
# #                 "entities": result.get("entities", {}),
# #                 "needs_clarification": result.get("needs_clarification", False),
# #                 "clarification_prompt": result.get("clarification_prompt", "")
# #             }
# #         except:
# #             return {
# #                 "intent": "education",
# #                 "needs_clarification": True,
# #                 "clarification_prompt": "Could you please rephrase your question?"
# #             }

# # class ResponseGenerator:
# #     """Reliable response generation with fallback content"""
    
# #     def __init__(self):
# #         self.memory = MemoryManager()
        
# #     def generate(self, context: dict) -> str:
# #         try:
# #             if context.get("needs_clarification"):
# #                 return context["clarification_prompt"]
                
# #             prompt = self._build_prompt(context)
            
# #             response = ChatBedrock(
# #                 client=bedrock_runtime,
# #                 model_id=Config.CHAT_MODEL
# #             ).invoke(prompt)
            
# #             # Store this conversation piece
# #             self._store_conversation(context, response.content)
# #             return response.content
            
# #         except Exception as e:
# #             logger.error(f"Response error: {str(e)}")
# #             return "I encountered an error. Please try again later."

# #     def _build_prompt(self, context: dict) -> str:
# #         components = [
# #             "Generate a professional educational response using:",
# #             f"1. Query: {context['query']}",
# #             f"2. History: {context.get('history', [])}",
# #             f"3. Documents: {context.get('documents', [])}",
# #             f"4. Memories: {context.get('memories', [])}",
# #             "Guidelines: Be concise, factual, and cite sources when available."
# #         ]
# #         return "\n".join(components)
        
# #     def _store_conversation(self, context: dict, response: str):
# #         self.memory.store_memory(
# #             text=f"Q: {context['query']}\nA: {response}",
# #             metadata={
# #                 "thread_id": context.get("thread_id"),
# #                 "intent": context["intent"]
# #             }
# #         )

# # class EducationAssistant:
# #     """Complete conversation workflow"""
    
# #     def __init__(self):
# #         self.memory = MemoryManager()
# #         self.graph = self._build_workflow()

# #     def _build_workflow(self):
# #         workflow = StateGraph(dict)
        
# #         workflow.add_node("retrieve_context", self._retrieve_context)
# #         workflow.add_node("classify_intent", self._classify_intent)
# #         workflow.add_node("fetch_knowledge", self._fetch_knowledge)
# #         workflow.add_node("generate_response", self._generate_response)
        
# #         workflow.add_edge(START, "retrieve_context")
# #         workflow.add_edge("retrieve_context", "classify_intent")
# #         workflow.add_conditional_edges(
# #             "classify_intent",
# #             self._route_intent,
# #             {
# #                 "education": "fetch_knowledge",
# #                 "greeting": "generate_response",
# #                 "memory": "generate_response",
# #                 "out_of_scope": "generate_response",
# #                 "default": "generate_response"
# #             }
# #         )
# #         workflow.add_edge("fetch_knowledge", "generate_response")
# #         workflow.add_edge("generate_response", END)
        
# #         return workflow.compile(checkpointer=MemorySaver())

# #     def _retrieve_context(self, state: dict):
# #         state["memories"] = self.memory.recall_memories(
# #             query=state["query"],
# #             filters={"thread_id": state.get("thread_id")},
# #             k=Config.MEMORY_SEARCH_K
# #         )
# #         return state

# #     def _classify_intent(self, state: dict):
# #         detector = IntentDetector()
# #         result = detector.detect(
# #             state.get("history", []),
# #             state["query"]
# #         )
# #         state.update(result)  # merges keys like "intent", "needs_clarification" into state
# #         return state

# #     def _fetch_knowledge(self, state: dict):
# #         state["documents"] = []
# #         if Config.BEDROCK_KB_ID:
# #             try:
# #                 retriever = AmazonKnowledgeBasesRetriever(
# #                     knowledge_base_id=Config.BEDROCK_KB_ID,
# #                     retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}}
# #                 )
# #                 docs = retriever.invoke(state["query"])
# #                 state["documents"] = [doc.page_content for doc in docs]
# #             except Exception as e:
# #                 logger.error(f"Knowledge error: {str(e)}")
# #         return state

# #     def _generate_response(self, state: dict):
# #         generator = ResponseGenerator()
# #         state["response"] = generator.generate(state)
# #         return state

# #     def _route_intent(self, state: dict):
# #         return state["intent"]

# # # Example Usage
# # if __name__ == "__main__":
# #     assistant = EducationAssistant()
# #     thread_id = "demo_thread6"
    
# #     queries = [
# #         "Hi there!",
# #         "What's the iit delhi CS admission process?",
# #         "What about eligibilt crieteria?",
# #         "What did we discussed about iit delhi",
# #         "what is this ",
# #         "What's the weather today",
# #         "What we discussed earlier",
# #         "What about new developments",
# #     ]
    
# #     history = []
# #     for query in queries:
# #         print(f"\nUser: {query}")
        
# #         result = assistant.graph.invoke(
# #             {"query": query, "history": history, "thread_id": thread_id},
# #             {"configurable": {"thread_id": thread_id}}
# #         )
        
# #         response = result.get("response", "Error processing request")
# #         history.append(f"User: {query}")
# #         history.append(f"Assistant: {response}")
# #         print(f"Bot: {response}")



# import os
# import re
# import json
# import logging
# from datetime import datetime
# from typing import List, Dict, Any

# import boto3
# from dotenv import load_dotenv
# from langchain.schema import Document
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# # from langchain.vectorstores import InMemoryVectorStore
# from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
# from langchain_aws import ChatBedrock, BedrockEmbeddings
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser

# import os
# import json
# import re
# import logging
# from datetime import datetime
# from typing import List, Dict
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END


# # AWS Components
# import boto3
# from langchain_aws import BedrockEmbeddings, ChatBedrock
# from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever

# # Core Components
# from langchain_core.documents import Document
# from langchain_core.vectorstores import InMemoryVectorStore

# # State Management
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver

# # # Environment setup
# # load_dotenv()
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # class Config:
# #     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
# #     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
# #     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
# #     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
# #     MEMORY_SEARCH_K = 5
# #     MAX_HISTORY = 5

# # # AWS Client Initialization
# # bedrock_runtime = boto3.client(
# #     service_name="bedrock-runtime",
# #     region_name=Config.AWS_REGION
# # )

# # class MemoryManager:
# #     """Enhanced memory management with hybrid search"""
    
# #     def __init__(self):
# #         self.vector_store = InMemoryVectorStore(
# #             BedrockEmbeddings(
# #                 client=bedrock_runtime,
# #                 model_id=Config.EMBED_MODEL
# #             )
# #         )
# #         self.all_docs = []  # Track all documents for keyword search
        
# #     def store_memory(self, text: str, metadata: dict):
# #         doc = Document(
# #             page_content=text,
# #             metadata={
# #                 "timestamp": datetime.utcnow().isoformat(),
# #                 **metadata
# #             }
# #         )
# #         self.vector_store.add_documents([doc])
# #         self.all_docs.append(doc)  # Maintain separate list for keyword search
        
# #     def recall_memories(self, query: str, filters: dict, k: int = 5) -> List[str]:
# #         try:
# #             def _filter(doc: Document) -> bool:
# #                 return all(doc.metadata.get(k) == v for k, v in filters.items())
            
# #             # First try vector search
# #             docs = self.vector_store.similarity_search(query, k=k, filter=_filter)
            
# #             # Fallback to keyword search using tracked documents
# #             if not docs:
# #                 keyword_match = [
# #                     doc for doc in self.all_docs
# #                     if query.lower() in doc.page_content.lower()
# #                     and _filter(doc)
# #                 ]
# #                 docs = sorted(keyword_match, 
# #                             key=lambda x: x.metadata["timestamp"], 
# #                             reverse=True)[:k]
            
# #             return [d.page_content for d in docs]
# #         except Exception as e:
# #             logger.error(f"Memory error: {str(e)}")
# #             return []

# # class IntentDetector:
# #     """Robust intent classification with error handling"""
    
# #     PROMPT_TEMPLATE = """Analyze the query and classify intent. Respond in JSON format:
# #     {{
# #         "intent": "education|greeting|memory|out_of_scope",
# #         "entities": {{
# #             "institution": "string",
# #             "program": "string"
# #         }},
# #         "needs_clarification": boolean,
# #         "clarification_prompt": "string"
# #     }}
    
# #     Query: {query}
# #     History: {history}"""
    
# #     def detect(self, history: List[str], query: str) -> dict:
# #         try:
# #             prompt = self.PROMPT_TEMPLATE.format(
# #                 query=query,
# #                 history="\n".join(history[-Config.MAX_HISTORY:])
# #             )
            
# #             response = ChatBedrock(
# #                 client=bedrock_runtime,
# #                 model_id=Config.CHAT_MODEL
# #             ).invoke(prompt)
            
# #             return self._parse_response(response.content)
# #         except Exception as e:
# #             logger.error(f"Intent error: {str(e)}")
# #             return {
# #                 "intent": "education",
# #                 "needs_clarification": True,
# #                 "clarification_prompt": "Could you please rephrase your question?"
# #             }

# #     def _parse_response(self, response_text: str) -> dict:
# #         try:
# #             json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
# #             result = json.loads(json_str)
# #             return {
# #                 "intent": result.get("intent", "education"),
# #                 "entities": result.get("entities", {}),
# #                 "needs_clarification": result.get("needs_clarification", False),
# #                 "clarification_prompt": result.get("clarification_prompt", "")
# #             }
# #         except:
# #             return {
# #                 "intent": "education",
# #                 "needs_clarification": True,
# #                 "clarification_prompt": "Could you please rephrase your question?"
# #             }

# # class ResponseGenerator:
# #     """Reliable response generation with fallback content"""
    
# #     def __init__(self):
# #         self.memory = MemoryManager()
        
# #     def generate(self, context: dict) -> str:
# #         try:
# #             if context.get("needs_clarification"):
# #                 return context["clarification_prompt"]
                
# #             prompt = self._build_prompt(context)
            
# #             response = ChatBedrock(
# #                 client=bedrock_runtime,
# #                 model_id=Config.CHAT_MODEL
# #             ).invoke(prompt)
            
# #             self._store_conversation(context, response.content)
# #             return response.content
            
# #         except Exception as e:
# #             logger.error(f"Response error: {str(e)}")
# #             return "I encountered an error. Please try again later."

# #     def _build_prompt(self, context: dict) -> str:
# #         components = [
# #             "Generate a professional educational response using:",
# #             f"1. Query: {context['query']}",
# #             f"2. History: {context.get('history', [])}",
# #             f"3. Documents: {context.get('documents', [])}",
# #             f"4. Memories: {context.get('memories', [])}",
# #             "Guidelines: Be concise, factual, and cite sources when available."
# #         ]
# #         return "\n".join(components)
        
# #     def _store_conversation(self, context: dict, response: str):
# #         self.memory.store_memory(
# #             text=f"Q: {context['query']}\nA: {response}",
# #             metadata={
# #                 "thread_id": context.get("thread_id"),
# #                 "intent": context["intent"]
# #             }
# #         )

# # class EducationAssistant:
# #     """Complete conversation workflow"""
    
# #     def __init__(self):
# #         self.memory = MemoryManager()
# #         self.conversation_chain = self._build_conversation_chain()

# #     def _build_conversation_chain(self):
# #         prompt = ChatPromptTemplate.from_messages([
# #             ("system", "You are a helpful educational assistant. Use the following context:"),
# #             MessagesPlaceholder(variable_name="history"),
# #             ("human", "{input}"),
# #             MessagesPlaceholder(variable_name="agent_scratchpad"),
# #         ])
        
# #         return RunnablePassthrough.assign(
# #             memories=lambda x: self.memory.recall_memories(
# #                 x["input"], {"thread_id": x.get("thread_id")}, Config.MEMORY_SEARCH_K
# #             )
# #         ) | prompt | ChatBedrock(
# #             client=bedrock_runtime,
# #             model_id=Config.CHAT_MODEL
# #         ) | StrOutputParser()

# #     def process_query(self, query: str, thread_id: str = "default") -> str:
# #         try:
# #             # Retrieve conversation context
# #             context = {
# #                 "input": query,
# #                 "thread_id": thread_id,
# #                 "history": self.memory.recall_memories(
# #                     query, {"thread_id": thread_id}, Config.MAX_HISTORY
# #                 )
# #             }

# #             # Generate response
# #             response = self.conversation_chain.invoke(context)
            
# #             # Store interaction
# #             self.memory.store_memory(
# #                 text=f"Q: {query}\nA: {response}",
# #                 metadata={"thread_id": thread_id, "type": "conversation"}
# #             )
            
# #             return response
            
# #         except Exception as e:
# #             logger.error(f"Processing error: {str(e)}")
# #             return "I encountered an error processing your request. Please try again."

# # # Example Usage
# # if __name__ == "__main__":
# #     assistant = EducationAssistant()
# #     thread_id = "demo_thread"
    
# #     queries = [
# #         "Hi there!",
# #         "What's the IIT Delhi CS admission process?",
# #         "What about eligibility criteria?",
# #         "What did we discuss about IIT Delhi?",
# #         "What's the weather today?",
# #         "What about new developments?",
# #     ]
    
# #     for query in queries:
# #         print(f"\nUser: {query}")
# #         response = assistant.process_query(query, thread_id)
# #         print(f"Bot: {response}")



# import json
# import re
# import logging
# from datetime import datetime
# from typing import List, Dict, Any

# # -------------------------------------------------------------------
# # 1. CONFIG & LOGGING
# # -------------------------------------------------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class Config:
#     EMBED_MODEL = "amazon.titan-embed-text-v2:0"  # Replace with your Bedrock Embeddings model ID
#     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"          # Replace with your Bedrock Chat model ID
#     BEDROCK_KB_ID = "BVWGHMKJOQ"            # Replace with your actual Knowledge Base ID
#     MAX_HISTORY = 3
#     MEMORY_SEARCH_K = 5  # how many memory items to retrieve


# # BEDROCK_SERVICE_NAME="bedrock-runtime"
# # BEDROCK_REGION_NAME="ap-south-1"
# # BEDROCK_MODEL_ID="amazon.titan-embed-text-v2:0"
# # CHAT_MODEL_ID="meta.llama3-70b-instruct-v1:0"
# # BEDROCK_EMBEDDING_MODEL_ID="amazon.titan-embed-text-v2:0"



# # Stub for an actual Bedrock runtime (replace with real session or client if you have it)
# bedrock_runtime = boto3.client(
#     service_name="bedrock-runtime",
#     region_name="ap-south-1"
# )

# # -------------------------------------------------------------------
# # 2. STUB CLASSES - Replace these with actual implementations if available
# # -------------------------------------------------------------------

# class Document:
#     """
#     A minimal Document class similar to langchain.docstore.document.Document.
#     Holds page_content and metadata.
#     """
#     def __init__(self, page_content: str, metadata: dict):
#         self.page_content = page_content
#         self.metadata = metadata

# class BedrockEmbeddings:
#     """
#     Stub for embeddings (replace with langchain.embeddings.BedrockEmbeddings).
#     """
#     def __init__(self, client, model_id: str):
#         self.client = client
#         self.model_id = model_id

#     def embed_query(self, text: str) -> List[float]:
#         # Return a dummy embedding vector
#         return [0.1]*768

# class InMemoryVectorStore:
#     """
#     Stub for langchain.vectorstores.InMemoryVectorStore.
#     """
#     def __init__(self, embeddings: BedrockEmbeddings):
#         self.embeddings = embeddings
#         self._collection = []

#     def add_documents(self, docs: List[Document]):
#         self._collection.extend(docs)

#     def similarity_search(self, query: str, k: int = 5, filter=None) -> List[Document]:
#         """
#         Naive approach: if 'filter' is provided, only keep docs that match.
#         We don't truly compute similarity—this is a placeholder.
#         """
#         results = []
#         for doc in self._collection:
#             if filter and not filter(doc):
#                 continue
#             results.append(doc)
#         return results[:k]

# class ChatBedrock:
#     """
#     Stub for langchain.chat_models.ChatBedrock
#     """
#     class Response:
#         def __init__(self, content: str):
#             self.content = content

#     def __init__(self, client, model_id: str):
#         self.client = client
#         self.model_id = model_id

#     def invoke(self, prompt: str):
#         """
#         Simulates a call to an LLM. Replace with real Amazon Bedrock call.
#         """
#         logger.info(f"[ChatBedrock] Invoking with prompt:\n{prompt}\n")
#         # Return a dummy JSON answer that says intent=education, no clarification needed
#         return self.Response(
#             content='{"intent":"education","entities":{},"needs_clarification":false,"clarification_prompt":""}'
#         )

# class AmazonKnowledgeBasesRetriever:
#     """
#     Stub for retrieving from Amazon Knowledge Base (Bedrock).
#     Replace with real call to your knowledge base if you have it.
#     """
#     def __init__(self, knowledge_base_id: str, retrieval_config: Dict[str, Any]):
#         self.kb_id = knowledge_base_id
#         self.config = retrieval_config

#     def invoke(self, query: str) -> List[Document]:
#         # Return a single dummy document for demonstration
#         dummy_meta = {
#             "location": {"s3Location": {"uri": "s3://my-bucket/dummy.pdf"}},
#             "score": 0.88,
#             "source_metadata": {"title": "Dummy doc from KB"}
#         }
#         dummy_doc = Document(
#             page_content="Some relevant content about IIT Delhi admission process.",
#             metadata=dummy_meta
#         )
#         return [dummy_doc]

# # -------------------------------------------------------------------
# # 3. STATE MACHINE STUB - Mimics a simplified version of langgraph
# # -------------------------------------------------------------------
# START = "START"
# END = "END"

# class StateGraph:
#     """
#     Minimal imitation of a graph-based conversation manager.
#     """
#     def __init__(self, state_type):
#         self.nodes = {}
#         self.edges = []
#         self.cond_edges = []
#         self.compiled = False

#     def add_node(self, name: str, func):
#         self.nodes[name] = func

#     def add_edge(self, from_node: str, to_node: str):
#         self.edges.append((from_node, to_node))

#     def add_conditional_edges(self, from_node: str, condition_function, edges_dict: dict):
#         """
#         edges_dict is like: {"education": "fetch_knowledge", "memory": "generate_response", "default": "generate_response"}
#         """
#         self.cond_edges.append((from_node, condition_function, edges_dict))

#     def compile(self, checkpointer=None):
#         self.compiled = True
#         return self

#     def invoke(self, initial_state: dict, config: dict = None) -> dict:
#         """
#         A simplistic BFS approach to run the graph. 
#         Will follow edges until END or no next node is found.
#         """
#         visited = set()
#         state = initial_state

#         # 1. Find the START edge
#         next_node = None
#         for edge in self.edges:
#             if edge[0] == START:
#                 next_node = edge[1]
#                 break

#         # 2. Traverse until no node or we find END
#         while next_node and next_node not in visited:
#             visited.add(next_node)

#             # Execute the node function
#             node_func = self.nodes[next_node]
#             state = node_func(state)  # pass state

#             # Check if there's a conditional edge from this node
#             cond_found = False
#             for (fnode, cond_func, mapping) in self.cond_edges:
#                 if fnode == next_node:
#                     # Evaluate the condition function
#                     route = cond_func(state)
#                     next_node = mapping.get(route, mapping.get("default", None))
#                     cond_found = True
#                     break

#             if cond_found:
#                 continue

#             # Otherwise, see if there's a direct edge from next_node to something else
#             next_edges = [e for e in self.edges if e[0] == next_node]
#             if next_edges:
#                 next_node = next_edges[0][1]  # take the first
#                 if next_node == END:
#                     break
#             else:
#                 # No more edges
#                 break

#         return state

# class MemorySaver:
#     pass

# # -------------------------------------------------------------------
# # 4. bedrock_retrieval & format_bedrock_retrieved_docs
# # -------------------------------------------------------------------
# def bedrock_retrieval(state: dict, config: dict) -> dict:
#     """
#     Retrieve documents from Bedrock based on state["query"].
#     """
#     bedrock_retriever = AmazonKnowledgeBasesRetriever(
#         knowledge_base_id=Config.BEDROCK_KB_ID,
#         retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}}
#     )

#     user_query = state["query"]
#     logger.info(f"[bedrock_retrieval] User query: {user_query}")

#     try:
#         retrieved_docs = bedrock_retriever.invoke(user_query)
#         logger.info(f"[bedrock_retrieval] Raw docs: {retrieved_docs}")
#     except Exception as e:
#         logger.error(f"[bedrock_retrieval] Error: {e}")
#         retrieved_docs = []

#     state["bedrock_retrieved_docs"] = retrieved_docs
#     return state

# def format_bedrock_retrieved_docs(state: dict) -> dict:
#     """
#     Converts the raw Document objects in `bedrock_retrieved_docs` into a structured dict list.
#     """
#     raw_docs = state.get("bedrock_retrieved_docs", [])
#     formatted_docs = []

#     for doc in raw_docs:
#         metadata = doc.metadata
#         page_content = doc.page_content
#         doc_struct = {
#             "s3_uri": metadata.get("location", {}).get("s3Location", {}).get("uri", "N/A"),
#             "score": metadata.get("score", 0),
#             "source_metadata": metadata.get("source_metadata", {}),
#             "content_preview": page_content[:200],  # just show 200 chars
#         }
#         formatted_docs.append(doc_struct)

#     state["bedrock_retrieved_docs"] = formatted_docs
#     return state

# # -------------------------------------------------------------------
# # 5. MEMORY MANAGER
# # -------------------------------------------------------------------
# class MemoryManager:
#     """Handles storing & retrieving conversation Q&A using a vector store + fallback."""

#     def __init__(self):
#         self.vector_store = InMemoryVectorStore(
#             BedrockEmbeddings(client=bedrock_runtime, model_id=Config.EMBED_MODEL)
#         )

#     def store_memory(self, text: str, metadata: dict):
#         try:
#             doc = Document(
#                 page_content=text,
#                 metadata={
#                     "timestamp": datetime.utcnow().isoformat(),
#                     **metadata
#                 }
#             )
#             self.vector_store.add_documents([doc])
#         except Exception as e:
#             logger.error(f"[MemoryManager] Error storing memory: {e}")

#     def recall_memories(self, query: str, filters: dict, k: int = 5) -> List[str]:
#         """
#         Retrieve relevant memories:
#           1) Vector similarity (with optional filters)
#           2) If empty, fallback to substring search
#         """
#         try:
#             def _filter(doc: Document) -> bool:
#                 return all(doc.metadata.get(fk) == fv for fk, fv in filters.items())

#             docs = self.vector_store.similarity_search(query, k=k, filter=_filter)
#             if not docs:
#                 # fallback substring match
#                 keyword_docs = [
#                     doc for doc in self.vector_store._collection
#                     if query.lower() in doc.page_content.lower()
#                 ]
#                 docs = sorted(keyword_docs, key=lambda x: x.metadata["timestamp"], reverse=True)[:k]

#             return [d.page_content for d in docs]
#         except Exception as e:
#             logger.error(f"[MemoryManager] Error recalling memory: {e}")
#             return []

# # -------------------------------------------------------------------
# # 6. INTENT DETECTOR
# # -------------------------------------------------------------------
# class IntentDetector:
#     """
#     Classifies user queries into (education|greeting|memory|out_of_scope)
#     with optional clarification.
#     """

#     PROMPT_TEMPLATE = """Analyze the query and classify intent in JSON format:
#     {{
#         "intent": "education|greeting|memory|out_of_scope",
#         "entities": {{
#             "institution": "string",
#             "program": "string"
#         }},
#         "needs_clarification": false,
#         "clarification_prompt": ""
#     }}

#     Query: {query}
#     History: {history}"""

#     def detect(self, history: List[str], query: str) -> dict:
#         try:
#             prompt = self.PROMPT_TEMPLATE.format(
#                 query=query,
#                 history="\n".join(history[-Config.MAX_HISTORY:])
#             )
#             response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
#             return self._parse_response(response.content)
#         except Exception as e:
#             logger.error(f"[IntentDetector] Error: {e}")
#             # fallback
#             return {
#                 "intent": "education",
#                 "needs_clarification": True,
#                 "clarification_prompt": "Could you please rephrase your question?"
#             }

#     def _parse_response(self, response_text: str) -> dict:
#         try:
#             json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
#             result = json.loads(json_str)
#             return {
#                 "intent": result.get("intent", "education"),
#                 "entities": result.get("entities", {}),
#                 "needs_clarification": result.get("needs_clarification", False),
#                 "clarification_prompt": result.get("clarification_prompt", "")
#             }
#         except Exception:
#             return {
#                 "intent": "education",
#                 "needs_clarification": True,
#                 "clarification_prompt": "Could you please rephrase your question?"
#             }

# # -------------------------------------------------------------------
# # 7. RESPONSE GENERATOR
# # -------------------------------------------------------------------
# class ResponseGenerator:
#     """
#     Generates a final answer based on the user query, conversation history, documents, and memory.
#     """

#     def __init__(self):
#         self.memory = MemoryManager()

#     def generate(self, context: dict) -> str:
#         try:
#             # If the system needs clarification, return that prompt right away
#             if context.get("needs_clarification"):
#                 return context.get("clarification_prompt", "Could you clarify?")

#             # Otherwise build a prompt and call ChatBedrock
#             prompt = self._build_prompt(context)
#             response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
#             # store conversation in memory
#             self._store_conversation(context, response.content)
#             return response.content
#         except Exception as e:
#             logger.error(f"[ResponseGenerator] Error: {e}")
#             return "Sorry, I encountered an error. Please try again."

#     def _build_prompt(self, context: dict) -> str:
#         # Build a prompt referencing the user query, history, docs, and memory
#         components = [
#             "Generate a helpful, concise educational response using:",
#             f"1) User Query: {context['query']}",
#             f"2) History: {context.get('history', [])}",
#             f"3) Documents: {context.get('documents', [])}",
#             f"4) Memories: {context.get('memories', [])}",
#             "Guidelines: Be concise, factual, and cite sources if available."
#         ]
#         return "\n".join(components)

#     def _store_conversation(self, context: dict, response: str):
#         # Save Q&A into memory
#         self.memory.store_memory(
#             text=f"Q: {context['query']}\nA: {response}",
#             metadata={"thread_id": context.get("thread_id"), "intent": context["intent"]}
#         )

# # -------------------------------------------------------------------
# # 8. EDUCATION ASSISTANT (STATE MACHINE ORCHESTRATION)
# # -------------------------------------------------------------------
# class EducationAssistant:
#     """
#     Combines memory, intent detection, knowledge retrieval, and response generation
#     with a strict gating approach for Bedrock calls.
#     """

#     def __init__(self):
#         self.memory = MemoryManager()
#         self.graph = self._build_workflow()

#     def _build_workflow(self):
#         workflow = StateGraph(dict)

#         workflow.add_node("retrieve_context", self._retrieve_context)
#         workflow.add_node("classify_intent", self._classify_intent)
#         workflow.add_node("fetch_knowledge", self._fetch_knowledge)
#         workflow.add_node("generate_response", self._generate_response)

#         workflow.add_edge(START, "retrieve_context")
#         workflow.add_edge("retrieve_context", "classify_intent")
#         workflow.add_conditional_edges(
#             "classify_intent",
#             self._route_intent,
#             {
#                 "education": "fetch_knowledge",
#                 "greeting": "generate_response",
#                 "memory": "generate_response",
#                 "out_of_scope": "generate_response",
#                 # default fallback if no matching intent found
#                 "default": "generate_response"
#             }
#         )
#         workflow.add_edge("fetch_knowledge", "generate_response")
#         workflow.add_edge("generate_response", END)

#         return workflow.compile(checkpointer=MemorySaver())

#     def _retrieve_context(self, state: dict) -> dict:
#         """
#         Called first. Retrieves relevant memories for the conversation so far.
#         """
#         thread_id = state.get("thread_id", None)
#         query = state["query"]
#         state["memories"] = self.memory.recall_memories(
#             query=query,
#             filters={"thread_id": thread_id},
#             k=Config.MEMORY_SEARCH_K
#         )
#         return state

#     def _classify_intent(self, state: dict) -> dict:
#         """
#         Next, uses LLM to decide if it's 'education', 'greeting', 'memory', or 'out_of_scope'.
#         """
#         detector = IntentDetector()
#         state.update(detector.detect(state.get("history", []), state["query"]))
#         return state

#     def _fetch_knowledge(self, state: dict) -> dict:
#         """
#         Strict gating:
#           - Only retrieve if intent == 'education'
#           - AND needs_clarification == False
#         Otherwise, skip retrieval.
#         """
#         state["documents"] = []

#         # Gate 1: Must be 'education'
#         if state.get("intent") != "education":
#             logger.info("[_fetch_knowledge] Skipping retrieval (intent != 'education').")
#             return state

#         # Gate 2: Must NOT need clarification
#         if state.get("needs_clarification", False):
#             logger.info("[_fetch_knowledge] Skipping retrieval (needs_clarification == True).")
#             return state

#         # If KB is not configured, skip
#         if not Config.BEDROCK_KB_ID:
#             logger.warning("[_fetch_knowledge] No BEDROCK_KB_ID configured, skipping.")
#             return state

#         try:
#             # 1) Retrieve raw docs
#             state = bedrock_retrieval(state, config={})
#             # 2) Format them
#             state = format_bedrock_retrieved_docs(state)
#             # 3) Map them to state["documents"] for final response generation
#             docs = state.get("bedrock_retrieved_docs", [])
#             state["documents"] = [doc["content_preview"] for doc in docs]
#         except Exception as e:
#             logger.error(f"[EducationAssistant._fetch_knowledge] Error: {e}")
#             state["documents"] = []

#         return state

#     def _generate_response(self, state: dict) -> dict:
#         """
#         Finally, build a response from the context, knowledge, memory, etc.
#         """
#         generator = ResponseGenerator()
#         state["response"] = generator.generate(state)
#         return state

#     def _route_intent(self, state: dict) -> str:
#         """
#         Called after classify_intent. Returns the user intent to decide next node.
#         """
#         return state.get("intent", "default")

# # -------------------------------------------------------------------
# # 9. DEMO USAGE
# # -------------------------------------------------------------------
# if __name__ == "__main__":
#     assistant = EducationAssistant()

#     history = []
#     queries = [
#         "Hi there!",
#         "What's the admission process for IIT Delhi CS?",
#         "Thanks. Also what's the weather like?",
#         "What else did we discuss about IIT Delhi?"
#     ]

#     thread_id = "demo_session"

#     for user_query in queries:
#         print(f"\nUSER: {user_query}")

#         # Build the input state
#         state_input = {
#             "query": user_query,
#             "history": history,
#             "thread_id": thread_id
#         }

#         # Invoke the state machine
#         result_state = assistant.graph.invoke(state_input)

#         # Fetch the response
#         response = result_state.get("response", "No response found.")
#         print(f"BOT: {response}")

#         # Update history
#         history.append(f"User: {user_query}")
#         history.append(f"Assistant: {response}")


# def format_bedrock_retrieved_docs(state: State) -> State:
#     """
#     Format the Bedrock retrieved documents into a structured format.

#     Args:
#         state (State): The current state containing `bedrock_retrieved_docs`.

#     Returns:
#         State: Updated state with structured documents.
#     """
#     # Retrieve raw documents from the state
#     retrieved_docs = state.get("bedrock_retrieved_docs", [])
#     formatted_docs = []

#     for doc in retrieved_docs:
#         # Extract metadata and content
#         metadata = doc.metadata
#         page_content = doc.page_content

#         # Parse relevant fields from metadata
#         s3_uri = metadata.get("location", {}).get("s3Location", {}).get("uri", "N/A")
#         source_metadata = metadata.get("source_metadata", {})
#         score = metadata.get("score", 0)

#         # Build a structured representation
#         formatted_doc = {
#             "s3_uri": s3_uri,
#             "score": score,
#             "source_metadata": source_metadata,
#             "content_preview": page_content[:200],  # Limit content preview to 200 chars
#         }
#         formatted_docs.append(formatted_doc)

#     # Add the formatted documents back to the state
#     state["bedrock_retrieved_docs"] = formatted_docs

#     return state



# def bedrock_retrieval(state: State, config: RunnableConfig) -> State:
#     """
#     Perform Bedrock retrieval, format the results, and update the state.

#     Args:
#         state (State): Current state of the conversation.
#         config (RunnableConfig): Configuration for Bedrock retrieval.

#     Returns:
#         State: Updated state with retrieved and formatted documents.
#     """
#     # Initialize Bedrock retriever
#     bedrock_retriever = AmazonKnowledgeBasesRetriever(
#         knowledge_base_id="BVWGHMKJOQ",
#         retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
#     )

#     # Get the user's query
#     user_query = state["messages"][-1].content
#     print("User query for Bedrock retrieval:", user_query)  # Debug the query

#     try:
#         # Perform retrieval
#         retrieved_docs = bedrock_retriever.invoke(user_query)
#         print("Raw retrieved documents:", retrieved_docs)  # Debug retrieved documents
#     except Exception as e:
#         print(f"Error during Bedrock retrieval: {e}")
#         retrieved_docs = []

#     if not retrieved_docs:
#         print("No documents retrieved from Bedrock.")
#         state["bedrock_retrieved_docs"] = []
#         return state

#     # Format the retrieved documents
#     formatted_docs = []
#     for doc in retrieved_docs:
#         metadata = doc.metadata
#         page_content = doc.page_content

#         # Build a structured document representation
#         formatted_doc = {
#             "s3_uri": metadata.get("location", {}).get("s3Location", {}).get("uri", "N/A"),
#             "score": metadata.get("score", 0),
#             "source_metadata": metadata.get("source_metadata", {}),
#             "content_preview": page_content,  # Truncate content preview
#         }
#         formatted_docs.append(formatted_doc)

#     # Update the state with formatted documents
#     state["bedrock_retrieved_docs"] = formatted_docs

#     print("bedrock_retrieved_docs",state["bedrock_retrieved_docs"])
#     return state




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
    BEDROCK_KB_ID = "BVWGHMKJOQ"

# AWS Client Initialization
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=Config.AWS_REGION
)


class SessionManager:
    """Manages session context in memory without Redis"""

    def __init__(self):
        self.sessions = {}  # Dictionary to store session data

    def store_session(self, thread_id, data):
        """Stores structured session context"""
        self.sessions[thread_id] = data

    def retrieve_session(self, thread_id):
        """Fetches session context"""
        return self.sessions.get(thread_id, {})

    def update_session(self, thread_id, key, value):
        """Updates specific session data"""
        session = self.retrieve_session(thread_id)
        session[key] = value
        self.store_session(thread_id, session)

# # Initialize session manager
session_manager = SessionManager()


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
        
    # def recall_memories(self, query: str, filters: dict, k: int = 5) -> List[str]:
    #     try:
    #         def _filter(doc: Document) -> bool:
    #             return all(doc.metadata.get(k) == v for k, v in filters.items())
            
    #         # First try vector search
    #         docs = self.vector_store.similarity_search(query, k=k, filter=_filter)

    #         print("docs+++++++++++++++++++++++",docs)
            
    #         # Fallback to keyword search
    #         if not docs:
    #             keyword_match = [doc for doc in self.vector_store._collection 
    #                            if query.lower() in doc.page_content.lower()]
    #             docs = sorted(keyword_match, 
    #                         key=lambda x: x.metadata["timestamp"], 
    #                         reverse=True)[:k]
            
    #         return [d.page_content for d in docs]
    #     except Exception as e:
    #         logger.error(f"Memory error: {str(e)}")
    #         return []


    def recall_memories(self, query: str, filters: dict, k: int = 5) -> List[str]:
        try:
            def _filter(doc: Document) -> bool:
                return all(doc.metadata.get(k) == v for k, v in filters.items())

            # 🔹 First try vector-based search (FAISS or Hybrid)
            docs = self.vector_store.similarity_search(query, k=k)

            # 🔹 Apply metadata filtering manually
            filtered_docs = [doc for doc in docs if _filter(doc)]

            # 🔹 Fallback: Keyword search if vector retrieval fails
            if not filtered_docs:
                all_docs = self.vector_store.get_all_documents()  # ✅ Use correct method to retrieve all docs
                keyword_match = [doc for doc in all_docs if query.lower() in doc.page_content.lower()]

                # Sort by timestamp for relevance
                filtered_docs = sorted(keyword_match, key=lambda x: x.metadata["timestamp"], reverse=True)[:k]

            return [d.page_content for d in filtered_docs]
        
        except Exception as e:
            logger.error(f"Memory retrieval error: {str(e)}")
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

    # def _retrieve_context(self, state: dict):
    #     state["memories"] = self.memory.recall_memories(
    #         query=state["query"],
    #         filters={"thread_id": state.get("thread_id")},
    #         k=Config.MEMORY_SEARCH_K
    #     )
    #     return state


    # def _retrieve_context(self, state: dict):
    #     """Retrieves past conversation context before processing a query"""
        
    #     thread_id = state["thread_id"]
        
    #     # Retrieve session data
    #     session = session_manager.retrieve_session(thread_id)
        
    #     # Store past memories & topics
    #     state["memories"] = self.memory.recall_memories(
    #         query=state["query"],
    #         filters={"thread_id": thread_id},
    #         k=Config.MEMORY_SEARCH_K
    #     )
        
    #     state["last_topic"] = session.get("last_topic", None)
    #     state["last_entities"] = session.get("last_entities", {})

    #     return state

    
    def _retrieve_context(self, state: dict):
        """Retrieves past conversation context before processing a query"""
        
        thread_id = state["thread_id"]
        
        # Retrieve session data
        session = session_manager.retrieve_session(thread_id)
        
        # Store past memories & last discussed topic
        state["memories"] = self.memory.recall_memories(
            query=state["query"],
            filters={"thread_id": thread_id},
            k=Config.MEMORY_SEARCH_K
        )

        state["last_topic"] = session.get("last_topic", None)
        state["last_entities"] = session.get("last_entities", {})

        # If user asks a vague follow-up (e.g., "What about eligibility?"), infer the topic
        if state["query"].strip().lower() in ["what about eligibility?", "what about fees?", "tell me more"]:
            if state["last_topic"]:
                state["query"] = f"{state['last_topic']} - {state['query']}"

        return state

    

    # def _classify_intent(self, state: dict):
    #     detector = IntentDetector()
    #     state.update(detector.detect(
    #         state.get("history", []),
    #         state["query"]
    #     ))
    #     return state


    def _classify_intent(self, state: dict):
        detector = IntentDetector()
        intent_data = detector.detect(state.get("history", []), state["query"])

        # Retrieve session data
        session = session_manager.retrieve_session(state["thread_id"])
        last_topic = session.get("last_topic", None)

        # If intent is different from last topic, clarify before switching
        if last_topic and last_topic != intent_data["intent"]:
            state["query"] = f"Are you asking about {intent_data['intent']} instead of {last_topic}?"

        # Store updated intent and topic
        session_manager.update_session(state["thread_id"], "last_topic", intent_data["intent"])
        session_manager.update_session(state["thread_id"], "last_entities", intent_data.get("entities", {}))

        state.update(intent_data)
        return state

    def _fetch_knowledge(self, state: dict):
        """Fetches knowledge dynamically from AWS Bedrock"""
        state["documents"] = []
        query = state["query"]
        
        if Config.BEDROCK_KB_ID:
            try:
                retriever = AmazonKnowledgeBasesRetriever(
                    knowledge_base_id=Config.BEDROCK_KB_ID,
                    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}}
                )
                docs = retriever.retrieve(query)  # 🔴 Fix incorrect `.invoke(query)`
                
                # Extract content properly
                state["documents"] = [doc.page_content for doc in docs]
            except Exception as e:
                logger.error(f"Knowledge retrieval error: {str(e)}")
        
        return state


    # def _fetch_knowledge(self, state: dict):
    #     """Fetches structured knowledge dynamically"""

    #     state["documents"] = []
    #     query = state["query"]
        
    #     # First, check if the knowledge base is enabled
    #     if Config.BEDROCK_KB_ID:
    #         try:
    #             retriever = AmazonKnowledgeBasesRetriever(
    #                 knowledge_base_id=Config.BEDROCK_KB_ID,
    #                 retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}}
    #             )

    #             docs = retriever.retrieve(query)
    #             retrieved_texts = [doc.page_content for doc in docs]

    #             # If incorrect data is retrieved, filter using keyword validation
    #             for text in retrieved_texts:
    #                 if "IIT Delhi" in text or "Computer Science" in text:
    #                     state["documents"].append(text)
                
    #             # If no relevant document is found, refine the query and try again
    #             if not state["documents"]:
    #                 refined_query = f"Provide official information on {query} related to IIT Delhi."
    #                 docs = retriever.invoke(refined_query)
    #                 retrieved_texts = [doc.page_content for doc in docs]
                    
    #                 for text in retrieved_texts:
    #                     if "IIT Delhi" in text or "Computer Science" in text:
    #                         state["documents"].append(text)

    #             # Final fallback - Use memory if knowledge base fails
    #             if not state["documents"]:
    #                 past_memories = self.memory.recall_memories(query, filters={"thread_id": state["thread_id"]}, k=2)
                    
    #                 if past_memories:
    #                     state["documents"] = past_memories
    #                 else:
    #                     # As a last resort, generate a response dynamically
    #                     fallback_prompt = f"Generate an informative response for: {query}"
    #                     fallback_response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(fallback_prompt)
    #                     state["documents"] = [fallback_response.content]

    #         except Exception as e:
    #             logger.error(f"Knowledge retrieval error: {str(e)}")

    #     return state

    # def _generate_response(self, state: dict):
    #     generator = ResponseGenerator()
    #     state["response"] = generator.generate(state)
    #     return state


    def _generate_response(self, state: dict):
        """Generates response and stores session context"""

        generator = ResponseGenerator()
        response_text = generator.generate(state)
        
        # Update session with latest context
        session_manager.update_session(
            state["thread_id"], "last_topic", state.get("intent", None)
        )
        session_manager.update_session(
            state["thread_id"], "last_entities", state.get("entities", {})
        )

        state["response"] = response_text
        return state


    def _route_intent(self, state: dict):
        return state["intent"]

# Example Usage
if __name__ == "__main__":
    assistant = EducationAssistant()
    thread_id = "demo_thread7"
    
    queries = [
        "Hi there!",
        "What's the iit delhi CS admission process?",
        "What about eligibilt crieteria?",
        "What did we discussed about iit delhi",
        "what is this ",
        "What's the weather today",
        "What we discussed earlier",
        "What about new developments",
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
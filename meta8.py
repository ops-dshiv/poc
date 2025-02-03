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
# from langchain_aws import ChatBedrock
# from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever

# # State Management
# from langgraph.checkpoint.memory import MemorySaver

# # Configuration
# load_dotenv()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# import os
# import json
# import re
# import logging
# from datetime import datetime
# from typing import List, Dict
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END
# import boto3
# from langchain_aws import BedrockEmbeddings, ChatBedrock
# from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
# from langchain_core.documents import Document
# from langchain_core.vectorstores import InMemoryVectorStore
# from langgraph.checkpoint.memory import MemorySaver

# # Load environment variables
# load_dotenv()

# # Configure Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configurations
# class Config:
#     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
#     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
#     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
#     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
#     MEMORY_SEARCH_K = 5
#     MAX_HISTORY = 5

# # AWS Bedrock Client Initialization
# bedrock_runtime = boto3.client(
#     service_name="bedrock-runtime",
#     region_name=Config.AWS_REGION
# )

# import os
# import json
# import re
# import logging
# from datetime import datetime
# from typing import List, Dict
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END
# import boto3
# from langchain_aws import BedrockEmbeddings, ChatBedrock
# from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
# from langchain_core.documents import Document
# from langchain_core.vectorstores import InMemoryVectorStore
# from langgraph.checkpoint.memory import MemorySaver

# # Load environment variables
# load_dotenv()

# # Configure Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configurations
# class Config:
#     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
#     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
#     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
#     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
#     MEMORY_SEARCH_K = 5
#     MAX_HISTORY = 5

# # AWS Bedrock Client Initialization
# bedrock_runtime = boto3.client(
#     service_name="bedrock-runtime",
#     region_name=Config.AWS_REGION
# )

# import os
# import json
# import re
# import logging
# from datetime import datetime
# from typing import List, Dict
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END
# import boto3
# from langchain_aws import BedrockEmbeddings, ChatBedrock
# from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
# from langchain_core.documents import Document
# from langchain_core.vectorstores import InMemoryVectorStore
# from langgraph.checkpoint.memory import MemorySaver

# # Load environment variables
# load_dotenv()

# # Configure Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configurations
# class Config:
#     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
#     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
#     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
#     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
#     MEMORY_SEARCH_K = 5
#     MAX_HISTORY = 5

# # AWS Bedrock Client Initialization
# bedrock_runtime = boto3.client(
#     service_name="bedrock-runtime",
#     region_name=Config.AWS_REGION
# )

# # Session Manager (Tracks User Context)
# class SessionManager:
#     def __init__(self):
#         self.sessions = {}

#     def store_session(self, thread_id, data):
#         self.sessions[thread_id] = data

#     def retrieve_session(self, thread_id):
#         return self.sessions.get(thread_id, {})

#     def update_session(self, thread_id, key, value):
#         session = self.retrieve_session(thread_id)
#         session[key] = value
#         self.store_session(thread_id, session)

# session_manager = SessionManager()

# # Memory Manager (Handles Long-Term Context)
# class MemoryManager:
#     def __init__(self):
#         self.vector_store = InMemoryVectorStore(
#             BedrockEmbeddings(client=bedrock_runtime, model_id=Config.EMBED_MODEL)
#         )

#     def store_memory(self, thread_id: str, query: str, response: str):
#         """Stores user conversations dynamically"""
#         doc = Document(
#             page_content=f"User: {query}\nBot: {response}",
#             metadata={"thread_id": thread_id, "timestamp": datetime.utcnow().isoformat()}
#         )
#         self.vector_store.add_documents([doc])

#     def recall_memories(self, thread_id: str, k: int = 5) -> List[str]:
#         """Retrieves past memory for a given user thread"""
#         try:
#             docs = self.vector_store.similarity_search(
#                 query="",  # Fetch all relevant past interactions
#                 k=k,
#                 filter=lambda d: d.metadata.get("thread_id") == thread_id
#             )
#             return [d.page_content for d in docs] if docs else []
#         except Exception as e:
#             logger.error(f"Memory error: {str(e)}")
#             return []

# # Knowledge Retriever (AWS Bedrock RAG Integration)
# class KnowledgeRetriever:
#     def retrieve_data(self, query):
#         """Fetches knowledge from Amazon Bedrock RAG"""
#         retrieval_request = {
#             'knowledgeBaseId': Config.BEDROCK_KB_ID,
#             'retrievalQuery': {'text': query},
#             'retrievalConfiguration': {'vectorSearchConfiguration': {'numberOfResults': 5, 'searchMethod': 'HYBRID'}}
#         }

#         try:
#             response = bedrock_runtime.invoke_model(
#                 modelId="BVWGHMKJOQ",
#                 body=json.dumps(retrieval_request)
#             )
#             result_data = json.loads(response['body'].read())
#             return [result['content'] for result in result_data.get('retrievalResults', [])]
#         except Exception as e:
#             logger.error(f"Knowledge retrieval error: {str(e)}")
#             return []

# # Intent Detector (Tracks User Intent & Context)
# class IntentDetector:
#     PROMPT_TEMPLATE = """Analyze the query and classify intent. Respond in JSON format:
#     {{
#         "intent": "education|greeting|memory|out_of_scope",
#         "entities": {{
#             "institution": "string",
#             "program": "string"
#         }},
#         "needs_clarification": boolean,
#         "clarification_prompt": "string"
#     }}
    
#     Query: {query}
#     History: {history}"""

#     def detect(self, history: List[str], query: str, thread_id: str) -> dict:
#         """Detects user intent dynamically"""
#         session = session_manager.retrieve_session(thread_id)
#         last_entities = session.get("last_entities", {})

#         response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(
#             self.PROMPT_TEMPLATE.format(query=query, history="\n".join(history[-Config.MAX_HISTORY:]))
#         )
#         result = self._parse_response(response.content)

#         if not result["entities"]:
#             result["entities"] = last_entities

#         session_manager.update_session(thread_id, "last_entities", result["entities"])
#         return result

#     def _parse_response(self, response_text: str) -> dict:
#         try:
#             json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
#             return json.loads(json_str)
#         except:
#             return {"intent": "education", "needs_clarification": True, "clarification_prompt": "Could you clarify your question?"}

# # Response Generator (Finalizes User Response)
# class ResponseGenerator:
#     def __init__(self):
#         self.memory = MemoryManager()

#     def generate(self, state: dict) -> str:
#         """Generates final response and updates memory"""
#         if state.get("needs_clarification"):
#             return state["clarification_prompt"]

#         prompt = f"Generate an educational response:\nQuery: {state['query']}\nHistory: {state.get('history', [])}\nDocuments: {state.get('documents', [])}\nMemories: {state.get('memories', [])}"
        
#         response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)

#         # Store Memory
#         self.memory.store_memory(
#             thread_id=state["thread_id"],
#             query=state["query"],
#             response=response.content
#         )
        
#         return response.content

# # Main Assistant Workflow
# class EducationAssistant:
#     def __init__(self):
#         self.memory = MemoryManager()
#         self.knowledge_retriever = KnowledgeRetriever()
#         self.graph = self._build_workflow()

#     def _build_workflow(self):
#         workflow = StateGraph(dict)

#         workflow.add_node("retrieve_context", self._retrieve_context)
#         workflow.add_node("classify_intent", self._classify_intent)
#         workflow.add_node("fetch_knowledge", self._fetch_knowledge)
#         workflow.add_node("generate_response", self._generate_response)

#         workflow.add_edge(START, "retrieve_context")
#         workflow.add_edge("retrieve_context", "classify_intent")
#         workflow.add_conditional_edges("classify_intent", self._route_intent, {
#             "education": "fetch_knowledge",
#             "greeting": "generate_response",
#             "memory": "generate_response",
#             "out_of_scope": "generate_response",
#             "default": "generate_response"
#         })
#         workflow.add_edge("fetch_knowledge", "generate_response")
#         workflow.add_edge("generate_response", END)

#         return workflow.compile(checkpointer=MemorySaver())

#     def _retrieve_context(self, state: dict):
#         """Retrieves past conversations and adds memory context"""
#         state["memories"] = self.memory.recall_memories(state["thread_id"], k=Config.MEMORY_SEARCH_K)
#         return state

#     def _classify_intent(self, state: dict):
#         detector = IntentDetector()
#         state.update(detector.detect(state.get("history", []), state["query"], state["thread_id"]))
#         return state

#     def _fetch_knowledge(self, state: dict):
#         state["documents"] = self.knowledge_retriever.retrieve_data(state["query"])
#         return state

#     def _generate_response(self, state: dict):
#         generator = ResponseGenerator()
#         state["response"] = generator.generate(state)
#         return state

#     def _route_intent(self, state: dict):
#         return state["intent"]


# # # Session Manager (Tracks User Context)
# # class SessionManager:
# #     def __init__(self):
# #         self.sessions = {}

# #     def store_session(self, thread_id, data):
# #         self.sessions[thread_id] = data

# #     def retrieve_session(self, thread_id):
# #         return self.sessions.get(thread_id, {})

# #     def update_session(self, thread_id, key, value):
# #         session = self.retrieve_session(thread_id)
# #         session[key] = value
# #         self.store_session(thread_id, session)

# # session_manager = SessionManager()

# # # Memory Manager (Handles Long-Term Context)
# # class MemoryManager:
# #     def __init__(self):
# #         self.vector_store = InMemoryVectorStore(
# #             BedrockEmbeddings(client=bedrock_runtime, model_id=Config.EMBED_MODEL)
# #         )

# #     def store_memory(self, text: str, metadata: dict):
# #         doc = Document(page_content=text, metadata={"timestamp": datetime.utcnow().isoformat(), **metadata})
# #         self.vector_store.add_documents([doc])

# #     def recall_memories(self, thread_id: str, query: str, k: int = 5) -> List[str]:
# #         """Retrieves relevant memory for a given user query."""
# #         try:
# #             docs = self.vector_store.similarity_search(query, k=k, filter=lambda d: d.metadata.get("thread_id") == thread_id)
# #             return [d.page_content for d in docs] if docs else []
# #         except Exception as e:
# #             logger.error(f"Memory error: {str(e)}")
# #             return []

# # # Knowledge Retriever (AWS Bedrock RAG Integration)
# # class KnowledgeRetriever:
# #     def retrieve_data(self, query):
# #         retrieval_request = {
# #             'knowledgeBaseId': Config.BEDROCK_KB_ID,
# #             'retrievalQuery': {'text': query},
# #             'retrievalConfiguration': {'vectorSearchConfiguration': {'numberOfResults': 5, 'searchMethod': 'HYBRID'}}
# #         }

# #         try:
# #             response = bedrock_runtime.invoke_model(
# #                 modelId="amazon.bedrock.rag",
# #                 body=json.dumps(retrieval_request)
# #             )
# #             result_data = json.loads(response['body'].read())
# #             return [result['content'] for result in result_data.get('retrievalResults', [])]
# #         except Exception as e:
# #             logger.error(f"Knowledge retrieval error: {str(e)}")
# #             return []

# # # Intent Detector (Tracks User Intent & Context)
# # class IntentDetector:
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

# #     def detect(self, history: List[str], query: str, thread_id: str) -> dict:
# #         session = session_manager.retrieve_session(thread_id)
# #         last_entities = session.get("last_entities", {})

# #         response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(
# #             self.PROMPT_TEMPLATE.format(query=query, history="\n".join(history[-Config.MAX_HISTORY:]))
# #         )
# #         result = self._parse_response(response.content)

# #         if not result["entities"]:
# #             result["entities"] = last_entities

# #         session_manager.update_session(thread_id, "last_entities", result["entities"])
# #         return result

# #     def _parse_response(self, response_text: str) -> dict:
# #         try:
# #             json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
# #             return json.loads(json_str)
# #         except:
# #             return {"intent": "education", "needs_clarification": True, "clarification_prompt": "Could you clarify your question?"}

# # # Response Generator (Finalizes User Response)
# # class ResponseGenerator:
# #     def __init__(self):
# #         self.memory = MemoryManager()

# #     def generate(self, context: dict) -> str:
# #         if context.get("needs_clarification"):
# #             return context["clarification_prompt"]

# #         prompt = f"Generate an educational response:\nQuery: {context['query']}\nHistory: {context.get('history', [])}\nDocuments: {context.get('documents', [])}\nMemories: {context.get('memories', [])}"
        
# #         response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
# #         self.memory.store_memory(
# #             text=f"Q: {context['query']}\nA: {response.content}",
# #             metadata={"thread_id": context["thread_id"], "intent": context["intent"]}
# #         )
        
# #         return response.content

# # # Main Assistant Workflow
# # class EducationAssistant:
# #     def __init__(self):
# #         self.memory = MemoryManager()
# #         self.knowledge_retriever = KnowledgeRetriever()
# #         self.graph = self._build_workflow()

# #     def _build_workflow(self):
# #         workflow = StateGraph(dict)

# #         workflow.add_node("retrieve_context", self._retrieve_context)
# #         workflow.add_node("classify_intent", self._classify_intent)
# #         workflow.add_node("fetch_knowledge", self._fetch_knowledge)
# #         workflow.add_node("generate_response", self._generate_response)

# #         workflow.add_edge(START, "retrieve_context")
# #         workflow.add_edge("retrieve_context", "classify_intent")
# #         workflow.add_conditional_edges("classify_intent", self._route_intent, {
# #             "education": "fetch_knowledge",
# #             "greeting": "generate_response",
# #             "memory": "generate_response",
# #             "out_of_scope": "generate_response",
# #             "default": "generate_response"
# #         })
# #         workflow.add_edge("fetch_knowledge", "generate_response")
# #         workflow.add_edge("generate_response", END)

# #         return workflow.compile(checkpointer=MemorySaver())

# #     def _retrieve_context(self, state: dict):
# #         state["memories"] = self.memory.recall_memories(state["thread_id"], state["query"], k=Config.MEMORY_SEARCH_K)
# #         return state

# #     def _classify_intent(self, state: dict):
# #         detector = IntentDetector()
# #         state.update(detector.detect(state.get("history", []), state["query"], state["thread_id"]))
# #         return state

# #     def _fetch_knowledge(self, state: dict):
# #         state["documents"] = self.knowledge_retriever.retrieve_data(state["query"])
# #         return state

# #     def _generate_response(self, state: dict):
# #         generator = ResponseGenerator()
# #         state["response"] = generator.generate(state)
# #         return state

# #     def _route_intent(self, state: dict):
# #         return state["intent"]


# # # Session Manager (Tracks User Context)
# # class SessionManager:
# #     def __init__(self):
# #         self.sessions = {}

# #     def store_session(self, thread_id, data):
# #         self.sessions[thread_id] = data

# #     def retrieve_session(self, thread_id):
# #         return self.sessions.get(thread_id, {})

# #     def update_session(self, thread_id, key, value):
# #         session = self.retrieve_session(thread_id)
# #         session[key] = value
# #         self.store_session(thread_id, session)

# # session_manager = SessionManager()

# # # Memory Manager (Handles Long-Term Context)
# # class MemoryManager:
# #     def __init__(self):
# #         self.vector_store = InMemoryVectorStore(
# #             BedrockEmbeddings(client=bedrock_runtime, model_id=Config.EMBED_MODEL)
# #         )

# #     def store_memory(self, text: str, metadata: dict):
# #         doc = Document(page_content=text, metadata={"timestamp": datetime.utcnow().isoformat(), **metadata})
# #         self.vector_store.add_documents([doc])

# #     def recall_memories(self, query: str, filters: dict, k: int = 5) -> List[str]:
# #         try:
# #             docs = self.vector_store.similarity_search(query, k=k, filter=lambda d: all(d.metadata.get(k) == v for k, v in filters.items()))
# #             return [d.page_content for d in docs] if docs else []
# #         except Exception as e:
# #             logger.error(f"Memory error: {str(e)}")
# #             return []

# # # Knowledge Retriever (AWS Bedrock RAG Integration)
# # class KnowledgeRetriever:
# #     def retrieve_data(self, query):
# #         retrieval_request = {
# #             'retrievalQuery': {'text': query},
# #             'knowledgeBaseId': Config.BEDROCK_KB_ID,
# #             'retrievalConfiguration': {'vectorSearchConfiguration': {'numberOfResults': 5, 'searchMethod': 'HYBRID'}}
# #         }
        
# #         try:
# #             response = bedrock_runtime.retrieve(**retrieval_request)
# #             return [result['content'] for result in response['retrievalResults']]
# #         except Exception as e:
# #             logger.error(f"Knowledge retrieval error: {str(e)}")
# #             return []

# # # Intent Detector (Tracks User Intent & Context)
# # class IntentDetector:
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

# #     def detect(self, history: List[str], query: str, thread_id: str) -> dict:
# #         session = session_manager.retrieve_session(thread_id)
# #         last_entities = session.get("last_entities", {})

# #         response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(
# #             self.PROMPT_TEMPLATE.format(query=query, history="\n".join(history[-Config.MAX_HISTORY:]))
# #         )
# #         result = self._parse_response(response.content)

# #         if not result["entities"]:
# #             result["entities"] = last_entities

# #         session_manager.update_session(thread_id, "last_entities", result["entities"])
# #         return result

# #     def _parse_response(self, response_text: str) -> dict:
# #         try:
# #             json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
# #             return json.loads(json_str)
# #         except:
# #             return {"intent": "education", "needs_clarification": True, "clarification_prompt": "Could you clarify your question?"}

# # # Response Generator (Finalizes User Response)
# # class ResponseGenerator:
# #     def __init__(self):
# #         self.memory = MemoryManager()

# #     def generate(self, context: dict) -> str:
# #         try:
# #             if context.get("needs_clarification"):
# #                 return context["clarification_prompt"]

# #             prompt = f"Generate an educational response:\nQuery: {context['query']}\nHistory: {context.get('history', [])}\nDocuments: {context.get('documents', [])}\nMemories: {context.get('memories', [])}"
            
# #             response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
# #             self.memory.store_memory(text=f"Q: {context['query']}\nA: {response.content}", metadata={"thread_id": context["thread_id"], "intent": context["intent"]})
            
# #             return response.content
# #         except Exception as e:
# #             logger.error(f"Response error: {str(e)}")
# #             return "I encountered an error. Please try again later."

# # # Main Assistant Workflow
# # class EducationAssistant:
# #     def __init__(self):
# #         self.memory = MemoryManager()
# #         self.knowledge_retriever = KnowledgeRetriever()
# #         self.graph = self._build_workflow()

# #     def _build_workflow(self):
# #         workflow = StateGraph(dict)

# #         workflow.add_node("retrieve_context", self._retrieve_context)
# #         workflow.add_node("classify_intent", self._classify_intent)
# #         workflow.add_node("fetch_knowledge", self._fetch_knowledge)
# #         workflow.add_node("generate_response", self._generate_response)

# #         workflow.add_edge(START, "retrieve_context")
# #         workflow.add_edge("retrieve_context", "classify_intent")
# #         workflow.add_conditional_edges("classify_intent", self._route_intent, {
# #             "education": "fetch_knowledge",
# #             "greeting": "generate_response",
# #             "memory": "generate_response",
# #             "out_of_scope": "generate_response",
# #             "default": "generate_response"
# #         })
# #         workflow.add_edge("fetch_knowledge", "generate_response")
# #         workflow.add_edge("generate_response", END)

# #         return workflow.compile(checkpointer=MemorySaver())

# #     def _retrieve_context(self, state: dict):
# #         state["memories"] = self.memory.recall_memories(state["query"], {"thread_id": state["thread_id"]}, k=Config.MEMORY_SEARCH_K)
# #         return state

# #     def _classify_intent(self, state: dict):
# #         detector = IntentDetector()
# #         state.update(detector.detect(state.get("history", []), state["query"], state["thread_id"]))
# #         return state

# #     def _fetch_knowledge(self, state: dict):
# #         state["documents"] = self.knowledge_retriever.retrieve_data(state["query"])
# #         return state

# #     def _generate_response(self, state: dict):
# #         generator = ResponseGenerator()
# #         state["response"] = generator.generate(state)
# #         return state

# #     def _route_intent(self, state: dict):
# #         return state["intent"]


# # class Config:
# #     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
# #     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
# #     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
# #     MEMORY_SEARCH_K = 5
# #     MAX_HISTORY = 5

# # # AWS Client Initialization
# # bedrock_runtime = boto3.client(
# #     service_name="bedrock-runtime",
# #     region_name=Config.AWS_REGION
# # )

# # class SessionManager:
# #     """Manages session context without Redis"""

# #     def __init__(self):
# #         self.sessions = {}

# #     def store_session(self, thread_id, data):
# #         self.sessions[thread_id] = data

# #     def retrieve_session(self, thread_id):
# #         return self.sessions.get(thread_id, {})

# #     def update_session(self, thread_id, key, value):
# #         session = self.retrieve_session(thread_id)
# #         session[key] = value
# #         self.store_session(thread_id, session)

# # # Initialize session manager
# # session_manager = SessionManager()

# # class KnowledgeBaseRetriever:
# #     """Retrieves relevant knowledge from Amazon Bedrock Knowledge Base"""

# #     def retrieve_data(self, query: str):
# #         """Fetch relevant documents from Amazon Bedrock KB"""
# #         if not Config.BEDROCK_KB_ID:
# #             logger.warning("Knowledge Base ID not set. Skipping retrieval.")
# #             return []

# #         try:
# #             retrieval_request = {
# #                 'retrievalQuery': {'text': query},
# #                 'knowledgeBaseId': Config.BEDROCK_KB_ID,
# #                 'retrievalConfiguration': {
# #                     'vectorSearchConfiguration': {
# #                         'numberOfResults': 5,
# #                         'searchMethod': 'HYBRID'
# #                     }
# #                 }
# #             }
# #             response = bedrock_runtime.retrieve(**retrieval_request)
# #             return [{"content": result['content'], "score": result['score']}
# #                     for result in response.get('retrievalResults', [])]

# #         except Exception as e:
# #             logger.error(f"Knowledge retrieval error: {str(e)}")
# #             return []

# # class IntentDetector:
# #     """Intent classification with entity tracking"""

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

# #     def detect(self, history, query, thread_id):
# #         session = session_manager.retrieve_session(thread_id)
# #         last_entities = session.get("last_entities", {})

# #         response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(
# #             self.PROMPT_TEMPLATE.format(query=query, history="\n".join(history[-Config.MAX_HISTORY:]))
# #         )
# #         result = self._parse_response(response.content)

# #         if not result["entities"]:
# #             result["entities"] = last_entities

# #         session_manager.update_session(thread_id, "last_entities", result["entities"])

# #         return result

# #     def _parse_response(self, response_text: str) -> dict:
# #         try:
# #             json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
# #             return json.loads(json_str)
# #         except:
# #             return {
# #                 "intent": "education",
# #                 "needs_clarification": True,
# #                 "clarification_prompt": "Could you please rephrase your question?"
# #             }

# # class ResponseGenerator:
# #     """Generates responses using Amazon Bedrock LLM"""

# #     def generate(self, context: dict) -> str:
# #         try:
# #             if context.get("needs_clarification"):
# #                 return context["clarification_prompt"]

# #             prompt = self._build_prompt(context)
# #             response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
# #             return response.content
# #         except Exception as e:
# #             logger.error(f"Response error: {str(e)}")
# #             return "I encountered an error. Please try again later."

# #     def _build_prompt(self, context: dict) -> str:
# #         return f"""Generate a response for:
# #         Query: {context['query']}
# #         History: {context.get('history', [])}
# #         Documents: {context.get('documents', [])}
# #         Guidelines: Be concise, factual, and cite sources when available."""

# # class EducationAssistant:
# #     """Complete conversation workflow"""

# #     def __init__(self):
# #         self.knowledge_retriever = KnowledgeBaseRetriever()
# #         self.graph = self._build_workflow()

# #     def _build_workflow(self):
# #         """Maintains original workflow with RAG integration"""
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
# #         """Retrieves past context before processing a query"""
# #         thread_id = state["thread_id"]
# #         session = session_manager.retrieve_session(thread_id)

# #         state["last_topic"] = session.get("last_topic", None)
# #         state["last_entities"] = session.get("last_entities", {})

# #         return state

# #     def _classify_intent(self, state: dict):
# #         detector = IntentDetector()
# #         intent_data = detector.detect(state.get("history", []), state["query"], state["thread_id"])
# #         session_manager.update_session(state["thread_id"], "last_topic", intent_data["intent"])
# #         state.update(intent_data)
# #         return state

# #     def _fetch_knowledge(self, state: dict):
# #         """Fetches structured knowledge dynamically"""

# #         state["documents"] = self.knowledge_retriever.retrieve_data(state["query"])
        
# #         # If no relevant results, attempt refined search
# #         if not state["documents"]:
# #             refined_query = f"Provide official information on {state['query']}"
# #             state["documents"] = self.knowledge_retriever.retrieve_data(refined_query)

# #         return state

# #     def _generate_response(self, state: dict):
# #         """Generates response using Amazon Bedrock"""

# #         generator = ResponseGenerator()
# #         state["response"] = generator.generate(state)

# #         return state

# #     def _route_intent(self, state: dict):
# #         return state["intent"]

# # Example Usage
# if __name__ == "__main__":
#     assistant = EducationAssistant()
#     thread_id = "demo_thread7"

#     queries = [
#         "Tell me about IIT Delhi's admission process",
#         "what about eligibility criteria"
#         "What courses does IIT Delhi offer?",
#         "What are the eligibility criteria for IIT Delhi's CS program?",
#         "Tell me about MIT's AI research.",
#         "what we discussed earlier"
#     ]

#     history = []
#     for query in queries:
#         print(f"\nUser: {query}")
#         result = assistant.graph.invoke(
#             {"query": query, "history": history, "thread_id": thread_id},
#             {"configurable": {"thread_id": thread_id}}
#         )
#         print(f"Bot: {result.get('response', 'Error processing request')}")

import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "BVWGHMKJOQ")  # Overriding with known ID
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

###############################################################################
# SessionManager
###############################################################################
class SessionManager:
    def __init__(self):
        self.sessions = {}

    def store_session(self, thread_id, data):
        self.sessions[thread_id] = data

    def retrieve_session(self, thread_id):
        return self.sessions.get(thread_id, {})

    def update_session(self, thread_id, key, value):
        session = self.retrieve_session(thread_id)
        session[key] = value
        self.store_session(thread_id, session)

session_manager = SessionManager()

###############################################################################
# MemoryManager
###############################################################################
class MemoryManager:
    """Enhanced memory management with hybrid search."""
    
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
        """
        Retrieves past memories relevant to `query`.
        1. Vector-based search, then filter by metadata.
        2. If none found, fallback to substring search on the same docs.
        """
        try:
            # Vector search
            docs = self.vector_store.similarity_search(query, k=k)

            # Filter by metadata
            def _filter(doc: Document) -> bool:
                return all(doc.metadata.get(fk) == fv for fk, fv in filters.items())
            
            filtered_docs = [doc for doc in docs if _filter(doc)]

            # If none, fallback to substring search on the same docs
            if not filtered_docs:
                substring_matches = [
                    doc for doc in docs if query.lower() in doc.page_content.lower()
                ]
                filtered_docs = substring_matches or []

            return [d.page_content for d in filtered_docs]
        except Exception as e:
            logger.error(f"Memory retrieval error: {str(e)}")
            return []

###############################################################################
# IntentDetector
###############################################################################
class IntentDetector:
    """Robust intent classification with error handling."""
    
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

###############################################################################
# ResponseGenerator
###############################################################################
class ResponseGenerator:
    """Reliable response generation with fallback content."""
    
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

###############################################################################
# KnowledgeFetcher
###############################################################################
class KnowledgeFetcher:
    """Uses AmazonKnowledgeBasesRetriever with `.invoke(...)` method instead of `.retrieve(...)`."""
    # def fetch(self, query: str) -> List[str]:
    #     if not Config.BEDROCK_KB_ID:
    #         logger.warning("No BEDROCK_KB_ID set. Returning empty docs.")
    #         return []

    #     try:
    #         retriever = AmazonKnowledgeBasesRetriever(
    #             knowledge_base_id=Config.BEDROCK_KB_ID,
    #             retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}}
    #         )
    #         # Correct method: .invoke(query), not .retrieve(query)
    #         docs = retriever.invoke(query)
    #         return [d.page_content for d in docs]
    #     except Exception as e:
    #         logger.error(f"Knowledge retrieval error: {str(e)}")
    #         return []

    def fetch(self, query: str) -> List[str]:
        """Fetch docs from knowledge base using .invoke(...) method."""
        if not Config.BEDROCK_KB_ID:
            return []

        try:
            retriever = AmazonKnowledgeBasesRetriever(
                knowledge_base_id=Config.BEDROCK_KB_ID,
                retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}}
            )
            docs = retriever.invoke(query)

            # Example optional filter: if user says "IIT Delhi", remove docs referencing "IIIT Delhi"
            query_lower = query.lower()
            if "iit delhi" in query_lower and "iiit delhi" not in query_lower:
                # Keep only docs referencing "IIT Delhi"
                filtered = []
                for d in docs:
                    content_lower = d.page_content.lower()
                    if "iit delhi" in content_lower and "iiit delhi" not in content_lower:
                        filtered.append(d)
                docs = filtered

            return [d.page_content for d in docs]
        except Exception as e:
            logger.error(f"Knowledge retrieval error: {str(e)}")
            return []

    # def fetch(self, user_query: str) -> List[str]:
    #     try:
    #         docs = self.retriever.invoke(user_query)

    #         # If user mentions 'IIT Delhi', we keep only docs referencing 'IIT Delhi'.
    #         # If user mentions 'IIIT Delhi', keep only docs referencing 'IIIT Delhi'.
    #         # Otherwise, keep all docs.
    #         filtered = []
    #         query_lower = user_query.lower()

    #         for doc in docs:
    #             content_lower = doc.page_content.lower()
                
    #             # Check if user specifically said "iit delhi" (or synonyms)
    #             if "iit delhi" in query_lower:
    #                 if "iit delhi" in content_lower and "iiit delhi" not in content_lower:
    #                     filtered.append(doc)
    #             elif "iiit delhi" in query_lower:
    #                 if "iiit delhi" in content_lower:
    #                     filtered.append(doc)
    #             else:
    #                 # If user didn't specify 'IIT' or 'IIIT', we keep all
    #                 filtered.append(doc)
            
    #         return [d.page_content for d in filtered]

    #     except Exception as e:
    #         logger.error(f"Knowledge retrieval error: {str(e)}")
    #         return []

knowledge_fetcher = KnowledgeFetcher()

###############################################################################
# EducationAssistant
###############################################################################
class EducationAssistant:
    """Complete conversation workflow."""
    def __init__(self):
        self.memory = MemoryManager()
        self.graph = self._build_workflow()
        self.intent_detector = IntentDetector()
        self.response_generator = ResponseGenerator()

    def _build_workflow(self):
        workflow = StateGraph(dict)

        # Graph nodes
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("fetch_knowledge", self._fetch_knowledge)
        workflow.add_node("generate_response", self._generate_response)

        # Edges
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "classify_intent")
        
        # Conditional next step based on intent
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
        """Retrieve relevant memory and minimal session data."""
        thread_id = state["thread_id"]
        session = session_manager.retrieve_session(thread_id)

        # Retrieve memory
        state["memories"] = self.memory.recall_memories(
            query=state["query"],
            filters={"thread_id": thread_id},
            k=Config.MEMORY_SEARCH_K
        )
        # Provide minimal 'history' for intent detection
        state["history"] = session.get("history", [])
        return state

    def _classify_intent(self, state: dict):
        """Classify user intent with LLM-based prompt."""
        detection_result = self.intent_detector.detect(state["history"], state["query"])
        state.update(detection_result)  # Merge into state
        return state

    def _fetch_knowledge(self, state: dict):
        """If intent is 'education', fetch from AWS Knowledge Base, else empty."""
        if state["intent"] == "education":
            docs = knowledge_fetcher.fetch(state["query"])
            state["documents"] = docs
        else:
            state["documents"] = []
        return state

    def _generate_response(self, state: dict):
        """Generate final response & store conversation in memory."""
        response_text = self.response_generator.generate(state)
        state["response"] = response_text

        # Update session with new conversation
        thread_id = state["thread_id"]
        session = session_manager.retrieve_session(thread_id)
        # Append the user query and bot response to history
        session.setdefault("history", []).append(state["query"])
        session.setdefault("history", []).append(response_text)
        session_manager.update_session(thread_id, "history", session["history"])

        return state

    def _route_intent(self, state: dict):
        """Route next node based on detected intent."""
        return state.get("intent", "default")

# Example usage
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
            {"query": query, "thread_id": thread_id},
            {"configurable": {"thread_id": thread_id}}
        )
        
        bot_response = result.get("response", "Error processing request")
        print(f"Bot: {bot_response}")
        history.append(f"User: {query}")
        history.append(f"Assistant: {bot_response}")



# import os
# import json
# import re
# import logging
# from datetime import datetime
# from typing import List, Dict, Optional
# from dotenv import load_dotenv

# # Graph workflow
# from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver

# # AWS & LangChain AWS
# import boto3
# from langchain_aws import BedrockEmbeddings, ChatBedrock
# from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever

# # Core library
# from langchain_core.documents import Document
# from langchain_core.vectorstores import InMemoryVectorStore

# # Configuration
# load_dotenv()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class Config:
#     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "BVWGHMKJOQ")  # Overriding with the known ID
#     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
#     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
#     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
#     MEMORY_SEARCH_K = 5
#     MAX_HISTORY = 5

# # AWS Client Initialization
# bedrock_runtime = boto3.client(
#     service_name="bedrock-runtime",
#     region_name=Config.AWS_REGION
# )

# # -------------------------------
# # SESSION MANAGER
# # -------------------------------
# class SessionManager:
#     """Manages user sessions and conversation context."""
#     def __init__(self):
#         self.sessions = {}

#     def store_session(self, thread_id, data):
#         self.sessions[thread_id] = data

#     def retrieve_session(self, thread_id):
#         return self.sessions.get(thread_id, {})

#     def update_session(self, thread_id, key, value):
#         session = self.retrieve_session(thread_id)
#         session[key] = value
#         self.store_session(thread_id, session)

# session_manager = SessionManager()

# # -------------------------------
# # MEMORY MANAGER
# # -------------------------------
# class MemoryManager:
#     """Manages memory with InMemoryVectorStore + fallback searching."""
#     def __init__(self):
#         # Create an in-memory vector store with Titan embeddings
#         self.vector_store = InMemoryVectorStore(
#             BedrockEmbeddings(
#                 client=bedrock_runtime,
#                 model_id=Config.EMBED_MODEL
#             )
#         )

#     def store_memory(self, text: str, metadata: dict):
#         """Stores conversation text into the vector store."""
#         doc = Document(
#             page_content=text,
#             metadata={
#                 "timestamp": datetime.utcnow().isoformat(),
#                 **metadata
#             }
#         )
#         self.vector_store.add_documents([doc])

#     def recall_memories(self, query: str, filters: dict, k: int = Config.MEMORY_SEARCH_K) -> List[str]:
#         """
#         Retrieves past conversations relevant to `query`.
#         1. Uses vector search first.
#         2. Manually filters by `filters`.
#         3. If no docs found, fallback to simple substring search within the vector store's memory.
#         """
#         try:
#             # 1. Vector-based search
#             docs = self.vector_store.similarity_search(query, k=k)
            
#             # 2. Filter by metadata
#             def _filter(doc: Document) -> bool:
#                 return all(doc.metadata.get(f_key) == f_val for f_key, f_val in filters.items())
#             filtered_docs = [doc for doc in docs if _filter(doc)]

#             # 3. Fallback if none found
#             if not filtered_docs:
#                 # No direct method to retrieve "all documents" from InMemoryVectorStore, so we rely only on the vector search results
#                 # We can do a secondary text search within the top retrieved docs if needed
#                 # But let's keep it simple: we already have 'docs' from similarity_search
#                 substring_matches = [
#                     doc for doc in docs if query.lower() in doc.page_content.lower()
#                 ]
#                 # If still none, we skip
#                 filtered_docs = substring_matches or []

#             return [d.page_content for d in filtered_docs[:k]]
#         except Exception as e:
#             logger.error(f"Memory retrieval error: {str(e)}")
#             return []

# # -------------------------------
# # INTENT DETECTOR
# # -------------------------------
# class IntentDetector:
#     """Robust LLM-based intent classification with fallback."""
#     PROMPT_TEMPLATE = """Analyze the query and classify intent. Respond in JSON format:
#     {{
#         "intent": "education|greeting|memory|out_of_scope",
#         "entities": {{
#             "institution": "string",
#             "program": "string"
#         }},
#         "needs_clarification": boolean,
#         "clarification_prompt": "string"
#     }}
    
#     Query: {query}
#     History: {history}"""

#     def detect(self, history: List[str], query: str) -> dict:
#         """Detect the user's intent using an LLM prompt."""
#         try:
#             prompt = self.PROMPT_TEMPLATE.format(
#                 query=query,
#                 history="\n".join(history[-Config.MAX_HISTORY:])
#             )
#             response = ChatBedrock(
#                 client=bedrock_runtime,
#                 model_id=Config.CHAT_MODEL
#             ).invoke(prompt)
#             return self._parse_response(response.content)
#         except Exception as e:
#             logger.error(f"Intent error: {str(e)}")
#             return {
#                 "intent": "education",
#                 "needs_clarification": False,
#                 "clarification_prompt": ""
#             }

#     def _parse_response(self, response_text: str) -> dict:
#         """Parses the JSON response from the LLM-based prompt."""
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
#             logger.warning("Failed to parse LLM response as JSON. Defaulting to education intent.")
#             return {
#                 "intent": "education",
#                 "needs_clarification": False,
#                 "clarification_prompt": ""
#             }

# # -------------------------------
# # RESPONSE GENERATOR
# # -------------------------------
# class ResponseGenerator:
#     """Generates final response using the LLM."""
#     def __init__(self):
#         self.memory = MemoryManager()

#     def generate(self, context: dict) -> str:
#         """Build prompt from context & generate the final user response."""
#         if context.get("needs_clarification"):
#             return context["clarification_prompt"]

#         try:
#             prompt = self._build_prompt(context)
#             response = ChatBedrock(
#                 client=bedrock_runtime,
#                 model_id=Config.CHAT_MODEL
#             ).invoke(prompt)
#             self._store_conversation(context, response.content)
#             return response.content
#         except Exception as e:
#             logger.error(f"Response generation error: {str(e)}")
#             return "I encountered an error. Please try again."

#     def _build_prompt(self, context: dict) -> str:
#         """Constructs the final LLM prompt from conversation details."""
#         lines = [
#             "Generate a professional educational response using:",
#             f"1. Query: {context['query']}",
#             f"2. History: {context.get('history', [])}",
#             f"3. Documents: {context.get('documents', [])}",
#             f"4. Memories: {context.get('memories', [])}",
#             "Be concise, factual, and cite sources when available."
#         ]
#         return "\n".join(lines)

#     def _store_conversation(self, context: dict, response: str):
#         """Stores the final Q&A into memory for future recall."""
#         self.memory.store_memory(
#             text=f"Q: {context['query']}\nA: {response}",
#             metadata={
#                 "thread_id": context["thread_id"],
#                 "intent": context["intent"]
#             }
#         )

# # -------------------------------
# # KNOWLEDGE FETCH STEP
# # -------------------------------
# class KnowledgeFetcher:
#     """Fetches from AmazonKnowledgeBasesRetriever with correct usage."""
#     def fetch(self, query: str) -> List[str]:
#         if not Config.BEDROCK_KB_ID:
#             logger.warning("No BEDROCK_KB_ID set. Returning empty docs.")
#             return []

#         try:
#             retriever = AmazonKnowledgeBasesRetriever(
#                 knowledge_base_id=Config.BEDROCK_KB_ID,
#                 retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}}
#             )
#             # The correct method to call is `.retrieve(...)`:
#             docs = retriever.retrieve(query)
#             return [d.page_content for d in docs]
#         except Exception as e:
#             logger.error(f"Knowledge retrieval error: {str(e)}")
#             return []

# knowledge_fetcher = KnowledgeFetcher()

# # -------------------------------
# # EDUCATION ASSISTANT
# # -------------------------------
# class EducationAssistant:
#     """Complete conversation workflow with a langgraph StateGraph."""
#     def __init__(self):
#         self.memory = MemoryManager()
#         self.intent_detector = IntentDetector()
#         self.response_generator = ResponseGenerator()
#         self.graph = self._build_workflow()

#     def _build_workflow(self):
#         workflow = StateGraph(dict)

#         workflow.add_node("retrieve_context", self._retrieve_context)
#         workflow.add_node("classify_intent", self._classify_intent)
#         workflow.add_node("fetch_knowledge", self._fetch_knowledge)
#         workflow.add_node("generate_response", self._generate_response)

#         workflow.add_edge(START, "retrieve_context")
#         workflow.add_edge("retrieve_context", "classify_intent")

#         # Use route_intent to decide next step
#         workflow.add_conditional_edges(
#             "classify_intent",
#             self._route_intent,
#             {
#                 "education": "fetch_knowledge",
#                 "greeting": "generate_response",
#                 "memory": "generate_response",
#                 "out_of_scope": "generate_response",
#                 "default": "generate_response"
#             }
#         )
#         workflow.add_edge("fetch_knowledge", "generate_response")
#         workflow.add_edge("generate_response", END)

#         return workflow.compile(checkpointer=MemorySaver())

#     def _retrieve_context(self, state: dict):
#         """Retrieves relevant memory before classification."""
#         thread_id = state["thread_id"]
#         # Retrieve session data
#         session = session_manager.retrieve_session(thread_id)

#         # Retrieve memories
#         state["memories"] = self.memory.recall_memories(
#             query=state["query"],
#             filters={"thread_id": thread_id},
#             k=Config.MEMORY_SEARCH_K
#         )

#         # Provide minimal history to the intent detection
#         state["history"] = session.get("history", [])
#         return state

#     def _classify_intent(self, state: dict):
#         """Classifies user intent with LLM prompt."""
#         result = self.intent_detector.detect(
#             state["history"],
#             state["query"]
#         )
#         # Merge results into state
#         state.update(result)
#         return state

#     def _fetch_knowledge(self, state: dict):
#         """Fetch structured knowledge from AWS Bedrock KB if intent is education."""
#         if state["intent"] == "education":
#             docs = knowledge_fetcher.fetch(state["query"])
#             state["documents"] = docs
#         else:
#             state["documents"] = []
#         return state

#     def _generate_response(self, state: dict):
#         """Generates the final user response & updates session memory."""
#         response = self.response_generator.generate(state)
#         state["response"] = response

#         # Update session
#         thread_id = state["thread_id"]
#         session = session_manager.retrieve_session(thread_id)
#         # Append conversation to session history
#         session.setdefault("history", []).append(state["query"])
#         session.setdefault("history", []).append(response)
#         session_manager.update_session(thread_id, "history", session["history"])
#         return state

#     def _route_intent(self, state: dict):
#         """Routes next step based on detected intent."""
#         intent = state.get("intent", "default")
#         # If not recognized, fallback to default
#         return intent if intent in ("education","greeting","memory","out_of_scope") else "default"


# # -------------------------------
# # Main Execution
# # -------------------------------
# if __name__ == "__main__":
#     assistant = EducationAssistant()
#     thread_id = "demo_thread7"
    
#     queries = [
#         "Hi there!",
#         "What's the iit delhi CS admission process?",
#         "What about eligibilt crieteria?",
#         "What did we discussed about iit delhi",
#         "what is this ",
#         "What's the weather today",
#         "What we discussed earlier",
#         "What about new developments",
#     ]
    
#     history = []
#     for query in queries:
#         print(f"\nUser: {query}")
#         result = assistant.graph.invoke(
#             {"query": query, "thread_id": thread_id},
#             {"configurable": {"thread_id": thread_id}}
#         )
#         bot_response = result.get("response", "Error processing request")
#         print(f"Bot: {bot_response}")
#         history.append(f"User: {query}")
#         history.append(f"Assistant: {bot_response}")

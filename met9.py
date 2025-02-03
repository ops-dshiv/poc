import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurations
class Config:
    BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
    AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
    EMBED_MODEL = "amazon.titan-embed-text-v2:0"
    CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
    MEMORY_SEARCH_K = 5
    MAX_HISTORY = 5

# AWS Bedrock Client Initialization
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=Config.AWS_REGION
)

import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
    AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
    EMBED_MODEL = "amazon.titan-embed-text-v2:0"
    CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
    MEMORY_SEARCH_K = 5
    MAX_HISTORY = 7
    TOPIC_SIMILARITY_THRESHOLD = 0.78

# AWS Clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)
embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id=Config.EMBED_MODEL)
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SessionManager:
    """Handles conversation context and topic tracking"""
    def __init__(self):
        self.sessions = {}
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def _get_topic_embedding(self, text: str) -> List[float]:
        return embeddings.embed_documents([text])[0]

    def create_session(self, thread_id: str):
        self.sessions[thread_id] = {
            "current_topic": "",
            "topic_embedding": [],
            "history": [],
            "entities": {},
            "pending_clarification": None
        }

    def update_session(self, thread_id: str, updates: dict):
        if thread_id in self.sessions:
            self.sessions[thread_id].update(updates)
            if "current_topic" in updates:
                new_embedding = self._get_topic_embedding(updates["current_topic"])
                if self.sessions[thread_id]["topic_embedding"]:
                    similarity = self._cosine_similarity(
                        self.sessions[thread_id]["topic_embedding"], 
                        new_embedding
                    )
                    if similarity < Config.TOPIC_SIMILARITY_THRESHOLD:
                        self.sessions[thread_id]["history"] = []
                self.sessions[thread_id]["topic_embedding"] = new_embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x*y for x,y in zip(a,b))
        norm_a = sum(x**2 for x in a)**0.5
        norm_b = sum(x**2 for x in b)**0.5
        return dot / (norm_a * norm_b)

class MemoryManager:
    """Handles conversation memory with vector-based recall"""
    def __init__(self):
        self.vector_store = InMemoryVectorStore(embeddings)
    
    def store_conversation(self, thread_id: str, query: str, response: str):
        doc = Document(
            page_content=f"User: {query}\nAssistant: {response}",
            metadata={
                "thread_id": thread_id,
                "timestamp": datetime.utcnow().isoformat(),
                "topic": self._detect_topic(query, response)
            }
        )
        self.vector_store.add_documents([doc])

    def _detect_topic(self, query: str, response: str) -> str:
        prompt = f"Identify the main topic from:\nQ: {query}\nA: {response}\nTopic:"
        return ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt).content

    def recall_context(self, thread_id: str, query: str) -> List[str]:
        docs = self.vector_store.similarity_search(
            query=query,
            k=Config.MEMORY_SEARCH_K,
            filter=lambda d: d.metadata["thread_id"] == thread_id
        )
        return [f"{d.metadata['timestamp']}: {d.page_content}" for d in docs]

class KnowledgeRetriever:
    """Handles Bedrock RAG integration with error recovery"""
    def retrieve(self, query: str) -> List[str]:
        try:
            response = bedrock_runtime.retrieve(
                retrievalQuery={'text': query},
                knowledgeBaseId=Config.BEDROCK_KB_ID,
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': 5,
                        'searchMethod': 'HYBRID'
                    }
                }
            )
            return [result['content']['text'] for result in response['retrievalResults']]
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {str(e)}")
            return []

class AmbiguityResolver:
    """Handles vague queries and ambiguous references"""
    AMBIGUOUS_TRIGGERS = {"this", "that", "it", "they", "what about", "how about"}
    
    def needs_clarification(self, query: str, history: List[str]) -> bool:
        query_lower = query.lower()
        if len(history) == 0 and any(t in query_lower for t in ["hi", "hello", "hey"]):
            return any(trigger in query_lower for trigger in self.AMBIGUOUS_TRIGGERS)
        return (
            len(query.split()) < 5 and
            any(trigger in query_lower for trigger in self.AMBIGUOUS_TRIGGERS)
        )

    def clarification_prompt(self, history: List[str]) -> str:
        if history:
            last_topic = self._extract_last_topic(history[-1])
            return f"You were asking about {last_topic}. Could you clarify your question?"
        return "Could you please provide more details about your query?"

    def _extract_last_topic(self, history_entry: str) -> str:
        return ChatBedrock(
            client=bedrock_runtime,
            model_id=Config.CHAT_MODEL
        ).invoke(f"Extract main topic from: {history_entry}").content

class EducationAssistant:
    """Main chatbot implementation with full feature set"""
    def __init__(self):
        self.sessions = SessionManager()
        self.memory = MemoryManager()
        self.retriever = KnowledgeRetriever()
        self.ambiguity = AmbiguityResolver()
        self.graph = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(dict)
        workflow.add_nodes([
            ("init_session", self._init_session),
            ("check_ambiguity", self._check_ambiguity),
            ("handle_clarification", self._handle_clarification),
            ("retrieve_context", self._retrieve_context),
            ("process_query", self._process_query),
            ("generate_response", self._generate_response)
        ])

        workflow.add_edge(START, "init_session")
        workflow.add_edge("init_session", "check_ambiguity")
        workflow.add_conditional_edges(
            "check_ambiguity",
            self._route_ambiguity,
            {"ambiguous": "handle_clarification", "clear": "retrieve_context"}
        )
        workflow.add_edge("handle_clarification", "generate_response")
        workflow.add_edge("retrieve_context", "process_query")
        workflow.add_edge("process_query", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile(checkpointer=MemorySaver())

    def _init_session(self, state: dict):
        thread_id = state["thread_id"]
        if thread_id not in self.sessions.sessions:
            self.sessions.create_session(thread_id)
        return state

    def _check_ambiguity(self, state: dict):
        thread_id = state["thread_id"]
        session = self.sessions.sessions[thread_id]
        state["needs_clarification"] = self.ambiguity.needs_clarification(
            state["query"], session["history"]
        )
        return state

    def _route_ambiguity(self, state: dict):
        return "ambiguous" if state["needs_clarification"] else "clear"

    def _handle_clarification(self, state: dict):
        thread_id = state["thread_id"]
        session = self.sessions.sessions[thread_id]
        state["response"] = self.ambiguity.clarification_prompt(session["history"])
        return state

    def _retrieve_context(self, state: dict):
        thread_id = state["thread_id"]
        state["memories"] = self.memory.recall_context(thread_id, state["query"])
        state["knowledge"] = self.retriever.retrieve(state["query"])
        return state

    def _process_query(self, state: dict):
        thread_id = state["thread_id"]
        prompt = self._build_prompt(state)
        response = ChatBedrock(
            client=bedrock_runtime,
            model_id=Config.CHAT_MODEL
        ).invoke(prompt)
        
        self.sessions.update_session(thread_id, {
            "current_topic": self.memory._detect_topic(state["query"], response.content),
            "history": self.sessions.sessions[thread_id]["history"] + [
                f"User: {state['query']}", f"Assistant: {response.content}"
            ]
        })
        
        state["response"] = response.content
        return state

    def _build_prompt(self, state: dict) -> str:
        components = [
            "Generate an educational response considering:",
            f"Query: {state['query']}",
            "Conversation History:",
            *state.get("memories", []),
            "Relevant Knowledge:",
            *state.get("knowledge", [])
        ]
        return "\n".join(components)

    def _generate_response(self, state: dict):
        thread_id = state["thread_id"]
        self.memory.store_conversation(thread_id, state["query"], state["response"])
        return state

# Example Usage
if __name__ == "__main__":
    assistant = EducationAssistant()
    thread_id = "demo_user_123"

    test_flow = [
        "Hi there!",
        "What's the computer science program like?",
        "What about admission requirements?",
        "How about fees?",
        "Tell me about mechanical engineering instead",
        "What's the duration?"
    ]

    for query in test_flow:
        print(f"\nUser: {query}")
        result = assistant.graph.invoke(
            {"query": query, "thread_id": thread_id},
            {"configurable": {"thread_id": thread_id}}
        )
        print(f"Assistant: {result['response']}")

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
#                 modelId="amazon.titan-embed-text-v2:0",
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


# # Example Usage
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
#             {"query": query, "history": history, "thread_id": thread_id},
#             {"configurable": {"thread_id": thread_id}}
#         )
        
#         response = result.get("response", "Error processing request")
#         history.append(f"User: {query}")
#         history.append(f"Assistant: {response}")
#         print(f"Bot: {response}")
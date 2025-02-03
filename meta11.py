# import os
# import json
# import re
# import logging
# from datetime import datetime
# from typing import List, Dict, Optional

# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver

# import boto3
# from langchain_aws import BedrockEmbeddings, ChatBedrock
# from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
# from langchain_core.documents import Document
# from langchain_core.vectorstores import InMemoryVectorStore

# # Load environment variables
# load_dotenv()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configuration
# class Config:
#     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "YOUR_KB_ID")
#     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
#     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
#     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
#     MEMORY_SEARCH_K = 5
#     MAX_HISTORY = 5
#     INTENT_EXAMPLES_FILE = "intent_examples.json"

# bedrock_runtime = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)

# # Session Manager
# class SessionManager:
#     def __init__(self):
#         self.sessions = {}

#     def store_session(self, thread_id: str, data: dict):
#         self.sessions[thread_id] = data

#     def retrieve_session(self, thread_id: str) -> dict:
#         return self.sessions.get(thread_id, {
#             "history": [],
#             "introduced": False,
#             "clarification_attempts": 0
#         })

#     def update_session(self, thread_id: str, updates: dict):
#         session = self.retrieve_session(thread_id)
#         session.update(updates)
#         self.store_session(thread_id, session)

# # Memory Manager
# class MemoryManager:
#     def __init__(self):
#         self.vector_store = InMemoryVectorStore(
#             BedrockEmbeddings(client=bedrock_runtime, model_id=Config.EMBED_MODEL)
#         )

#     def store_memory(self, text: str, metadata: dict):
#         self.vector_store.add_documents([Document(
#             page_content=text,
#             metadata={"timestamp": datetime.utcnow().isoformat(), **metadata}
#         )])

#     def recall_memories(self, query: str, filters: dict, k: int = Config.MEMORY_SEARCH_K) -> List[str]:
#         try:
#             docs = self.vector_store.similarity_search(query, k=k)
#             return [d.page_content for d in docs if all(
#                 d.metadata.get(fk) == fv for fk, fv in filters.items()
#             )]
#         except Exception as e:
#             logger.error(f"Memory error: {str(e)}")
#             return []

# # Intent Detector
# class IntentDetector:
#     def __init__(self):
#         with open(Config.INTENT_EXAMPLES_FILE) as f:
#             self.examples = json.load(f)

#     PROMPT_TEMPLATE = """Analyze query and history. Examples: {examples}
# Respond in JSON: {{
#     "intent": "education|greeting|memory|out_of_scope|vague|introduction",
#     "entities": {{"institution": "...", "program": "..."}},
#     "needs_clarification": boolean,
#     "clarification_prompt": "..."
# }}"""

#     def detect(self, history: List[dict], query: str) -> dict:
#         try:
#             prompt = self.PROMPT_TEMPLATE.format(
#                 examples=json.dumps(self.examples["intents"]),
#                 query=query,
#                 history=self._format_history(history)
#             )
#             response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
#             return self._parse_response(response.content)
#         except Exception as e:
#             logger.error(f"Intent error: {str(e)}")
#             return self._fallback_intent()

#     def _format_history(self, history: List[dict]) -> str:
#         return "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in history[-3:]]) if history else "No history"

#     def _parse_response(self, response_text: str) -> dict:
#         try:
#             return json.loads(re.search(r'{.*}', response_text, re.DOTALL).group())
#         except:
#             return self._fallback_intent()

#     def _fallback_intent(self) -> dict:
#         return {"intent": "education", "needs_clarification": False}

# class SessionManager:
#     """Manages session context (history, last topic, etc.) in memory."""
#     def __init__(self):
#         self.sessions = {}

#     def store_session(self, thread_id: str, data: dict):
#         self.sessions[thread_id] = data

#     def retrieve_session(self, thread_id: str) -> dict:
#         return self.sessions.get(thread_id, {})

#     def update_session(self, thread_id: str, key: str, value):
#         session = self.retrieve_session(thread_id)
#         session[key] = value
#         self.store_session(thread_id, session)

# session_manager = SessionManager()

# # Response Generator
# class ResponseGenerator:
#     INTRODUCTION = """🎓 *Edu360 - Your Complete Education Advisor* 📚

# I specialize in Indian higher education with 360° insights on:
# ✅ Top Institutions (IITs, NITs, IIITs, Central Universities)
# ✅ 1000+ Courses (Engineering, Medicine, Management, Arts)
# ✅ Admission Processes & Entrance Exams
# ✅ Fee Structures & Scholarships
# ✅ Campus Facilities & Placements

# Ask me about:
# • IIT Delhi CS admissions
# • NIRF rankings
# • JEE Advanced cutoffs
# • College comparisons
# • Course curricula

# How can I assist you today?"""

#     def __init__(self):
#         self.memory = MemoryManager()
#         with open(Config.INTENT_EXAMPLES_FILE) as f:
#             self.responses = json.load(f)["responses"]

#     def generate(self, context: dict) -> str:
#         intent = context["intent"]
        
#         if intent == "greeting":
#             return self._handle_greeting(context)
#         if intent == "introduction":
#             return self.INTRODUCTION
#         if intent == "out_of_scope":
#             return self._handle_out_of_scope()
#         if intent == "vague":
#             return self._handle_vague(context)
#         if context.get("needs_clarification"):
#             return context["clarification_prompt"]
        
#         return self._generate_education_response(context)

#     def _handle_greeting(self, context: dict) -> str:
#         session = SessionManager().retrieve_session(context["thread_id"])
#         return self.INTRODUCTION if not session.get("introduced") else "How can I help you further?"

#     def _handle_out_of_scope(self) -> str:
#         return random.choice(self.responses["out_of_scope"])

#     def _handle_vague(self, context: dict) -> str:
#         history = context.get("history", [])
#         last_topic = self._detect_last_topic(history)
#         return f"Are you asking about {last_topic} or something else? Please specify."

#     def _generate_education_response(self, context: dict) -> str:
#         prompt = self._build_prompt(context)
#         try:
#             response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
#             return self._validate_response(response.content, context.get("documents", []))
#         except Exception as e:
#             logger.error(f"Response error: {str(e)}")
#             return "I encountered an error. Please try again."

#     # def _build_prompt(self, context: dict) -> str:
#     #     return "\n".join([
#     #         f"Generate comprehensive response about {context['entities'].get('institution', 'Indian institution')}",
#     #         f"Query: {context['query']}",
#     #         f"History: {self._summarize_history(context.get('history', []))}",
#     #         f"Documents: {context.get('documents', [])[:3]}",
#     #         "Include: Admission process, Eligibility, Fees, Placements, Facilities",
#     #         "Format: [Details] [Official Source]"
#     #     ])


#     def _summarize_history(self, history: list) -> str:
#         education_points = []
#         for entry in history[-3:]:
#             if isinstance(entry, dict) and "admission" in entry.get("bot", "").lower():
#                 education_points.append(entry["bot"])
#         return "\n".join([
#             f"{i+1}. {point.split('. ')[1]}" 
#             for i, point in enumerate(education_points[-3:])
#         ]) or "No specific admission details discussed yet"


#     def _build_prompt(self, context: dict) -> str:
#         return "\n".join([
#             f"Generate comprehensive response about {context.get('entities', {}).get('institution', 'Indian institution')}",
#             f"Query: {context['query']}",
#             f"History: {self._summarize_history(context.get('history', []))}",
#             f"Documents: {str(context.get('documents', []))[:300]}",
#             "Include: Admission process, Eligibility, Fees, Placements, Facilities",
#             "Format: [Details] [Official Source]"
#         ])



# class KnowledgeFetcher:
#     """Fetch docs from AWS Bedrock Knowledge Base using .invoke(...)."""
#     def fetch(self, query: str) -> List[str]:
#         if not Config.BEDROCK_KB_ID:
#             logger.warning("No BEDROCK_KB_ID set, returning empty.")
#             return []

#         try:
#             retriever = AmazonKnowledgeBasesRetriever(
#                 knowledge_base_id=Config.BEDROCK_KB_ID,
#                 retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}}
#             )
#             docs = retriever.invoke(query)

#             # Optional post-filter if user specifically says "IIT Delhi"
#             q_lower = query.lower()
#             if "iit delhi" in q_lower and "iiit delhi" not in q_lower:
#                 filtered = []
#                 for d in docs:
#                     if "iit delhi" in d.page_content.lower() and "iiit delhi" not in d.page_content.lower():
#                         filtered.append(d)
#                 docs = filtered

#             return [doc.page_content for doc in docs]
#         except Exception as e:
#             logger.error(f"Knowledge retrieval error: {str(e)}")
#             return []

# knowledge_fetcher = KnowledgeFetcher()

# # Education Assistant
# class EducationAssistant:
#     def __init__(self):
#         self.graph = self._build_workflow()
#         self.intent_detector = IntentDetector()
#         self.response_generator = ResponseGenerator()
#         self.memory = MemoryManager()


#     def _build_workflow(self):
#         workflow = StateGraph(dict)
#         nodes = ["retrieve_context", "classify_intent", "fetch_knowledge", "generate_response"]
#         for node in nodes:
#             workflow.add_node(node, getattr(self, f"_{node}"))
        
#         workflow.add_edge(START, "retrieve_context")
#         workflow.add_edge("retrieve_context", "classify_intent")
#         workflow.add_conditional_edges(
#             "classify_intent", self._route_intent,
#             {"education": "fetch_knowledge", "default": "generate_response"}
#         )
#         workflow.add_edge("fetch_knowledge", "generate_response")
#         workflow.add_edge("generate_response", END)
        
#         return workflow.compile(checkpointer=MemorySaver())

#     def _retrieve_context(self, state: dict):
#         """
#         Retrieve relevant memory & minimal session data before classification.
#         """
#         thread_id = state["thread_id"]
#         session = session_manager.retrieve_session(thread_id)

#         # 1. Memory retrieval
#         state["memories"] = self.memory.recall_memories(
#             query=state["query"],
#             filters={"thread_id": thread_id},
#             k=Config.MEMORY_SEARCH_K
#         )

#         # 2. Provide minimal 'history' for intent detection
#         state["history"] = session.get("history", [])
#         return state

#     def _classify_intent(self, state: dict):
#         """
#         Classify user intent with LLM-based prompt.
#         """
#         detection_result = self.intent_detector.detect(
#             state["history"],
#             state["query"]
#         )
#         state.update(detection_result)
#         return state

#     def _fetch_knowledge(self, state: dict):
#         """
#         If intent is 'education', fetch from AWS Knowledge Base, else skip.
#         """
#         if state["intent"] == "education":
#             docs = knowledge_fetcher.fetch(state["query"])
#             state["documents"] = docs
#         else:
#             state["documents"] = []
#         return state

#     def _generate_response(self, state: dict):
#         """
#         Generate the final user response & store conversation in memory.
#         """
#         response_text = self.response_generator.generate(state)
#         state["response"] = response_text

#         # Update session
#         thread_id = state["thread_id"]
#         session = session_manager.retrieve_session(thread_id)

#         # Append user query & bot response to conversation history
#         session.setdefault("history", []).append(state["query"])
#         session.setdefault("history", []).append(response_text)
#         session_manager.update_session(thread_id, "history", session["history"])

#         return state

#     def _route_intent(self, state: dict):
#         """
#         Route next node based on detected intent.
#         If unknown, fallback to 'default'.
#         """
#         print(">>>>>>>>>>>>>>>>>> intent understanding", state.get("intent", "default"))
        
#         intent = state.get("intent", "default").lower()


#         # if intent in {"vauge", "vague"}:
#         #     return "education"
#         return state.get("intent", "default")


#     # Implement node methods and other helper functions here

# if __name__ == "__main__":
#     assistant = EducationAssistant()
#     thread_id = "main_session"
    
#     test_queries = [
#         "Hi there!",
#         "What's the IIT Delhi CS admission process?",
#         "What about eligibility criteria?",
#         "What's the weather today?",
#         "What did we discuss?",
#         "Tell me about new developments"
#     ]
    
#     for query in test_queries:
#         print(f"\nUser: {query}")
#         result = assistant.graph.invoke(
#             {"query": query, "thread_id": thread_id},
#             {"configurable": {"thread_id": thread_id}}
#         )
#         print(f"Bot: {result['response'][:500]}...")


import os
import json
import re
import logging
import random  # Needed for random.choice in out-of-scope handling
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

# Configuration
class Config:
    BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "BVWGHMKJOQ")
    AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
    EMBED_MODEL = "amazon.titan-embed-text-v2:0"
    CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
    MEMORY_SEARCH_K = 5
    MAX_HISTORY = 5
    INTENT_EXAMPLES_FILE = "intent_examples.json"

# Create the Bedrock client
bedrock_runtime = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)

# Session Manager (single definition)
class SessionManager:
    def __init__(self):
        self.sessions = {}

    def store_session(self, thread_id: str, data: dict):
        self.sessions[thread_id] = data

    def retrieve_session(self, thread_id: str) -> dict:
        return self.sessions.get(thread_id, {
            "history": [],
            "introduced": False,
            "clarification_attempts": 0
        })

    def update_session(self, thread_id: str, updates: dict):
        session = self.retrieve_session(thread_id)
        session.update(updates)
        self.store_session(thread_id, session)

session_manager = SessionManager()

# Global knowledge fetcher instance
knowledge_fetcher = AmazonKnowledgeBasesRetriever(client=bedrock_runtime, kb_id=Config.BEDROCK_KB_ID)

# Memory Manager
class MemoryManager:
    def __init__(self):
        self.vector_store = InMemoryVectorStore(
            BedrockEmbeddings(client=bedrock_runtime, model_id=Config.EMBED_MODEL)
        )

    def store_memory(self, text: str, metadata: dict):
        self.vector_store.add_documents([Document(
            page_content=text,
            metadata={"timestamp": datetime.utcnow().isoformat(), **metadata}
        )])

    def recall_memories(self, query: str, filters: dict, k: int = Config.MEMORY_SEARCH_K) -> List[str]:
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [d.page_content for d in docs if all(
                d.metadata.get(fk) == fv for fk, fv in filters.items()
            )]
        except Exception as e:
            logger.error(f"Memory error: {str(e)}")
            return []

# Intent Detector
class IntentDetector:
    def __init__(self):
        with open(Config.INTENT_EXAMPLES_FILE) as f:
            self.examples = json.load(f)

    PROMPT_TEMPLATE = """Analyze query and history. Examples: {examples}
Respond in JSON: {{
    "intent": "education|greeting|memory|out_of_scope|vague|introduction",
    "entities": {{"institution": "...", "program": "..."}},
    "needs_clarification": boolean,
    "clarification_prompt": "..."
}}"""

    def detect(self, history: List[dict], query: str) -> dict:
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                examples=json.dumps(self.examples["intents"]),
                query=query,
                history=self._format_history(history)
            )
            response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
            return self._parse_response(response.content)
        except Exception as e:
            logger.error(f"Intent error: {str(e)}")
            return self._fallback_intent()

    def _format_history(self, history: List[dict]) -> str:
        return "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in history[-3:]]) if history else "No history"

    def _parse_response(self, response_content) -> dict:
        # Ensure we are working with a string
        if not isinstance(response_content, str):
            try:
                response_content = json.dumps(response_content)
            except Exception as e:
                logger.error(f"Error converting response to string: {e}")
                return self._fallback_intent()
        try:
            m = re.search(r'{.*}', response_content, re.DOTALL)
            if m:
                return json.loads(m.group())
            else:
                return self._fallback_intent()
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return self._fallback_intent()

    def _fallback_intent(self) -> dict:
        return {"intent": "education", "needs_clarification": False}

# Response Generator
class ResponseGenerator:
    INTRODUCTION = """🎓 *Edu360 - Your Complete Education Advisor* 📚

I specialize in Indian higher education with 360° insights on:
✅ Top Institutions (IITs, NITs, IIITs, Central Universities)
✅ 1000+ Courses (Engineering, Medicine, Management, Arts)
✅ Admission Processes & Entrance Exams
✅ Fee Structures & Scholarships
✅ Campus Facilities & Placements

Ask me about:
• IIT Delhi CS admissions
• NIRF rankings
• JEE Advanced cutoffs
• College comparisons
• Course curricula

How can I assist you today?"""

    def __init__(self):
        self.memory = MemoryManager()
        with open(Config.INTENT_EXAMPLES_FILE) as f:
            self.responses = json.load(f)["responses"]

    def generate(self, context: dict) -> str:
        intent = context["intent"]
        if intent == "greeting":
            return self._handle_greeting(context)
        if intent == "introduction":
            return self.INTRODUCTION
        if intent == "out_of_scope":
            return self._handle_out_of_scope()
        if intent == "vague":
            return self._handle_vague(context)
        if context.get("needs_clarification"):
            return context["clarification_prompt"]
        return self._generate_education_response(context)

    def _handle_greeting(self, context: dict) -> str:
        session = session_manager.retrieve_session(context["thread_id"])
        return self.INTRODUCTION if not session.get("introduced") else "How can I help you further?"

    def _handle_out_of_scope(self) -> str:
        return random.choice(self.responses["redirect_to_scope"])

    def _handle_vague(self, context: dict) -> str:
        history = context.get("history", [])
        last_topic = self._detect_last_topic(history)
        return f"Are you asking about {last_topic} or something else? Please specify."

    def _generate_education_response(self, context: dict) -> str:
        prompt = self._build_prompt(context)
        try:
            response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
            return self._validate_response(response.content, context.get("documents", []))
        except Exception as e:
            logger.error(f"Response error: {str(e)}")
            return "I encountered an error. Please try again."

    def _build_prompt(self, context: dict) -> str:
        return "\n".join([
            f"Generate comprehensive response about {context.get('entities', {}).get('institution', 'Indian institution')}",
            f"Query: {context['query']}",
            f"History: {self._summarize_history(context.get('history', []))}",
            f"Documents: {str(context.get('documents', []))[:300]}",
            "Include: Admission process, Eligibility, Fees, Placements, Facilities",
            "Format: [Details] [Official Source]"
        ])

    def _summarize_history(self, history: list) -> str:
        # Simplified history summarization; you can adjust as needed.
        if not history:
            return "No prior conversation."
        return "\n".join(history[-3:])

    def _detect_last_topic(self, history: list) -> str:
        # A simple implementation: return the last bot response if available.
        if not history:
            return "this topic"
        # Assuming history is a list of strings (bot responses and user queries)
        return history[-1]

    def _validate_response(self, response_text: str, documents: List[str]) -> str:
        return response_text.strip() if response_text.strip() else "I'm sorry, I didn't get that."

# Knowledge Fetcher
class KnowledgeFetcher:
    """Fetch docs from AWS Bedrock Knowledge Base using .invoke(...)."""
    def fetch(self, query: str) -> List[str]:
        if not Config.BEDROCK_KB_ID:
            logger.warning("No BEDROCK_KB_ID set, returning empty.")
            return []
        try:
            retriever = AmazonKnowledgeBasesRetriever(
                knowledge_base_id=Config.BEDROCK_KB_ID,
                retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}}
            )
            docs = retriever.invoke(query)
            # Optional post-filtering for IIT Delhi queries:
            q_lower = query.lower()
            if "iit delhi" in q_lower and "iiit delhi" not in q_lower:
                filtered = []
                for d in docs:
                    if "iit delhi" in d.page_content.lower() and "iiit delhi" not in d.page_content.lower():
                        filtered.append(d)
                docs = filtered
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Knowledge retrieval error: {str(e)}")
            return []

knowledge_fetcher = KnowledgeFetcher()

# Education Assistant
class EducationAssistant:
    def __init__(self):
        self.memory = MemoryManager()
        self.graph = self._build_workflow()
        self.intent_detector = IntentDetector()
        self.response_generator = ResponseGenerator()

    def _build_workflow(self):
        workflow = StateGraph(dict)
        nodes = ["retrieve_context", "classify_intent", "fetch_knowledge", "generate_response"]
        for node in nodes:
            workflow.add_node(node, getattr(self, f"_{node}"))
        
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "classify_intent")
        workflow.add_conditional_edges(
            "classify_intent", self._route_intent,
            {"education": "fetch_knowledge", "default": "generate_response"}
        )
        workflow.add_edge("fetch_knowledge", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    def _retrieve_context(self, state: dict):
        """
        Retrieve relevant memory & minimal session data before classification.
        """
        thread_id = state["thread_id"]
        session = session_manager.retrieve_session(thread_id)
        state["memories"] = self.memory.recall_memories(
            query=state["query"],
            filters={"thread_id": thread_id},
            k=Config.MEMORY_SEARCH_K
        )
        state["history"] = session.get("history", [])
        return state

    def _classify_intent(self, state: dict):
        """
        Classify user intent with LLM-based prompt.
        """
        detection_result = self.intent_detector.detect(
            state["history"],
            state["query"]
        )
        state.update(detection_result)
        return state

    def _fetch_knowledge(self, state: dict):
        """
        If intent is 'education', fetch from AWS Knowledge Base; otherwise skip.
        """
        if state.get("intent") == "education":
            docs = knowledge_fetcher.fetch(state["query"])
            state["documents"] = docs
        else:
            state["documents"] = []
        return state

    def _generate_response(self, state: dict):
        """
        Generate the final user response & store conversation in memory.
        """
        response_text = self.response_generator.generate(state)
        state["response"] = response_text

        # Update session history: append query and response
        thread_id = state["thread_id"]
        session = session_manager.retrieve_session(thread_id)
        session.setdefault("history", []).append(state["query"])
        session.setdefault("history", []).append(response_text)
        session_manager.update_session(thread_id, {"history": session["history"]})

        return state

    def _route_intent(self, state: dict):
        """
        Route next node based on detected intent. Fallback to 'default' if unknown.
        """
        print(">>>>>>>>>>>>>>>>>> intent understanding", state.get("intent", "default"))
        return state.get("intent", "default")

if __name__ == "__main__":
    assistant = EducationAssistant()
    thread_id = "main_session"
    
    test_queries = [
        "Hi there!",
        "What's the IIT Delhi CS admission process?",
        "What about eligibility criteria?",
        "What's the weather today?",
        "What did we discuss?",
        "Tell me about new developments"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        result = assistant.graph.invoke(
            {"query": query, "thread_id": thread_id},
            {"configurable": {"thread_id": thread_id}}
        )
        print(f"Bot: {result['response'][:500]}...")

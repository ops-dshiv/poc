import os
import json
import re
import logging
from datetime import datetime
from typing import List, Dict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
class Config:
    BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "BVWGHMKJOQ")
    AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
    EMBED_MODEL = "amazon.titan-embed-text-v2:0"
    CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
    MEMORY_SEARCH_K = 5
    MAX_HISTORY = 5

# AWS Client Initialization
bedrock_runtime = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)

# -------------------------------------------------------------------
# SessionManager
# -------------------------------------------------------------------
class SessionManager:
    """Manages session context (history, last topic, etc.) in memory."""
    def __init__(self):
        self.sessions = {}

    def store_session(self, thread_id: str, data: dict):
        self.sessions[thread_id] = data

    def retrieve_session(self, thread_id: str) -> dict:
        return self.sessions.get(thread_id, {})

    def update_session(self, thread_id: str, key: str, value):
        session = self.retrieve_session(thread_id)
        session[key] = value
        self.store_session(thread_id, session)

session_manager = SessionManager()

# -------------------------------------------------------------------
# MemoryManager
# -------------------------------------------------------------------
class MemoryManager:
    """Manages memory with vector search + fallback substring search (if needed)."""
    def __init__(self):
        self.vector_store = InMemoryVectorStore(
            BedrockEmbeddings(client=bedrock_runtime, model_id=Config.EMBED_MODEL)
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

    def recall_memories(self, query: str, filters: dict, k: int = Config.MEMORY_SEARCH_K) -> List[str]:
        """
        1. Vector-based search
        2. Filter by 'filters' keys
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            # Apply metadata filtering
            def _filter(doc: Document) -> bool:
                return all(doc.metadata.get(fk) == fv for fk, fv in filters.items())

            filtered = [doc for doc in docs if _filter(doc)]
            return [d.page_content for d in filtered]
        except Exception as e:
            logger.error(f"Memory retrieval error: {str(e)}")
            return []

# -------------------------------------------------------------------
# IntentDetector
# -------------------------------------------------------------------
class IntentDetector:
    """Robust LLM-based intent classification."""
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
            response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
            return self._parse_response(response.content)
        except Exception as e:
            logger.error(f"Intent error: {str(e)}")
            return {
                "intent": "education",
                "needs_clarification": False,
                "clarification_prompt": ""
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
            logger.warning("Could not parse intent JSON from LLM response.")
            return {
                "intent": "education",
                "needs_clarification": False,
                "clarification_prompt": ""
            }

# -------------------------------------------------------------------
# KnowledgeFetcher
# -------------------------------------------------------------------
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

            # Optional post-filter if user specifically says "IIT Delhi"
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

# -------------------------------------------------------------------
# ResponseGenerator
# -------------------------------------------------------------------
class ResponseGenerator:
    """Generates the final response using an LLM prompt."""
    def __init__(self):
        self.memory = MemoryManager()

    def generate(self, context: dict) -> str:
        if context.get("needs_clarification"):
            return context["clarification_prompt"]

        prompt = self._build_prompt(context)
        try:
            response = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL).invoke(prompt)
            self._store_conversation(context, response.content)
            return response.content
        except Exception as e:
            logger.error(f"Response error: {str(e)}")
            return "I encountered an error. Please try again later."

    def _build_prompt(self, context: dict) -> str:
        lines = [
            "Generate a professional educational response using:",
            f"1. Query: {context['query']}",
            f"2. History: {context.get('history', [])}",
            f"3. Documents: {context.get('documents', [])}",
            f"4. Memories: {context.get('memories', [])}",
            "Guidelines: Be concise, factual, and cite sources when available."
        ]
        return "\n".join(lines)

    def _store_conversation(self, context: dict, response: str):
        """Stores the final user-bot exchange in memory for future recall."""
        self.memory.store_memory(
            text=f"Q: {context['query']}\nA: {response}",
            metadata={
                "thread_id": context["thread_id"],
                "intent": context["intent"]
            }
        )

# -------------------------------------------------------------------
# EducationAssistant
# -------------------------------------------------------------------
class EducationAssistant:
    def __init__(self):
        self.memory = MemoryManager()
        self.graph = self._build_workflow()
        self.intent_detector = IntentDetector()
        self.response_generator = ResponseGenerator()

    def _build_workflow(self):
        workflow = StateGraph(dict)

        # Graph Nodes
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
        """
        Retrieve relevant memory & minimal session data before classification.
        """
        thread_id = state["thread_id"]
        session = session_manager.retrieve_session(thread_id)

        # 1. Memory retrieval
        state["memories"] = self.memory.recall_memories(
            query=state["query"],
            filters={"thread_id": thread_id},
            k=Config.MEMORY_SEARCH_K
        )

        # 2. Provide minimal 'history' for intent detection
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
        If intent is 'education', fetch from AWS Knowledge Base, else skip.
        """
        if state["intent"] == "education":
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

        # Update session
        thread_id = state["thread_id"]
        session = session_manager.retrieve_session(thread_id)

        # Append user query & bot response to conversation history
        session.setdefault("history", []).append(state["query"])
        session.setdefault("history", []).append(response_text)
        session_manager.update_session(thread_id, "history", session["history"])

        return state

    def _route_intent(self, state: dict):
        """
        Route next node based on detected intent.
        If unknown, fallback to 'default'.
        """
        return state.get("intent", "default")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    assistant = EducationAssistant()
    thread_id = "demo_thread7"

    queries = [
        "Hi there!",
        "What's the IIT Delhi CS admission process?",
        "What about eligibility criteria?",
        "What did we discuss about IIT Delhi?",
        "what is this ",
        "What's the weather today",
        "What we discussed earlier",
        "What about new developments",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        result = assistant.graph.invoke(
            {"query": query, "thread_id": thread_id},
            {"configurable": {"thread_id": thread_id}}
        )
        bot_response = result.get("response", "Error processing request")
        print(f"Bot: {bot_response}")

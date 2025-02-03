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
    AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "ap-south-1")
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
    # PROMPT_TEMPLATE = """Analyze the query and classify intent. Respond in JSON format:
    # {{
    #     "intent": "education|greeting|vauge|memory|out_of_scope|",
    #     "entities": {{
    #         "institution": "string",
    #         "program": "string"
    #     }},
    #     "needs_clarification": boolean,
    #     "clarification_prompt": "string"
    # }}
    # Query: {query}
    # History: {history} """
    

    PROMPT_TEMPLATE = """**Intent Classification Task**

    You are an expert classifier for educational queries in the Indian context. Your goal is to classify the user's query into one of the following intents and extract relevant entities based on explicit mentions.

    **Rules & Guidelines:**

    1. **Intent Categories:**
    - **greeting:** When the query is a simple greeting.
    - **education:** When the query is about admission processes, rankings, placements, fee structures, or any academic information related to educational institutions.
    - **vague:** When the query is too general or ambiguous, missing critical details such as the institution, program, or academic level.
    - **memory:** When the query refers to or requests recall of previous conversation history.
    - **out_of_scope:** When the query is unrelated to educational topics (e.g., recipes, weather, sports).

    2. **Classification Priorities:**
    - Prioritize specificity: For instance, "IIT Delhi CS admission" is more specific than "college admission."
    - Use conversation history for context if provided.
    - Assume queries are within the Indian education system unless stated otherwise.

    3. **Entity Extraction Rules:**
    - **Institution:** Extract the full official name only if explicitly mentioned (e.g., "IIT Delhi" not "IIT").
    - **Program:** Use standard abbreviations (e.g., "CS" for Computer Science, "B.Tech" for Bachelor of Technology) if provided. Leave empty if not mentioned.
    - Do not infer or invent any entities.

    4. **Clarification Criteria:**
    - Mark the intent as **vague** and set `needs_clarification` to true if:
        - A program-specific query is missing the institution.
        - Ambiguous pronouns are used (e.g., "this college," "that program").
        - The query uses overly generic terms like "admission" without further context.
    - Provide a clear and concise `clarification_prompt` that lists the missing elements (e.g., "Please specify: Institution, Program, Academic level").

    5. **Output Requirements:**
    - Output must be a single, valid JSON object.
    - Follow the exact JSON structure:
        {{
            "intent": "education|greeting|vague|memory|out_of_scope",
            "entities": {{"institution": "string", "program": "string"}},
            "needs_clarification": boolean,
            "clarification_prompt": "string"
        }}
    - Do not include any markdown formatting, commentary, or additional text.
    - Ensure that empty entities are represented as empty strings.

    6. **Handling Conversation History:**
    - When a `History` is provided, use it to resolve ambiguous queries.
    - If history includes previously mentioned institutions or programs, incorporate them into the entity extraction if the current query references them implicitly.

    7. **Common Pitfalls to Avoid:**
    - Do not invent or assume entities that are not explicitly stated.
    - Do not output extra text, commentary, or markdown.
    - Avoid confusing generic academic terms with specific queries.
    - Escape special characters properly.

    **Examples by Category:**

    === Greeting Intent ===
    Query: "Hello, good morning!"
    Expected Output: {{"intent": "greeting", "entities": {{"institution": "", "program": ""}}, "needs_clarification": false, "clarification_prompt": ""}}
    
    Query: "Namaste!"
    Expected Output: {{"intent": "greeting", "entities": {{"institution": "", "program": ""}}, "needs_clarification": false, "clarification_prompt": ""}}

    === Education Intent ===
    Query: "B.Tech CS admission process at IIT Bombay"
    Expected Output: {{"intent": "education", "entities": {{"institution": "IIT Bombay", "program": "B.Tech CS"}}, "needs_clarification": false, "clarification_prompt": ""}}
    
    Query: "NIRF ranking for NIT Trichy"
    Expected Output: {{"intent": "education", "entities": {{"institution": "NIT Trichy", "program": ""}}, "needs_clarification": false, "clarification_prompt": ""}}
    
    Query: "What are the scholarship options at Delhi University?"
    Expected Output: {{"intent": "education", "entities": {{"institution": "Delhi University", "program": ""}}, "needs_clarification": false, "clarification_prompt": ""}}

    === Vague Intent ===
    Query: "Tell me about admissions"
    Expected Output: {{"intent": "vague", "entities": {{"institution": "", "program": ""}}, "needs_clarification": true, "clarification_prompt": "Please specify: Institution, Program, Academic level"}}
    
    Query: "what is this"
    Expected Output: {{"intent": "vague", "entities": {{"institution": "", "program": ""}}, "needs_clarification": true, "clarification_prompt": "Please specify what you are referring to: e.g., which institution or program?"}}

    Query: "What about that engineering college?"
    Expected Output: {{"intent": "vague", "entities": {{"institution": "", "program": "engineering"}}, "needs_clarification": true, "clarification_prompt": "Which specific engineering college are you referring to?"}}
    
    Query: "Give me details on courses"
    Expected Output: {{"intent": "vague", "entities": {{"institution": "", "program": ""}}, "needs_clarification": true, "clarification_prompt": "Please specify: Institution, Program, and Course type"}}

    === Memory Intent ===
    Query: "What did we discuss earlier about IIIT Hyderabad?" [History: "IIIT Hyderabad CSE placement stats"]
    Expected Output: {{"intent": "memory", "entities": {{"institution": "IIIT Hyderabad", "program": "CSE"}}, "needs_clarification": false, "clarification_prompt": ""}}
    
    Query: "Recall our previous conversation regarding admissions"
    Expected Output: {{"intent": "memory", "entities": {{"institution": "", "program": ""}}, "needs_clarification": false, "clarification_prompt": ""}}
    
    Query: "What were the details on fees we talked about?" [History: "Fees for B.Tech at Amity University"]
    Expected Output: {{"intent": "memory", "entities": {{"institution": "Amity University", "program": "B.Tech"}}, "needs_clarification": false, "clarification_prompt": ""}}

    === Out-of-Scope Intent ===
    Query: "How to make biryani?"
    Expected Output: {{"intent": "out_of_scope", "entities": {{"institution": "", "program": ""}}, "needs_clarification": false, "clarification_prompt": ""}}
    
    Query: "What's the weather like today?"
    Expected Output: {{"intent": "out_of_scope", "entities": {{"institution": "", "program": ""}}, "needs_clarification": false, "clarification_prompt": ""}}
    
    Query: "Tell me the latest sports news."
    Expected Output: {{"intent": "out_of_scope", "entities": {{"institution": "", "program": ""}}, "needs_clarification": false, "clarification_prompt": ""}}

    **Current Task:**

    Analyze this query in the context of the conversation history provided:

    Query: {query}
    History: {history}

    **Output Requirements:**
    - Output only the JSON object in the exact format as shown above.
    - Do not include any additional text or markdown.
    - Ensure all special characters are properly escaped.
    """


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


import yaml

with open("follow_up_questions.yaml", "r") as f:
    FOLLOW_UP_CORPUS = yaml.safe_load(f)

class FollowUpManager:
    def __init__(self, corpus: dict):
        self.corpus = corpus

    def get_follow_up(self, query: str) -> str:
        normalized_query = query.strip().lower()
        for key, follow_up in self.corpus.items():
            if key.strip().lower() == normalized_query:
                return follow_up
        return "Could you please provide more details?"


class IntentJudge:
    VALIDATION_PROMPT = """# Intent Validation Task

    You are an expert quality analyst evaluating intent classifications. Analyze this case:

    **Query:** {query}
    **History:** {history}
    **Initial Intent:** {initial_intent}
    **Entities Detected:** {entities}

    **Evaluation Guidelines:**
    1. Verify if intent matches ALL these criteria:
    - Institution/program explicitly mentioned (for education)
    - Contains educational keywords (for education)
    - References previous conversation (memory)
    - Contains non-educational terms (out_of_scope)
    - Lacks specificity (vague)

    2. **Override Rules:**
    - Education → Vague if missing critical details
    - Vague → Education if entities exist in history
    - Out_of_scope → Education if contains "college", "university", "admission" 
    - Memory → Vague if no prior relevant history

    **Output JSON:**
    {{
        "final_intent": "corrected_intent",
        "override_reason": "concise reason for change",
        "needs_clarification": boolean,
        "confidence": 0-1
    }}"""

    def evaluate(self, query: str, history: List[str], 
                initial_intent: str, entities: dict) -> dict:
        prompt = self.VALIDATION_PROMPT.format(
            query=query,
            history=json.dumps(history[-3:]),
            initial_intent=initial_intent,
            entities=json.dumps(entities)
        )
        
        response = ChatBedrock(
            client=bedrock_runtime,
            model_id=Config.CHAT_MODEL
        ).invoke(prompt)
        
        return self._parse_judgement(response.content)

    def _parse_judgement(self, response: str) -> dict:
        try:
            judgement = json.loads(response)
            return {
                "final_intent": judgement["final_intent"],
                "override_reason": judgement.get("override_reason", ""),
                "needs_clarification": judgement.get("needs_clarification", False),
                "confidence": min(max(float(judgement.get("confidence", 0.7)), 0), 1)
            }
        except:
            return {"final_intent": "education", "confidence": 0.5}

# -------------------------------------------------------------------
# ResponseGenerator
# -------------------------------------------------------------------
class ResponseGenerator:
    """Generates the final response using an LLM prompt."""
    def __init__(self):
        self.memory = MemoryManager()
        self.followup_manager = FollowUpManager(FOLLOW_UP_CORPUS)

        self.introduction = """👋 *Hi! I'm Carrers360 - Your Comprehensive Education Assistant* 🎓

        I specialize in providing 360° information about:
        ✅ College Details (IITs, NITs, Private Universities)
        ✅ Course Curriculum & Specializations
        ✅ Admission Processes & Cutoffs
        ✅ Fee Structures & Scholarships
        ✅ Placement Statistics
        ✅ Campus Facilities

        Ask me anything about:
        • Engineering (B.Tech, M.Tech)
        • Medical (MBBS, BDS)
        • Management (MBA, BBA)
        • Science & Arts Programs

        How can I assist you today?"""

    def generate(self, context: dict) -> str:
        if context.get("needs_clarification"):
            return context["clarification_prompt"]


        elif context.get("intent") == "greeting" or context.get("intent") == "out_of_scope":
            pass



        elif context.get("intent") == "vague" and not context.get("clarification_prompt"):
            follow_up = self.followup_manager.get_follow_up(context["query"])
            context["clarification_prompt"] = follow_up
            return follow_up

        
        # print("nsdmmmmmmmmmmmm",context)

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
        self.introduction = """👋 *Hi! I'm Carrers360 chatbot - Your Comprehensive Education Assistant* 🎓

        I specialize in providing 360° information about:
        ✅ College Details (IITs, NITs, Private Universities)
        ✅ Course Curriculum & Specializations
        ✅ Admission Processes & Cutoffs
        ✅ Fee Structures & Scholarships
        ✅ Placement Statistics
        ✅ Campus Facilities

        Ask me anything about:
        • Engineering (B.Tech, M.Tech)
        • Medical (MBBS, BDS)
        • Management (MBA, BBA)
        • Science & Arts Programs

        How can I assist you today?"""


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
                "vauge": "generate_response",
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

        ambiguous_phrases = ["what is this", "what's this", "explain this", "what about this"]
        query_lower = state["query"].strip().lower()
        # if any(phrase in query_lower for phrase in ambiguous_phrases) < 3:
        #     detection_result["intent"] = "vague"


        state.update(detection_result)

        # print("out_of_scope++++++++++++",state["force_introduction"])
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
        if state.get("intent") == "out_of_scope":
            state["response"] = self.introduction
        else:
            state["response"] = response_text

        # Update session
        thread_id = state["thread_id"]
        session = session_manager.retrieve_session(thread_id)

        # Append user query & bot response to conversation history
        session.setdefault("history", []).append(state["query"])
        session.setdefault("history", []).append(response_text)
        session_manager.update_session(thread_id, "history", session["history"])

        return state


    def _handle_out_of_scope(self, state: dict):
    # Check if the query is out-of-scope
        if state.get("intent") == "out_of_scope":
            state["force_introduction"] = True

            # Instead of a dynamic, topic-specific message, provide a default introduction.
            # introduction = (
            #     "🎓 *Edu360 - Your Complete Education Advisor* 📚\n\n"
            #     "I specialize in Indian higher education with 360° insights on:\n"
            #     "✅ Top Institutions (IITs, NITs, IIITs, Central Universities)\n"
            #     "✅ 1000+ Courses (Engineering, Medicine, Management, Arts)\n"
            #     "✅ Admission Processes & Entrance Exams\n"
            #     "✅ Fee Structures & Scholarships\n"
            #     "✅ Campus Facilities & Placements\n\n"
            #     "Ask me about:\n"
            #     "• IIT Delhi CS admissions\n"
            #     "• NIRF rankings\n"
            #     "• JEE Advanced cutoffs\n"
            #     "• College comparisons\n"
            #     "• Course curricula\n\n"
            #     "How can I assist you today?"
            # )
            # state["response"] = introduction
        return state

    def _route_intent(self, state: dict):
        """
        Route next node based on detected intent.
        If unknown, fallback to 'default'.
        """
        print(">>>>>>>>>>>>>>>>>> intent understanding", state.get("intent", "default"))
        
        intent = state.get("intent", "default").lower()

        if intent == "out_of_scope":
            state["force_introduction"] = True

            # state = self._handle_out_of_scope(state)
            # return "generate_response"



        intent = state.get("intent", "default").lower()

        print("intent+++++++",intent)
        if intent not in {"education", "greeting", "memory", "out_of_scope"}:
            intent = "default"
        return intent



        # if intent in {"vauge", "vague"}:
        #     return "education"
        # return state.get("intent", "default")

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
        "can you recall What we discussed earlier",
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

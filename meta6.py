# # """
# # AWS Bedrock Educational Chatbot - Enterprise Grade Version
# # """

# # import os
# # import json
# # import re
# # import logging
# # from datetime import datetime
# # from typing import List, Dict, Optional, Tuple
# # from dotenv import load_dotenv
# # from langgraph.graph import StateGraph, END
# # from langgraph.checkpoint.base import BaseCheckpointSaver
# # from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# # # AWS Components
# # import boto3
# # from botocore.exceptions import ClientError
# # from langchain_aws import BedrockEmbeddings, ChatBedrock
# # from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
# # from langchain_core.vectorstores import InMemoryVectorStore

# # # Core Components
# # from langchain_core.documents import Document
# # from langchain_core.vectorstores import VectorStore
# # from pydantic import BaseModel, ValidationError

# # # Configuration
# # load_dotenv()
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # class Config:
# #     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
# #     AWS_REGION = os.getenv("BEDROCK_REGION_NAME", "us-west-2")
# #     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
# #     PRIMARY_MODEL = "meta.llama3-70b-instruct-v1:0"
# #     FALLBACK_MODEL = "amazon.titan-embed-text-v2:0"
# #     MEMORY_SEARCH_K = 7
# #     MAX_HISTORY = 10
# #     INTENT_CONFIDENCE_THRESHOLD = 0.7
# #     MODEL_TIMEOUT = 30

# # # AWS Clients
# # session = boto3.Session(region_name=Config.AWS_REGION)
# # bedrock_runtime = session.client("bedrock-runtime")
# # dynamodb = session.resource("dynamodb")
# # feedback_table = dynamodb.Table("ChatbotFeedback")

# # # Pydantic Models
# # class IntentResult(BaseModel):
# #     primary_intent: str
# #     sub_intent: Optional[str] = None
# #     confidence: float
# #     entities: Dict[str, str]
# #     needs_clarification: bool
# #     clarification_prompt: Optional[str]
# #     fallback_triggered: bool = False

# # class ConversationContext(BaseModel):
# #     thread_id: str
# #     current_query: str
# #     history: List[str]
# #     summary: Optional[str]
# #     carried_entities: Dict[str, str]
# #     knowledge_gaps: List[str]

# # # Helper Decorators
# # def hybrid_retry(func):
# #     @retry(
# #         stop=stop_after_attempt(3),
# #         wait=wait_exponential(multiplier=1, min=2, max=10),
# #         retry=retry_if_exception_type((ClientError, TimeoutError)),
# #     )
# #     async def wrapper(*args, **kwargs):
# #         return await func(*args, **kwargs)
# #     return wrapper

# # class EnhancedMemoryManager:
# #     """Advanced memory with temporal awareness and entity tracking"""
    
# #     def __init__(self):
# #         self.vector_store = InMemoryVectorStore(
# #             BedrockEmbeddings(
# #                 client=bedrock_runtime,
# #                 model_id=Config.EMBED_MODEL
# #             )
# #         )
# #         self.entity_tracker = {}
        
# #     def store_conversation(self, text: str, metadata: dict, entities: dict):
# #         doc = Document(
# #             page_content=text,
# #             metadata={
# #                 "timestamp": datetime.utcnow().isoformat(),
# #                 "entities": json.dumps(entities),
# #                 **metadata
# #             }
# #         )
# #         self.vector_store.add_documents([doc])
# #         self._update_entity_tracker(entities)
        
# #     def recall_context(self, query: str, filters: dict, k: int = 5) -> Tuple[List[str], dict]:
# #         try:
# #             # Temporal-weighted vector search
# #             docs = self.vector_store.similarity_search(query, k=k*2)
# #             docs = sorted(docs, 
# #                         key=lambda x: (
# #                             -self._calculate_temporal_score(x.metadata["timestamp"]),
# #                             -self._entity_match_score(x.metadata.get("entities", "{}"))
# #                         ))[:k]
            
# #             carried_entities = self._get_relevant_entities(query)
# #             return [d.page_content for d in docs], carried_entities
            
# #         except Exception as e:
# #             logger.error(f"Memory recall error: {str(e)}")
# #             return [], {}
    
# #     def _calculate_temporal_score(self, timestamp: str) -> float:
# #         age = datetime.utcnow() - datetime.fromisoformat(timestamp)
# #         return max(0, 1 - age.days/30)  # Linear decay over 30 days
    
# #     def _entity_match_score(self, entity_json: str) -> float:
# #         try:
# #             entities = json.loads(entity_json)
# #             return sum(1 for k, v in entities.items() if v in self.entity_tracker.get(k, []))
# #         except:
# #             return 0
            
# #     def _update_entity_tracker(self, entities: dict):
# #         for key, value in entities.items():
# #             if value:
# #                 self.entity_tracker.setdefault(key, set()).add(value.lower())
    
# #     def _get_relevant_entities(self, query: str) -> dict:
# #         return {
# #             key: max(values, key=lambda v: len(v)) 
# #             for key, values in self.entity_tracker.items()
# #             if any(v in query.lower() for v in values)
# #         }

# # class MultiLayerIntentDetector:
# #     """Advanced intent classification with confidence scoring"""
    
# #     SYSTEM_PROMPT = """You are an expert intent classifier for an educational chatbot. Analyze the query and conversation history to determine:
# # 1. Primary intent (education, administration, resources, greeting, or other)
# # 2. Sub-intent specific to university operations
# # 3. Confidence score (0-1)
# # 4. Relevant entities
# # 5. Whether clarification is needed

# # Respond ONLY with valid JSON:"""
    
# #     PROMPT_TEMPLATE = {
# #         "primary_intent": "education",
# #         "sub_intent": "admission_process",
# #         "confidence": 0.95,
# #         "entities": {
# #             "institution": "IIT Delhi",
# #             "program": "Computer Science"
# #         },
# #         "needs_clarification": False,
# #         "clarification_prompt": False
# #     }

# #     @hybrid_retry
# #     async def detect_intent(self, context: ConversationContext) -> IntentResult:
# #         try:
# #             prompt = self._build_prompt(context)
# #             response = await self._call_model_with_fallback(prompt)
# #             return self._parse_response(response)
# #         except Exception as e:
# #             logger.error(f"Intent detection failed: {str(e)}")
# #             return IntentResult(
# #                 primary_intent="education",
# #                 confidence=0.0,
# #                 entities={},
# #                 needs_clarification=True,
# #                 clarification_prompt="I'm having trouble understanding. Could you please rephrase your question?"
# #             )

# #     def _build_prompt(self, context: ConversationContext) -> str:
# #         history = context.summary or "\n".join(context.history[-3:])
# #         return f"{self.SYSTEM_PROMPT}\n\nQuery: {context.current_query}\nHistory: {history}"

# #     async def _call_model_with_fallback(self, prompt: str) -> dict:
# #         try:
# #             return await self._call_primary_model(prompt)
# #         except Exception as e:
# #             logger.warning(f"Primary model failed, using fallback: {str(e)}")
# #             return await self._call_fallback_model(prompt)

# #     async def _call_primary_model(self, prompt: str) -> dict:
# #         client = ChatBedrock(
# #             client=bedrock_runtime,
# #             model_id=Config.PRIMARY_MODEL,
# #             model_kwargs={"temperature": 0.2}
# #         )
# #         response = await client.ainvoke(prompt)
# #         return json.loads(response.content)

# #     async def _call_fallback_model(self, prompt: str) -> dict:
# #         client = ChatBedrock(
# #             client=bedrock_runtime,
# #             model_id=Config.FALLBACK_MODEL,
# #             model_kwargs={"temperature": 0.1}
# #         )
# #         response = await client.ainvoke(prompt)
# #         return json.loads(response.content)

# #     def _parse_response(self, raw_response: dict) -> IntentResult:
# #         try:
# #             return IntentResult(
# #                 primary_intent=raw_response.get("primary_intent", "education"),
# #                 sub_intent=raw_response.get("sub_intent"),
# #                 confidence=raw_response.get("confidence", 0.0),
# #                 entities=raw_response.get("entities", {}),
# #                 needs_clarification=raw_response.get("needs_clarification", False),
# #                 clarification_prompt=raw_response.get("clarification_prompt")
# #             )
# #         except ValidationError as e:
# #             logger.error(f"Intent validation error: {str(e)}")
# #             return IntentResult(
# #                 primary_intent="education",
# #                 confidence=0.0,
# #                 entities={},
# #                 needs_clarification=True
# #             )

# # class SelfHealingResponseGenerator:
# #     """Robust response generation with automatic fallback"""
    
# #     def __init__(self):
# #         self.memory = EnhancedMemoryManager()
# #         self.feedback_handler = FeedbackHandler()
    
# #     @hybrid_retry
# #     async def generate_response(self, context: ConversationContext, intent: IntentResult) -> Tuple[str, dict]:
# #         try:
# #             if intent.needs_clarification:
# #                 return intent.clarification_prompt, {}
                
# #             knowledge = await self._retrieve_knowledge(context, intent)
# #             prompt = self._build_prompt(context, intent, knowledge)
            
# #             try:
# #                 response = await self._call_primary_model(prompt)
# #             except Exception as e:
# #                 response = await self._call_fallback_model(prompt)
                
# #             self._store_interaction(context, intent, response, knowledge)
# #             self._check_knowledge_gaps(knowledge, context)
            
# #             return response, knowledge
            
# #         except Exception as e:
# #             logger.error(f"Response generation failed: {str(e)}")
# #             return self._generate_fallback_response(context), {}

# #     async def _retrieve_knowledge(self, context: ConversationContext, intent: IntentResult) -> dict:
# #         knowledge = {"documents": [], "memories": []}
        
# #         try:
# #             # Knowledge Base Retrieval
# #             if Config.BEDROCK_KB_ID:
# #                 retriever = AmazonKnowledgeBasesRetriever(
# #                     knowledge_base_id=Config.BEDROCK_KB_ID,
# #                     retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}}
# #                 )
# #                 knowledge["documents"] = [doc.page_content async for doc in retriever.ainvoke(context.current_query)]
            
# #             # Long-term Memory Recall
# #             memory_results, carried_entities = self.memory.recall_context(
# #                 context.current_query,
# #                 {"thread_id": context.thread_id},
# #                 Config.MEMORY_SEARCH_K
# #             )
# #             knowledge["memories"] = memory_results
# #             context.carried_entities.update(carried_entities)
            
# #         except Exception as e:
# #             logger.error(f"Knowledge retrieval error: {str(e)}")
            
# #         return knowledge

# #     def _build_prompt(self, context: ConversationContext, intent: IntentResult, knowledge: dict) -> str:
# #         components = [
# #             "Generate an expert educational response following these guidelines:",
# #             f"- Query: {context.current_query}",
# #             f"- Intent: {intent.primary_intent} ({intent.sub_intent})",
# #             f"- Conversation Summary: {context.summary}",
# #             "- Retrieved Documents:",
# #             *knowledge["documents"][:3],
# #             "- Relevant Memories:",
# #             *knowledge["memories"][:2],
# #             "- Style: Professional yet approachable, cite sources when available",
# #             "- Length: 2-3 concise paragraphs",
# #             "- Safety: Do not disclose sensitive information"
# #         ]
# #         return "\n".join(components)

# #     async def _call_primary_model(self, prompt: str) -> str:
# #         client = ChatBedrock(
# #             client=bedrock_runtime,
# #             model_id=Config.PRIMARY_MODEL,
# #             model_kwargs={"temperature": 0.5}
# #         )
# #         response = await client.ainvoke(prompt)
# #         return response.content

# #     async def _call_fallback_model(self, prompt: str) -> str:
# #         client = ChatBedrock(
# #             client=bedrock_runtime,
# #             model_id=Config.FALLBACK_MODEL,
# #             model_kwargs={"temperature": 0.3}
# #         )
# #         response = await client.ainvoke(prompt)
# #         return response.content

# #     def _store_interaction(self, context: ConversationContext, intent: IntentResult, response: str, knowledge: dict):
# #         self.memory.store_conversation(
# #             text=f"Q: {context.current_query}\nA: {response}",
# #             metadata={
# #                 "thread_id": context.thread_id,
# #                 "intent": intent.primary_intent,
# #                 "sub_intent": intent.sub_intent
# #             },
# #             entities=intent.entities
# #         )
        
# #     def _check_knowledge_gaps(self, knowledge: dict, context: ConversationContext):
# #         if not knowledge["documents"] and not knowledge["memories"]:
# #             self.feedback_handler.log_knowledge_gap(
# #                 context.thread_id,
# #                 context.current_query,
# #                 "No relevant information found"
# #             )

# #     def _generate_fallback_response(self, context: ConversationContext) -> str:
# #         return f"I'm experiencing technical difficulties. Please try again later. (Reference: {context.thread_id})"

# # class FeedbackHandler:
# #     """Active learning and continuous improvement"""
    
# #     def log_feedback(self, thread_id: str, query: str, response: str, rating: int):
# #         try:
# #             feedback_table.put_item(
# #                 Item={
# #                     "thread_id": thread_id,
# #                     "timestamp": datetime.utcnow().isoformat(),
# #                     "query": query,
# #                     "response": response,
# #                     "rating": rating,
# #                     "reviewed": False
# #                 }
# #             )
# #         except Exception as e:
# #             logger.error(f"Feedback logging failed: {str(e)}")
    
# #     def log_knowledge_gap(self, thread_id: str, query: str, gap_details: str):
# #         try:
# #             feedback_table.put_item(
# #                 Item={
# #                     "thread_id": thread_id,
# #                     "timestamp": datetime.utcnow().isoformat(),
# #                     "type": "knowledge_gap",
# #                     "query": query,
# #                     "details": gap_details,
# #                     "resolved": False
# #                 }
# #             )
# #         except Exception as e:
# #             logger.error(f"Knowledge gap logging failed: {str(e)}")

# # class EducationWorkflow:
# #     """Complete stateful conversation workflow"""
    
# #     def __init__(self):
# #         self.memory = EnhancedMemoryManager()
# #         self.intent_detector = MultiLayerIntentDetector()
# #         self.response_generator = SelfHealingResponseGenerator()
# #         self.feedback_handler = FeedbackHandler()
# #         self.graph = self._build_state_graph()

# #     def _build_state_graph(self):
# #         workflow = StateGraph(ConversationContext)
        
# #         # Define workflow nodes
# #         workflow.add_node("initialize_context", self._initialize_context)
# #         workflow.add_node("summarize_history", self._summarize_history)
# #         workflow.add_node("detect_intent", self._detect_intent)
# #         workflow.add_node("retrieve_knowledge", self._retrieve_knowledge)
# #         workflow.add_node("generate_response", self._generate_response)
# #         workflow.add_node("handle_feedback", self._handle_feedback)
        
# #         # Build workflow
# #         workflow.set_entry_point("initialize_context")
        
# #         workflow.add_edge("initialize_context", "summarize_history")
# #         workflow.add_edge("summarize_history", "detect_intent")
# #         workflow.add_conditional_edges(
# #             "detect_intent",
# #             self._route_intent,
# #             {
# #                 "clarify": "generate_response",
# #                 "continue": "retrieve_knowledge"
# #             }
# #         )
# #         workflow.add_edge("retrieve_knowledge", "generate_response")
# #         workflow.add_edge("generate_response", "handle_feedback")
# #         workflow.add_edge("handle_feedback", END)
        
# #         return workflow.compile(checkpointer=MemorySaver())

# #     async def _initialize_context(self, state: ConversationContext):
# #         state.history = state.history[-Config.MAX_HISTORY:]
# #         return state

# #     async def _summarize_history(self, state: ConversationContext):
# #         if len(state.history) > 4:
# #             state.summary = await self._generate_summary(state.history)
# #         return state

# #     async def _generate_summary(self, history: List[str]) -> str:
# #         client = ChatBedrock(
# #             client=bedrock_runtime,
# #             model_id=Config.FALLBACK_MODEL,
# #             model_kwargs={"temperature": 0.1}
# #         )
# #         response = await client.ainvoke(
# #             f"Summarize this conversation history in 3 sentences:\n{'\n'.join(history)}"
# #         )
# #         return response.content

# #     async def _detect_intent(self, state: ConversationContext):
# #         state.intent_result = await self.intent_detector.detect_intent(state)
# #         return state

# #     def _route_intent(self, state: ConversationContext):
# #         if state.intent_result.needs_clarification:
# #             return "clarify"
# #         return "continue"

# #     async def _retrieve_knowledge(self, state: ConversationContext):
# #         state.knowledge = await self.response_generator._retrieve_knowledge(state, state.intent_result)
# #         return state

# #     async def _generate_response(self, state: ConversationContext):
# #         response, knowledge = await self.response_generator.generate_response(state, state.intent_result)
# #         state.response = response
# #         state.knowledge = knowledge
# #         return state

# #     async def _handle_feedback(self, state: ConversationContext):
# #         if state.intent_result.confidence < 0.5:
# #             self.feedback_handler.log_knowledge_gap(
# #                 state.thread_id,
# #                 state.current_query,
# #                 f"Low confidence response: {state.response}"
# #             )
# #         return state

# # # Example Usage
# # if __name__ == "__main__":
# #     async def main():
# #         workflow = EducationWorkflow()
# #         thread_id = "demo_thread6"
        
# #         queries = [
# #             "Hi there!",
# #             "What's the IIT Delhi CS admission process?",
# #             "What about eligibility criteria?",
# #             "What did we discuss earlier regarding IIT Delhi?"
# #         ]
        
# #         context = ConversationContext(
# #             thread_id=thread_id,
# #             current_query="",
# #             history=[],
# #             summary=None,
# #             carried_entities={},
# #             knowledge_gaps=[]
# #         )
        
# #         for query in queries:
# #             print(f"\nUser: {query}")
# #             context.current_query = query
# #             context = await workflow.graph.ainvoke(context)
            
# #             print(f"Bot: {context.response}")
# #             context.history.extend([
# #                 f"User: {query}",
# #                 f"Assistant: {context.response}"
# #             ])

# #     import asyncio
# #     asyncio.run(main())



# """
# AWS Bedrock Educational Chatbot with Vector Memory
# """

# import os
# import json
# import logging
# from datetime import datetime
# from typing import List, Dict, Optional
# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_aws import ChatBedrock, BedrockEmbeddings
# from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
# import boto3
# from langchain_core.vectorstores import InMemoryVectorStore

# # Configuration
# load_dotenv()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class Config:
#     AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
#     BEDROCK_KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
#     EMBED_MODEL = "amazon.titan-embed-text-v2:0"
#     CHAT_MODEL = "meta.llama3-70b-instruct-v1:0"
#     MEMORY_SEARCH_K = 5
#     MAX_HISTORY = 10

# # AWS Clients
# bedrock_runtime = boto3.client("bedrock-runtime", region_name="ap-south-1")

# class VectorMemoryManager:
#     """Long-term memory with vector-based recall"""
    
#     def __init__(self):
#         self.embeddings = BedrockEmbeddings(
#             client=bedrock_runtime,
#             model_id="amazon.titan-embed-text-v2:0"
#         )
#         self.vector_store = InMemoryVectorStore(self.embeddings)
#         self.conversation_index = {}

#     def store_conversation(self, query: str, response: str, metadata: dict):
#         doc = Document(
#             page_content=f"Q: {query}\nA: {response}",
#             metadata={
#                 "timestamp": datetime.utcnow().isoformat(),
#                 "thread_id": metadata.get("thread_id", "global"),
#                 **metadata
#             }
#         )
#         self.vector_store.add_documents([doc])
#         self._update_index(metadata.get("thread_id"), doc.page_content)

#     def recall_conversations(self, query: str, thread_id: str) -> List[str]:
#         try:
#             docs = self.vector_store.similarity_search(
#                 query=query,
#                 k=Config.MEMORY_SEARCH_K,
#                 filter=lambda doc: doc.metadata.get("thread_id") == thread_id
#             )
#             return [d.page_content for d in docs]
#         except Exception as e:
#             logger.error(f"Memory recall error: {str(e)}")
#             return []

#     def _update_index(self, thread_id: str, content: str):
#         self.conversation_index.setdefault(thread_id, []).append(content)

# class IntentClassifier:
#     """Advanced intent classification using Claude-3"""
    
    
#     SYSTEM_PROMPT = """
#             You are a specialized intent classification system for an educational chatbot focused on higher education topics. 
#             Your task is to analyze user queries and classify them into specific intent categories with high precision.

#             ### Core Intent Categories:

#             1. **GREETING**
#             - **Primary Characteristics**:
#                 - Contains common greeting words/phrases (hi, hello, hey, good morning/evening/afternoon).
#                 - No substantial question or request follows.
#                 - Usually brief (1-3 words).
#             - **Positive Examples**:
#                 - "Hello there!"
#                 - "Good morning"
#                 - "Hey, how are you?"
#                 - "Hi"
#             - **Negative Examples**:
#                 - "Hello, what are the admission requirements?" (This is Greeting + Education).
#                 - "Hi, what's the weather like?" (This is Greeting + Out-of-Scope).

#             2. **EDUCATION_DOMAIN**
#             - **Primary Characteristics**:
#                 - Questions about higher education institutions, programs, or processes.
#                 - Contains education-specific keywords (college, university, degree, course, admission, etc.).
#                 - Seeks information about academic topics.
#             - **Key Topic Indicators**:
#                 - Admissions and Applications.
#                 - Course Information.
#                 - Degree Programs.
#                 - College/University Details.
#                 - Academic Requirements.
#                 - Scholarships and Financial Aid.
#                 - Campus Life.
#                 - Career Prospects.
#                 - Research Programs.
#                 - Faculty Information.
#             - **Positive Examples**:
#                 - "What are the admission requirements for Harvard?"
#                 - "Tell me about computer science programs in top universities."
#                 - "How much is the tuition fee for MBA programs?"
#                 - "What scholarships are available for international students?"
#                 - "Can you explain the different types of engineering degrees?"
#             - **Negative Examples**:
#                 - "How do I make pasta?" (Out-of-Scope).
#                 - "What's the weather like?" (Out-of-Scope).
#                 - "Hello, which universities offer AI programs?" (Greeting + Education).

#             3. **GREETING_EDUCATION**
#             - **Primary Characteristics**:
#                 - Combines a greeting with an education-related question.
#                 - Must have BOTH:
#                     1. Clear greeting element.
#                     2. Valid education domain question.
#             - **Positive Examples**:
#                 - "Hi, what are the best medical schools in the US?"
#                 - "Hello, can you tell me about MBA programs?"
#                 - "Good morning, I need information about college admissions."
#                 - "Hey there, what scholarships are available?"
#             - **Negative Examples**:
#                 - "Hello" (Greeting only).
#                 - "What are the best universities?" (Education only).
#                 - "Hi, what's today's date?" (Greeting + Out-of-Scope).

#             4. **OUT_OF_SCOPE**
#             - **Primary Characteristics**:
#                 - Questions unrelated to higher education.
#                 - Topics outside the educational domain.
#                 - General knowledge questions.
#             - **Common Categories**:
#                 - Weather.
#                 - Cooking/Recipes.
#                 - Entertainment.
#                 - Sports.
#                 - General News.
#                 - Personal Advice.
#                 - Technical Support.
#                 - Business/Finance (non-educational).
#             - **Positive Examples**:
#                 - "What's the best pizza recipe?"
#                 - "How do I invest in stocks?"
#                 - "What's the weather forecast?"
#                 - "Who won the Oscar for best picture?"
#             - **Negative Examples**:
#                 - "What are the best colleges?" (Education).
#                 - "Hi, how do I cook pasta?" (Greeting + Out-of-Scope).

#             5. **GREETING_OUT_OF_SCOPE**
#             - **Primary Characteristics**:
#                 - Combines greeting with a non-education question.
#                 - Must have BOTH:
#                     1. Clear greeting element.
#                     2. Question unrelated to education.
#             - **Positive Examples**:
#                 - "Hello, what's the weather today?"
#                 - "Hi, can you recommend a good restaurant?"
#                 - "Good morning, what's the latest news?"
#                 - "Hey, how do I fix my printer?"
#             - **Negative Examples**:
#                 - "Hello, what universities are in London?" (Greeting + Education).
#                 - "What's the weather like?" (Out-of-Scope only).

#             6. **HARMFUL**
#             - **Primary Characteristics**:
#                 - Content related to:
#                     - Violence or self-harm.
#                     - Illegal activities.
#                     - Academic dishonesty.
#                     - Cybercrime.
#                     - Hate speech.
#                     - Harassment.
#                 - **Key Points**:
#                     - Always classify as HARMFUL if ANY harmful content is present.
#                     - Greeting + harmful content still classifies as HARMFUL.
#                     - Education-related harmful content (cheating, plagiarism) is still HARMFUL.
#             - **Positive Examples**:
#                 - "How can I cheat on my exam?"
#                 - "What's the best way to hack a university database?"
#                 - "Hello, how do I plagiarize my thesis?"
#                 - "Ways to harm myself."

#             ### MEMORY_QUERY
#             - **Primary Characteristics**:
#                 - References previous conversations or information.
#                 - Asks about earlier mentioned topics.
#                 - Requests historical context.
#             - **Key Indicators**:
#                 - "Remember when..."
#                 - "Earlier you said..."
#                 - "Last time we discussed..."
#                 - "Going back to..."
#                 - "As mentioned before..."
#                 - "What did you say about..."
#                 - "Can you remind me..."
#             - **Positive Examples**:
#                 - "What did we discuss last time about Harvard?"
#                 - "Can you remind me of the admission requirements you mentioned?"
#                 - "Earlier you told me about scholarships, can you repeat that?"

#             ### VAGUE_QUERY
#             - **Primary Characteristics**:
#                 - Unclear or incomplete questions.
#                 - Missing crucial context.
#                 - Ambiguous references.
#                 - Single word queries.
#             - **Subtypes**:
#                 1. **VAGUE_EDUCATION**:
#                     - Examples:
#                         - "What about colleges?"
#                         - "Studies?"
#                         - "Tell me more."
#                 2. **VAGUE_OUT_OF_SCOPE**:
#                     - Examples:
#                         - "What do you think?"
#                         - "How does it work?"

#             ### Output Format:
#             {{
#                 "intent": "string",
#                 "confidence": float,
#                 "reasoning": "string",
#                 "needs_clarification": boolean,
#                 "suggested_prompts": ["string", ...]
#             }}

#             Classify this query: "{query}"
#     """


    
#     def __init__(self):
#         self.client = ChatBedrock(
#             client=bedrock_runtime,
#             model_id=Config.CHAT_MODEL,
#             model_kwargs={"temperature": 0.0}
#         )

#     async def classify(self, query: str, history: List[str]) -> dict:
#         prompt = self._build_prompt(query, history)
#         try:
#             response = await self.client.ainvoke(prompt)
#             return self._parse_response(response.content)
#         except Exception as e:
#             logger.error(f"Intent classification failed: {str(e)}")
#             return self._fallback_response()

#     def _build_prompt(self, query: str, history: List[str]) -> str:
#         return f"""\n\n{self.SYSTEM_PROMPT}\n\nQuery: {query}\nHistory: {", ".join(history[-3:])}"""

#     def _parse_response(self, response: str) -> dict:
#         try:
#             json_str = response[response.index("{"):response.rindex("}")+1]
#             result = json.loads(json_str)
#             return {
#                 "intent": result.get("intent", "EDUCATION_DOMAIN"),
#                 "confidence": result.get("confidence", 0.0),
#                 "needs_clarification": result.get("needs_clarification", False),
#                 "suggested_prompts": result.get("suggested_prompts", [])
#             }
#         except (json.JSONDecodeError, KeyError):
#             return self._fallback_response()

#     def _fallback_response(self) -> dict:
#         return {
#             "intent": "EDUCATION_DOMAIN",
#             "confidence": 0.0,
#             "needs_clarification": True,
#             "suggested_prompts": ["Could you please rephrase your question?"]
#         }

# class ContentSafetyChecker:
#     """Harmful content detection"""
    
#     def __init__(self):
#         self.client = ChatBedrock(
#             client=bedrock_runtime,
#             model_id=Config.CHAT_MODEL,
#             model_kwargs={"temperature": 0.0}
#         )

#     async def is_unsafe(self, text: str) -> bool:
#         prompt = f"""Analyze this text for harmful content. Respond with JSON:
#         {{
#             "unsafe": boolean,
#             "reason": string
#         }}
#         Text: {text}"""
        
#         try:
#             response = await self.client.ainvoke(prompt)
#             result = json.loads(response.content)
#             return result.get("unsafe", False)
#         except Exception as e:
#             logger.error(f"Safety check failed: {str(e)}")
#             return False

# class EducationChatbot:
#     """Complete chatbot workflow"""
    
#     def __init__(self):
#         self.memory = VectorMemoryManager()
#         self.intent_classifier = IntentClassifier()
#         self.safety_checker = ContentSafetyChecker()
#         self.knowledge_retriever = AmazonKnowledgeBasesRetriever(
#             knowledge_base_id=Config.BEDROCK_KB_ID,
#             retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}}
#         )

#     async def process_query(self, query: str, thread_id: str = "default") -> str:
#         # Step 1: Safety check
#         if await self.safety_checker.is_unsafe(query):
#             return "I cannot assist with that request."
            
#         # Step 2: Retrieve conversation context
#         history = self.memory.recall_conversations(query, thread_id)
        
#         # Step 3: Classify intent
#         intent = await self.intent_classifier.classify(query, history)
        
#         # Handle special intents
#         if intent["needs_clarification"]:
#             return "\n".join(intent["suggested_prompts"])
            
#         # Step 4: Retrieve knowledge
#         knowledge = [doc.page_content async for doc in self.knowledge_retriever.ainvoke(query)]
        
#         # Step 5: Generate response
#         response = await self._generate_response(query, intent, knowledge, history)
        
#         # Step 6: Store interaction
#         self.memory.store_conversation(
#             query=query,
#             response=response,
#             metadata={
#                 "thread_id": thread_id,
#                 "intent": intent["intent"],
#                 "timestamp": datetime.utcnow().isoformat()
#             }
#         )
        
#         return response

#     async def _generate_response(self, query: str, intent: dict, knowledge: List[str], history: List[str]) -> str:
#         client = ChatBedrock(client=bedrock_runtime, model_id=Config.CHAT_MODEL)
#         prompt = f"""Generate an educational response following these guidelines:
#         - Query: {query}
#         - Intent: {intent['intent']}
#         - Knowledge: {knowledge[:2]}
#         - History: {history[:2]}
#         - Style: Professional and helpful"""
        
#         return (await client.ainvoke(prompt)).content

# # Example Usage
# async def main():
#     chatbot = EducationChatbot()
    
#     # Test conversation
#     thread_id = "test_session_123"
#     queries = [
#         "Hello! What's the admission process for MIT Computer Science?",
#         "What scholarships are available?",
#         "Can you repeat that last part about scholarships?",
#         "How do I hack into the university system?"
#     ]
    
#     for query in queries:
#         print(f"\nUser: {query}")
#         response = await chatbot.process_query(query, thread_id)
#         print(f"Assistant: {response}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())



######################
## full working code
######################
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
                    knowledge_base_id="BVWGHMKJOQ",
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
import json
import re
from langchain_core.tools import tool
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import Dict, Optional
from dataclasses import dataclass
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.graphs import Neo4jGraph
# from IPython.display import display
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from langchain.schema import AIMessage as LCAIMessage  # Import LangChain AIMessage
import uuid
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Literal, Optional,Dict,Tuple
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnableConfig
import getpass
import os
import tiktoken
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import boto3
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from typing import Sequence
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    trim_messages,
    BaseMessage,
)
from enum import Enum
import pprint



BEDROCK_SERVICE_NAME = os.getenv('BEDROCK_SERVICE_NAME')
BEDROCK_REGION_NAME = os.getenv('BEDROCK_REGION_NAME')
BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID')
BEDROCK_EMBEDDING_MODEL_ID = os.getenv('BEDROCK_EMBEDDING_MODEL_ID')
CHAT_MODEL_ID = os.getenv('CHAT_MODEL_ID')



bedrock_client = boto3.client(service_name=BEDROCK_SERVICE_NAME, region_name=BEDROCK_REGION_NAME)  # Specify your region

embeddings = BedrockEmbeddings(client=bedrock_client, model_id=BEDROCK_MODEL_ID)

recall_vector_store = InMemoryVectorStore(BedrockEmbeddings(client=bedrock_client, model_id=BEDROCK_MODEL_ID))


model = ChatBedrock(
    model_id=CHAT_MODEL_ID,
    model_kwargs={
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9
    },
    region_name=BEDROCK_REGION_NAME
)

llm = model
tokenizer = tiktoken.get_encoding("cl100k_base")

encoding = tiktoken.get_encoding("cl100k_base")
# tokenizer = tiktoken.encoding_for_model("meta.llama3-70b-instruct-v1:0")



class IntentType(Enum):
    GREETING = "greeting"
    GREETING_INTRO = "greeting_introduction"
    SINGLE_QUERY = "single_query"
    MULTI_QUERY = "multi_query"
    MEMORY_QUERY = "memory_query"
    CLARIFICATION = "clarification_query"
    HARMFUL = "harmful_content"
    OUT_OF_SCOPE = "out_of_scope"
    VAGUE_QUERY  = "vauge_query"
    UNKNOWN = "unknown"


class ActionType(Enum):
    RESPOND = "respond"
    CLARIFY = "clarify"
    TERMINATE = "terminate"
    NEXT_NODE = "next_node"
    FALLBACK = "fallback"




class State(MessagesState):
    recall_memories: List[str] = []
    intents: List[str] = []
    query_type: List[str] = []  
    reasoning: str = ""  
    multi_query: List[str] = []  
    single_query: str = ""  
    parameters: Dict[str, str] = {}  
    memory: Dict[str, List[Dict[str, str]]] = {
        "summary": "",  
        "top_k_memories": [],  
        "query_history": [],  
        "response_history": [],  
        "user_name": [],
        "event_history": [],          
        "terminate_reason": None,
        "query": [],
        "intent": [],
        "reasoning": [],
        "follow_up": []
    }
    cypher_statement: str = ""  
    response: str = ""  
    vector_results: List[str] = []  
    graph_results: List[str] = []  
    cypher_errors: List[str] = []  
    terminate_conversation: str = ""
    conversation_history: List[str] = []
    terminate_conversation = False  
    terminate_conversation = False


    intents: List[str] = []     
    
    reasoning: str = ""        
    
    action: str = ""            
                
    bedrock_hypothetical_doc: str = ""
    bedrock_retrieved_docs: List[str] = []
    needs_clarification: bool = False  
    current_intent: str = "UNCLASSIFIED"  
    confidence: float = 0.0  
    suggested_prompts: List[str] = []  





def get_conversation_history(state):
    """
    Safely constructs a conversation history string from state['messages'].

    Args:
        state (dict): The state containing messages.

    Returns:
        str: The concatenated conversation history.
    """
    # Validate that state is a dictionary
    if not isinstance(state, dict):
        raise TypeError("state must be a dictionary.")

    # Ensure state["messages"] is a list
    messages = state.get("messages")
    if not isinstance(messages, list):
        raise TypeError("state['messages'] must be a list.")

    # Safely retrieve the content of each message
    conversation_history = " ".join(
        msg.content if hasattr(msg, "content") else msg.get("content", "")
        for msg in messages
        if isinstance(msg, (dict, object))
    )
    
    return conversation_history



def load_memory(state: State, config: RunnableConfig) -> State:
    """
    Load top-K relevant memories and summarize conversation history.
    """
    # Ensure messages exist
    if not state["messages"]:
        raise ValueError("No messages available in the state to process.")

    # Step 1: Recall top-K memories
    last_message = state["messages"][-1].content
    recall_memories = search_recall_memories.invoke(last_message, config)

    recall_memories = list(set(recall_memories))


    # Store recalled memories
    state["recall_memories"] = recall_memories

    # Initialize memory if missing
    if "memory" not in state:
        state["memory"] = {"summary": "", "top_k_memories": []}

    # Step 2: Summarize conversation history
    conversation_history = " ".join([msg.content for msg in state["messages"]])
    tokens = tokenizer.encode(conversation_history)
    if len(tokens) > 2048:
        print("Warning: Conversation history truncated due to token limit.")
        conversation_history = tokenizer.decode(tokens[:2048])

    # Generate a summary using the LLM
    prompt = f"""
    Summarize the following conversation in a concise manner:
    {conversation_history}
    """
    summary = llm.predict(prompt)

    # Update memory
    state["memory"]["summary"] = summary
    state["memory"]["top_k_memories"] = recall_memories


    return state



@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """
    Save memory to vectorstore for later semantic retrieval.
    Avoid redundant storage by checking if the memory already exists.
    """
    user_id = config["configurable"].get("user_id")
    if not user_id:
        raise ValueError("User ID is required.")
    
    # Avoid saving redundant memories
    existing_memories = [doc.page_content for doc in recall_vector_store.similarity_search(memory, k=3)]
    if memory in existing_memories:
        print("Memory already exists. Skipping save.")
        return memory

    document = Document(page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id})
    recall_vector_store.add_documents([document])
    print("addeded")
    return memory



@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """
    Search for relevant memories in the vectorstore.
    """
    user_id = config["configurable"].get("user_id")
    if not user_id:
        raise ValueError("User ID is required.")

    def _filter(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(query, k=3, filter=_filter)
    return [doc.page_content for doc in documents]


tools = [save_recall_memory, search_recall_memories]


model_with_tools = model.bind_tools(tools)



INTENT_CLASSIFICATION_PROMPT_TEMPLATE = """
        You are a specialized intent classification system for an educational chatbot focused on higher education topics. 
        Your task is to analyze user queries and classify them into specific intent categories with high precision.

        ### Core Intent Categories:

        1. **GREETING**
        - **Primary Characteristics**:
            - Contains common greeting words/phrases (hi, hello, hey, good morning/evening/afternoon).
            - No substantial question or request follows.
            - Usually brief (1-3 words).
        - **Positive Examples**:
            - "Hello there!"
            - "Good morning"
            - "Hey, how are you?"
            - "Hi"
        - **Negative Examples**:
            - "Hello, what are the admission requirements?" (This is Greeting + Education).
            - "Hi, what's the weather like?" (This is Greeting + Out-of-Scope).

        2. **EDUCATION_DOMAIN**
        - **Primary Characteristics**:
            - Questions about higher education institutions, programs, or processes.
            - Contains education-specific keywords (college, university, degree, course, admission, etc.).
            - Seeks information about academic topics.
        - **Key Topic Indicators**:
            - Admissions and Applications.
            - Course Information.
            - Degree Programs.
            - College/University Details.
            - Academic Requirements.
            - Scholarships and Financial Aid.
            - Campus Life.
            - Career Prospects.
            - Research Programs.
            - Faculty Information.
        - **Positive Examples**:
            - "What are the admission requirements for Harvard?"
            - "Tell me about computer science programs in top universities."
            - "How much is the tuition fee for MBA programs?"
            - "What scholarships are available for international students?"
            - "Can you explain the different types of engineering degrees?"
        - **Negative Examples**:
            - "How do I make pasta?" (Out-of-Scope).
            - "What's the weather like?" (Out-of-Scope).
            - "Hello, which universities offer AI programs?" (Greeting + Education).

        3. **GREETING_EDUCATION**
        - **Primary Characteristics**:
            - Combines a greeting with an education-related question.
            - Must have BOTH:
                1. Clear greeting element.
                2. Valid education domain question.
        - **Positive Examples**:
            - "Hi, what are the best medical schools in the US?"
            - "Hello, can you tell me about MBA programs?"
            - "Good morning, I need information about college admissions."
            - "Hey there, what scholarships are available?"
        - **Negative Examples**:
            - "Hello" (Greeting only).
            - "What are the best universities?" (Education only).
            - "Hi, what's today's date?" (Greeting + Out-of-Scope).

        4. **OUT_OF_SCOPE**
        - **Primary Characteristics**:
            - Questions unrelated to higher education.
            - Topics outside the educational domain.
            - General knowledge questions.
        - **Common Categories**:
            - Weather.
            - Cooking/Recipes.
            - Entertainment.
            - Sports.
            - General News.
            - Personal Advice.
            - Technical Support.
            - Business/Finance (non-educational).
        - **Positive Examples**:
            - "What's the best pizza recipe?"
            - "How do I invest in stocks?"
            - "What's the weather forecast?"
            - "Who won the Oscar for best picture?"
        - **Negative Examples**:
            - "What are the best colleges?" (Education).
            - "Hi, how do I cook pasta?" (Greeting + Out-of-Scope).

        5. **GREETING_OUT_OF_SCOPE**
        - **Primary Characteristics**:
            - Combines greeting with a non-education question.
            - Must have BOTH:
                1. Clear greeting element.
                2. Question unrelated to education.
        - **Positive Examples**:
            - "Hello, what's the weather today?"
            - "Hi, can you recommend a good restaurant?"
            - "Good morning, what's the latest news?"
            - "Hey, how do I fix my printer?"
        - **Negative Examples**:
            - "Hello, what universities are in London?" (Greeting + Education).
            - "What's the weather like?" (Out-of-Scope only).

        6. **HARMFUL**
        - **Primary Characteristics**:
            - Content related to:
                - Violence or self-harm.
                - Illegal activities.
                - Academic dishonesty.
                - Cybercrime.
                - Hate speech.
                - Harassment.
            - **Key Points**:
                - Always classify as HARMFUL if ANY harmful content is present.
                - Greeting + harmful content still classifies as HARMFUL.
                - Education-related harmful content (cheating, plagiarism) is still HARMFUL.
        - **Positive Examples**:
            - "How can I cheat on my exam?"
            - "What's the best way to hack a university database?"
            - "Hello, how do I plagiarize my thesis?"
            - "Ways to harm myself."

        ### MEMORY_QUERY
        - **Primary Characteristics**:
            - References previous conversations or information.
            - Asks about earlier mentioned topics.
            - Requests historical context.
        - **Key Indicators**:
            - "Remember when..."
            - "Earlier you said..."
            - "Last time we discussed..."
            - "Going back to..."
            - "As mentioned before..."
            - "What did you say about..."
            - "Can you remind me..."
        - **Positive Examples**:
            - "What did we discuss last time about Harvard?"
            - "Can you remind me of the admission requirements you mentioned?"
            - "Earlier you told me about scholarships, can you repeat that?"

        ### VAGUE_QUERY
        - **Primary Characteristics**:
            - Unclear or incomplete questions.
            - Missing crucial context.
            - Ambiguous references.
            - Single word queries.
        - **Subtypes**:
            1. **VAGUE_EDUCATION**:
                - Examples:
                    - "What about colleges?"
                    - "Studies?"
                    - "Tell me more."
            2. **VAGUE_OUT_OF_SCOPE**:
                - Examples:
                    - "What do you think?"
                    - "How does it work?"

        ### Output Format:
        {{
            "intent": "string",
            "confidence": float,
            "reasoning": "string",
            "needs_clarification": boolean,
            "suggested_prompts": ["string", ...]
        }}

        Classify this query: "{query}"
"""



from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


def create_intent_classification_prompt(query: str) -> str:
    """
    Create a formatted intent classification prompt
    
    Args:
        query (str): User's input query
    
    Returns:
        str: Formatted prompt for intent classification
    """
    return INTENT_CLASSIFICATION_PROMPT_TEMPLATE.format(query=json.dumps(query))




def detect_intent(state: State) -> State:
    latest_message = state['messages'][-1].content

    intent_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=create_intent_classification_prompt(latest_message))
    ])

    intent_chain = intent_prompt | llm | StrOutputParser()

    try:
        intent_result = intent_chain.invoke({})
        print("LLM Raw Output:", intent_result)  # Debug LLM output
        # parsed_result = json.loads(intent_result)
        json_match = re.search(r'{.*}', intent_result, re.DOTALL)
        parsed_result = json.loads(json_match.group(0))


    except (json.JSONDecodeError, TypeError):
        print("Fallback Triggered - Raw Output:", intent_result)
        parsed_result = {
            "intent": "UNCLASSIFIED",
            "confidence": 0.3,
            "reasoning": "Unable to parse intent.",
            "needs_clarification": True,
            "suggested_prompts": ["Could you clarify your question about IIT Delhi?"]
        }

    state["current_intent"] = parsed_result.get("intent", "UNCLASSIFIED")
    state["confidence"] = parsed_result.get("confidence", 0.0)
    state["reasoning"] = parsed_result.get("reasoning", "")
    state["needs_clarification"] = parsed_result.get("needs_clarification", False)
    state["suggested_prompts"] = parsed_result.get("suggested_prompts", [])
    return state

import streamlit as st



def route_based_on_intent(state: State) -> str:
    """
    Route the workflow based on detected intent
    
    Args:
        state (State): Current conversation state
    
    Returns:
        str: Next node to execute or END
    """
    # Get the first intent (assuming intents is a list)


    print("state of the curent intent what has been saved ",state['current_intent'])


    intent = state['current_intent']
    
    # Routing logic
    intent_routing = {
        'GREETING': 'generate_response',
        'EDUCATION_DOMAIN': 'bedrock_retrieval',
        'GREETING_EDUCATION': 'bedrock_retrieval',
        'MEMORY_QUERY': 'generate_response',  
        'OUT_OF_SCOPE': 'generate_response',
        'GREETING_OUT_OF_SCOPE': 'generate_response',
        'HARMFUL': 'generate_response',
        'VAGUE_EDUCATION': 'generate_response',  
        'VAGUE_QUERY': 'generate_response'

    }



    return intent_routing.get(intent, END)

def format_bedrock_retrieved_docs(state: State) -> State:
    """
    Format the Bedrock retrieved documents into a structured format.

    Args:
        state (State): The current state containing `bedrock_retrieved_docs`.

    Returns:
        State: Updated state with structured documents.
    """
    # Retrieve raw documents from the state
    retrieved_docs = state.get("bedrock_retrieved_docs", [])
    formatted_docs = []

    for doc in retrieved_docs:
        # Extract metadata and content
        metadata = doc.metadata
        page_content = doc.page_content

        # Parse relevant fields from metadata
        s3_uri = metadata.get("location", {}).get("s3Location", {}).get("uri", "N/A")
        source_metadata = metadata.get("source_metadata", {})
        score = metadata.get("score", 0)

        # Build a structured representation
        formatted_doc = {
            "s3_uri": s3_uri,
            "score": score,
            "source_metadata": source_metadata,
            "content_preview": page_content[:200],  # Limit content preview to 200 chars
        }
        formatted_docs.append(formatted_doc)

    # Add the formatted documents back to the state
    state["bedrock_retrieved_docs"] = formatted_docs

    return state





def bedrock_retrieval(state: State, config: RunnableConfig) -> State:
    """
    Perform Bedrock retrieval, format the results, and update the state.

    Args:
        state (State): Current state of the conversation.
        config (RunnableConfig): Configuration for Bedrock retrieval.

    Returns:
        State: Updated state with retrieved and formatted documents.
    """
    # Initialize Bedrock retriever
    bedrock_retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="BVWGHMKJOQ",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
    )

    # Get the user's query
    user_query = state["messages"][-1].content
    print("User query for Bedrock retrieval:", user_query)  

    try:
        # Perform retrieval
        retrieved_docs = bedrock_retriever.invoke(user_query)
        print("Raw retrieved documents:", retrieved_docs)  
    except Exception as e:
        print(f"Error during Bedrock retrieval: {e}")
        retrieved_docs = []

    if not retrieved_docs:
        print("No documents retrieved from Bedrock.")
        state["bedrock_retrieved_docs"] = []
        return state

    # Format the retrieved documents
    formatted_docs = []
    for doc in retrieved_docs:
        metadata = doc.metadata
        page_content = doc.page_content

        # Build a structured document representation
        formatted_doc = {
            "s3_uri": metadata.get("location", {}).get("s3Location", {}).get("uri", "N/A"),
            "score": metadata.get("score", 0),
            "source_metadata": metadata.get("source_metadata", {}),
            "content_preview": page_content,  # Truncate content preview
        }
        formatted_docs.append(formatted_doc)

    # Update the state with formatted documents
    state["bedrock_retrieved_docs"] = formatted_docs

    print("bedrock_retrieved_docs",state["bedrock_retrieved_docs"])
    return state


class AIMessage:
    def __init__(self, content):
        self.content = content

from datetime import datetime

def save_interaction_to_memory(state: State, user_query: str, ai_response: str, bedrock_context: str):
    """
    Save the current interaction to memory.
    """
    current_interaction = {
        "query": user_query,
        "response": ai_response,
        "bedrock_context": bedrock_context,
        "timestamp": datetime.utcnow().isoformat()  # Add a timestamp
    }

    # Add to memory
    if "memory" not in state:
        state.memory = {"top_k_memories": []}
    state["memory"]["top_k_memories"].append(current_interaction)

    # Limit memory to the last 5 interactions
    state["memory"]["top_k_memories"] = state["memory"]["top_k_memories"][-5:]


    # Update recall_memories with a summary of this interaction
    summary = f"Query: {user_query}, Response: {ai_response[:100]}..."  
    state["recall_memories"].append(summary)

    # Limit recall_memories to the last 5 entries
    state["recall_memories"] = state["recall_memories"][-5:]


def recall_previous_question(state: State) -> str:
    """
    Recall the user's previous query and response.
    """
    memory = state.memory.get("top_k_memories", [])
    if memory:
        last_interaction = memory[-1]
        return (
            f"Your previous question was: '{last_interaction['query']}'.\n"
            f"My response was: '{last_interaction['response']}'"
        )
    return "I'm sorry, I couldn't find any previous questions in memory."

def is_recall_request(user_query: str) -> bool:
    """
    Detect if the user's query is asking about previous questions.
    """
    recall_keywords = ["previous question", "last question", "what did I ask"]
    return any(keyword in user_query.lower() for keyword in recall_keywords)


def generate_summary(state: State) -> str:
    """
    Generate a concise summary of the conversation history using Tiktoken.
    """
    # encoding = tiktoken.encoding_for_model("gpt-4")
    conversation_history = " ".join([msg.content for msg in state["messages"]])
    max_tokens = 2048  # Limit tokens for summary generation

    # Trim conversation history if needed
    history_tokens = encoding.encode(conversation_history)
    if len(history_tokens) > max_tokens:
        conversation_history = encoding.decode(history_tokens[:max_tokens])

    summary_prompt = f"""
    Summarize the following conversation in a concise manner:
    {conversation_history}
    """
    return llm.predict(summary_prompt).strip()


@dataclass
class PromptTemplate:
    template: str
    example_response: str  # For reference, not used in actual prompts


class IntentPromptHandler:
    def __init__(self):
        self.prompt_templates = {
            "VAGUE_EDUCATION": PromptTemplate(
                template="""Context: The user has made a vague education-related query. Help them clarify their needs while maintaining a supportive and encouraging tone.

User Query: {user_query}
Retrieved Context: {context}

Your response should:
1. Acknowledge their interest in learning
2. Explain why more specific information would help you assist them better
3. Provide 3-4 specific examples of what details would be helpful
4. Include 2-3 example questions showing the level of detail that would be ideal
5. End with an encouraging note about their learning journey

Remember to:
- Be warm and supportive
- Use examples relevant to their original query
- Make suggestions that guide them toward more specific questions
- Maintain an educational and mentoring tone""",
                example_response="I can see you're interested in learning, which is fantastic! To provide you with the most helpful guidance..."
            ),
            
            "VAGUE_QUERY": PromptTemplate(
                template="""Context: The user has submitted a query that needs more clarification to provide a meaningful response.

User Query: {user_query}
Retrieved Context: {context}

Your response should:
1. Acknowledge their question
2. Identify the specific aspects that need clarification
3. Provide 3-4 example clarifying questions
4. Suggest how they might rephrase their query
5. Include 2-3 examples of well-formed queries similar to what they might be asking

Guidelines:
- Be specific about what information would help
- Use examples related to their domain of interest
- Maintain a helpful and patient tone
- Show how more specific queries lead to better answers""",
                example_response="I see what you're asking about, and I'd love to help. To provide the most useful response..."
            ),
            
            "GREETING": PromptTemplate(
                template="""Context: The user has initiated a conversation with a greeting.

User Query: {user_query}
Retrieved Context: {context}

Your response should:
1. Return a warm and professional greeting
2. Briefly explain your capabilities relevant to their needs
3. Suggest 3-4 specific types of questions you can help with
4. Provide 2-3 example queries that showcase your abilities
5. End with an inviting prompt for their specific question

Guidelines:
- Keep the tone professional but friendly
- Focus on educational and learning-related capabilities
- Use examples that demonstrate the range of your expertise
- Encourage specific, detailed questions""",
                example_response="Hello! Thank you for reaching out. I'm here to help you with your learning journey..."
            ),
            
            "GREETING_OUT_OF_SCOPE": PromptTemplate(
                template="""Context: The user has greeted you but may be seeking help outside your primary educational focus.

User Query: {user_query}
Retrieved Context: {context}

Your response should:
1. Acknowledge their greeting warmly
2. Clearly explain your focus on educational assistance
3. Provide 3-4 examples of topics you can help with
4. Suggest how they might rephrase their query to fit within your scope
5. Include 2-3 example educational queries you can assist with

Guidelines:
- Be polite but clear about your educational focus
- Redirect the conversation constructively
- Demonstrate the types of educational help you can provide
- Maintain a helpful tone while setting clear boundaries""",
                example_response="Hi there! While I'm focused on educational topics, I'd be happy to help you learn about..."
            ),
            
            "HARMFUL": PromptTemplate(
                template="""Context: The user has made a query that could lead to harmful outcomes. Redirect them constructively while maintaining engagement.

User Query: {user_query}
Retrieved Context: {context}

Your response should:
1. Acknowledge their underlying goal or interest
2. Suggest 3-4 constructive alternative approaches
3. Provide examples of positive outcomes from these alternatives
4. Include specific, actionable steps they can take
5. End with an encouraging note about positive problem-solving

Guidelines:
- Focus on constructive alternatives
- Maintain a supportive and non-judgmental tone
- Emphasize positive outcomes
- Provide specific, actionable guidance
- Encourage ethical and beneficial approaches""",
                example_response="I understand you're looking to achieve [goal]. Let's explore some effective approaches that..."
            )
        }

    def get_prompt(self, intent: str, user_query: str, context: str) -> str:
        if intent not in self.prompt_templates:
            return self._get_default_prompt(user_query, context)
            
        template = self.prompt_templates[intent]
        return template.template.format(user_query=user_query, context=context)
    
    def _get_default_prompt(self, user_query: str, context: str) -> str:
        return f"""Context: The user has made a query that needs a response.

User Query: {user_query}
Retrieved Context: {context}

Your response should:
1. Address their query directly
2. Provide relevant information based on the context
3. Include specific examples or explanations
4. Suggest next steps or follow-up questions

Guidelines:
- Be clear and concise
- Use specific examples
- Maintain a helpful tone
- Provide actionable information"""




valid_intents = ["GREETING", "VAGUE_EDUCATION", "OUT_OF_SCOPE", "GREETING_OUT_OF_SCOPE", "HARMFUL", "VAGUE_QUERY"]


def generate_response(state: State, config: RunnableConfig) -> State:
    """
    Generate a response for the user query, handling vague intents appropriately.
    """
    user_query = state["messages"][-1].content
    detected_intent = state.get("current_intent", "UNKNOWN")
    needs_clarification = state.get("needs_clarification", False)
    suggested_prompts = state.get("suggested_prompts", [])
    bedrock_retrieved_docs = state.get("bedrock_retrieved_docs", [])

    # Handle Bedrock context
    bedrock_context = "\n".join(
        [f"- {doc['content_preview']}" for doc in bedrock_retrieved_docs]
    ) if bedrock_retrieved_docs else "No additional documents retrieved from Bedrock."


    if detected_intent == "MEMORY_QUERY":

        memory = state["memory"].get("top_k_memories", [])
        if memory:
            last_interaction = memory[-1]
            ai_response =  [
                f"Your previous question was: '{last_interaction['query']}'.\n"
                f"My response was: '{last_interaction['response']}'"
            
            ]
        else:
            ai_response = "please provide question so that i can provide you result"

    #     ai_response = recall_previous_question(state)

        




    # Determine response based on intent
    if detected_intent == "VAGUE_EDUCATION" and needs_clarification:
        # Generate a clarification response
        clarification_response = (
            "It seems your query lacks some details. Could you clarify?\n\n"
            "Here are some suggestions:\n"
            + "\n".join([f"- {prompt}" for prompt in suggested_prompts])
        )
        ai_response = clarification_response

    elif detected_intent == "VAGUE_QUERY" and needs_clarification:
        # Generate a clarification response
        clarification_response = (
            "Could you please provide more context or clarify what you mean by?\n\n"
            "Here are some suggestions:\n"
            + "\n".join([f"- {prompt}" for prompt in suggested_prompts])
        )
        ai_response = clarification_response

    # elif detected_intent in valid_intents:
    #     # Generate a clarification response

    #     prompt_handler = IntentPromptHandler()

    #     prompt = prompt_handler.get_prompt(detected_intent, user_query, bedrock_context)

    #     print("prompt++++",prompt)

    #     clarification_response = (
    #         "Could you please provide more context or clarify what you mean by?\n\n"
    #         "Here are some suggestions:\n"
    #         + "\n".join([f"- {prompt}" for prompt in suggested_prompts])
    #     )
    #     ai_response = prompt
    #     print("ai_response",ai_response)
        
    elif detected_intent == "GREETING_OUT_OF_SCOPE":
        # Generate a clarification response
        clarification_response = (
            "Could you please provide more context or clarify what you mean by?\n\n"
            "Here are some suggestions:\n"
            + "\n".join([f"- {prompt}" for prompt in suggested_prompts])
        )
        ai_response = clarification_response
    
    elif detected_intent == "HARMFUL":
        # Generate a clarification response
        clarification_response = (
            "Could you please provide more context or clarify what you mean by?\n\n"
            "Here are some suggestions:\n"
            + "\n".join([f"- {prompt}" for prompt in suggested_prompts])
        )
        ai_response = clarification_response

    elif detected_intent == "GREETING":
        # Generate a clarification response
        clarification_response = (
            "Could you please provide more context or clarify what you mean by?\n\n"
            "Here are some suggestions:\n"
            + "\n".join([f"- {prompt}" for prompt in suggested_prompts])
        )
        ai_response = clarification_response
    else:
        # Construct a regular prompt with context for other intents
        prompt_template = f"""
        Query: {user_query}
        Context: {bedrock_context}
        Detected Intent: {detected_intent}

        Guidelines:
        - Respond concisely and informatively.
        - If the query is vague, ask for clarification in a polite tone.
        """

        try:
            llm_response = llm.invoke(prompt_template)
            ai_response = (
                llm_response.content
                if isinstance(llm_response, LCAIMessage)
                else llm_response
                if isinstance(llm_response, str)
                else "Unexpected response format."
            )
        except Exception as e:
            ai_response = "I'm sorry, something went wrong while generating the response."

    # Save the interaction to memory
    save_interaction_to_memory(state, user_query, ai_response, bedrock_context)

    # Save recall memory for persistence
    conversation_to_save = {
        "query": user_query,
        "response": ai_response,
        "bedrock_context": bedrock_context,
    }
    # save_recall_memory.invoke(json.dumps(conversation_to_save), config)

    conversation_to_save = f"User: {user_query}\nAI: {ai_response}"
    save_recall_memory.invoke(conversation_to_save, config)

    # Append AI response to state
    state["messages"].append(LCAIMessage(content=ai_response))

    return state






graph = StateGraph(State)

graph.add_node("load_memory", load_memory)
graph.add_node("detect_intent", detect_intent)

graph.add_node("bedrock_retrieval", bedrock_retrieval)
graph.add_node("generate_response", generate_response)





graph.add_edge(START, "load_memory")
graph.add_edge("load_memory", "detect_intent")


graph.add_conditional_edges(
    "detect_intent", 
    route_based_on_intent, 
    {
        "bedrock_retrieval": "bedrock_retrieval",  # Explicitly map the return value
        "generate_response": "generate_response",  # Ensure this exists
        END: END  # Explicitly map END
    }
)

# graph.add_edge("bedrock_retrieval", END)
graph.add_edge("bedrock_retrieval", "generate_response")
graph.add_edge("generate_response", END)


# Compile LangGraph
memory = MemorySaver()

compiled_graph = graph.compile(checkpointer=memory)






def pretty_print_stream_chunk(chunk):
    """
    Pretty print updates from each node in the graph execution.
    """
    for node, updates in chunk.items():
        print(f"=== Update from Node: {node} ===")

        if "messages" in updates:
            print("Messages:")
            for message in updates["messages"]:

                print("teteteettete",message)
                print(f"  - Role: {getattr(message, 'role', 'unknown')}")
                print(f"    Content: {getattr(message, 'content', 'N/A')}")

        if "state" in updates:
            print("State Updates:")
            pprint.pprint(updates["state"], indent=4)

        for key, value in updates.items():
            if key not in {"messages", "state"}:
                print(f"Other Update - {key}: {value}")
        print("\n")



# input_state = {
#     "messages": [
#         {"role": "user", "content": "what is eligibility crieteria of  iit delhi?"}
#     ],
#     "memory": {"session_context": []},
# }

# from langchain.schema import HumanMessage

# input_state = {
#     "messages": [HumanMessage(content="What about this college")],
#     "memory": {"session_context": []},
# }


# from langchain_core.runnables import RunnableConfig



# config = RunnableConfig(configurable={"user_id": "10", "thread_id": "2"})


# # Run the Workflow
# for chunk in compiled_graph.stream(input_state, config=config):
#     pretty_print_stream_chunk(chunk)




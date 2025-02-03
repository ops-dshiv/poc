import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from met9 import SessionManager, MemoryManager, KnowledgeRetriever, IntentDetector, ResponseGenerator, EducationAssistant

class ChatbotTestCases(unittest.TestCase):

    def setUp(self):
        """Setup necessary instances before running each test."""
        self.session_manager = SessionManager()
        self.memory_manager = MemoryManager()
        self.knowledge_retriever = KnowledgeRetriever()
        self.intent_detector = IntentDetector()
        self.response_generator = ResponseGenerator()
        self.assistant = EducationAssistant()
        self.thread_id = "test_thread_123"

    ### ✅ **Session Manager Tests**
    def test_store_and_retrieve_session(self):
        """Test storing and retrieving session data."""
        self.session_manager.store_session(self.thread_id, {"last_topic": "IIT Delhi"})
        session_data = self.session_manager.retrieve_session(self.thread_id)
        self.assertEqual(session_data["last_topic"], "IIT Delhi")

    def test_update_session(self):
        """Test updating session context dynamically."""
        self.session_manager.update_session(self.thread_id, "last_topic", "MIT AI Research")
        session_data = self.session_manager.retrieve_session(self.thread_id)
        self.assertEqual(session_data["last_topic"], "MIT AI Research")

    ### ✅ **Memory Manager Tests**
    def test_store_and_recall_memories(self):
        """Test storing and retrieving past conversations."""
        query = "Tell me about IIT Delhi"
        response = "IIT Delhi is a premier engineering institute in India."
        self.memory_manager.store_memory(self.thread_id, query, response)
        
        memories = self.memory_manager.recall_memories(self.thread_id, k=2)
        self.assertIn(query, memories[0])
        self.assertIn(response, memories[0])

    def test_recall_empty_memory(self):
        """Test behavior when no memory exists for a thread."""
        memories = self.memory_manager.recall_memories("unknown_thread", k=2)
        self.assertEqual(memories, [])

    ### ✅ **Knowledge Retriever Tests**
    @patch("chatbot.bedrock_runtime.invoke_model")
    def test_knowledge_retrieval(self, mock_bedrock):
        """Test retrieving knowledge from AWS Bedrock."""
        mock_bedrock.return_value = {
            'body': MagicMock(read=MagicMock(return_value=json.dumps({
                "retrievalResults": [{"content": "IIT Delhi admission process"}]
            })))
        }
        results = self.knowledge_retriever.retrieve_data("IIT Delhi admission")
        self.assertEqual(results, ["IIT Delhi admission process"])

    @patch("chatbot.bedrock_runtime.invoke_model", side_effect=Exception("AWS Error"))
    def test_knowledge_retrieval_failure(self, mock_bedrock):
        """Test handling of AWS Bedrock failure."""
        results = self.knowledge_retriever.retrieve_data("IIT Delhi admission")
        self.assertEqual(results, [])

    ### ✅ **Intent Detection Tests**
    @patch("chatbot.ChatBedrock.invoke")
    def test_intent_detection_education(self, mock_bedrock):
        """Test intent detection for an educational query."""
        mock_bedrock.return_value.content = '{"intent": "education", "entities": {"institution": "IIT Delhi"}, "needs_clarification": false}'
        intent_data = self.intent_detector.detect(["history"], "Tell me about IIT Delhi", self.thread_id)
        self.assertEqual(intent_data["intent"], "education")
        self.assertEqual(intent_data["entities"]["institution"], "IIT Delhi")

    @patch("chatbot.ChatBedrock.invoke")
    def test_intent_detection_clarification(self, mock_bedrock):
        """Test intent detection requiring clarification."""
        mock_bedrock.return_value.content = '{"intent": "education", "needs_clarification": true, "clarification_prompt": "Do you mean IIT Delhi or IIT Bombay?"}'
        intent_data = self.intent_detector.detect(["history"], "Tell me about IIT", self.thread_id)
        self.assertTrue(intent_data["needs_clarification"])
        self.assertEqual(intent_data["clarification_prompt"], "Do you mean IIT Delhi or IIT Bombay?")

    ### ✅ **Response Generation Tests**
    @patch("chatbot.ChatBedrock.invoke")
    def test_response_generation(self, mock_bedrock):
        """Test generating response for an educational query."""
        mock_bedrock.return_value.content = "IIT Delhi is a top engineering institute."
        response = self.response_generator.generate({"query": "Tell me about IIT Delhi", "thread_id": self.thread_id})
        self.assertEqual(response, "IIT Delhi is a top engineering institute.")

    ### ✅ **Full Workflow Integration Test**
    @patch("chatbot.ChatBedrock.invoke")
    def test_full_conversation_flow(self, mock_bedrock):
        """Test the entire chatbot flow from input to response."""
        mock_bedrock.side_effect = [
            MagicMock(content=json.dumps({"intent": "education", "entities": {"institution": "IIT Delhi"}, "needs_clarification": False})),  # Intent detection
            MagicMock(content=json.dumps({"retrievalResults": [{"content": "IIT Delhi admission process"}]})),  # Knowledge retrieval
            MagicMock(content="IIT Delhi admission is based on JEE Advanced.")  # Response generation
        ]

        state = {"query": "Tell me about IIT Delhi admission", "thread_id": self.thread_id}
        state = self.assistant._retrieve_context(state)
        state = self.assistant._classify_intent(state)
        state = self.assistant._fetch_knowledge(state)
        state = self.assistant._generate_response(state)

        self.assertEqual(state["response"], "IIT Delhi admission is based on JEE Advanced.")

if __name__ == "__main__":
    unittest.main()

from agentlite.actions import BaseAction
from transformers import AutoModel, AutoTokenizer
import hazm
import numpy as np

from agentlite.agents import BaseAgent

# Action for checking relevance using ParsBERT
class CheckRelevance(BaseAction):
    action_name = "CheckRelevance"
    action_desc = "Check if the response is relevant to the question using semantic similarity."
    params_doc = {"question": "The input question", "response": "The generated response"}
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
        self.model = AutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
        self.normalizer = hazm.Normalizer()
    
    def __call__(self, question, response):
        # Normalize question and response
        question = self.normalizer.normalize(question)
        response = self.normalizer.normalize(response)
        
        # Encode question and response
        inputs_q = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        inputs_r = self.tokenizer(response, return_tensors="pt", padding=True, truncation=True)
        
        # Get embeddings
        embeddings_q = self.model(**inputs_q).last_hidden_state.mean(dim=1).detach().numpy()
        embeddings_r = self.model(**inputs_r).last_hidden_state.mean(dim=1).detach().numpy()
        
        # Compute cosine similarity
        similarity = np.dot(embeddings_q, embeddings_r.T) / (np.linalg.norm(embeddings_q) * np.linalg.norm(embeddings_r))
        return "Relevant" if similarity > 0.7 else "Not Relevant"

# Action for checking completeness (simplified)
class CheckCompleteness(BaseAction):
    action_name = "CheckCompleteness"
    action_desc = "Check if the response is complete based on length and keywords."
    params_doc = {"response": "The generated response"}
    
    def __call__(self, response):
        words = len(hazm.word_tokenize(response))
        return "Complete" if words > 10 else "Incomplete"  # Simple rule for demo

# Action for retrying the process
class Retry(BaseAction):
    action_name = "Retry"
    action_desc = "Instruct the Individual Agent to retry the task."
    params_doc = {"task_instruction": "The original task instruction"}
    
    def __call__(self, task_instruction):
        return f"Retry the task with instruction: {task_instruction}"


# Validator Agent
class ValidatorAgent(BaseAgent):
    def __init__(
            self,
            name: str = "ValidatorAgent",
            role: str = "Evaluate responses from Individual Agent"
    ):
        """ValidatorAgent inherits BaseAgent. It has all methods for base agent
        and it can communicate with other agent. It controls LaborAgents to complete tasks.
        Also, one can initialize ManagerAgent with a list of PeerAgents
        or add the peerAgent later for discussion.

        :param name: the name of this agent, defaults to "ValidatorAgent"
        :type name: str, optional
        :param role: the role of this agent, defaults to "Evaluate responses from Individual Agent"
        :type role: str, optional
        """
        actions = [
            CheckRelevance(),
            CheckCompleteness(),
            Retry()
        ]
        super().__init__(
            name=name, 
            role=role,
            actions=actions
        )

    def evaluate_response(self, question, response, task_instruction):
        # Check relevance
        relevance = self.call_action("CheckRelevance", question=question, response=response)
        if relevance == "Not Relevant":
            return False # self.call_action("Retry", task_instruction=task_instruction)
        
        # Check completeness
        completeness = self.call_action("CheckCompleteness", response=response)
        if completeness == "Incomplete":
            return False # self.call_action("Retry", task_instruction=task_instruction)
        
        return True # "Response is valid"
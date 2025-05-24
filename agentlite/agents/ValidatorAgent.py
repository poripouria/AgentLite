from agentlite.actions import BaseAction
from agentlite.agents import BaseAgent
from agentlite.commons import TaskPackage
import json

class ValidateAnswer(BaseAction):
    def __init__(self):
        """
        Initialize the ValidateAnswer action which checks the correctness of an answer
        to a given question and provides structured feedback.
        parameters:
            question (str): The original question to validate against.
            answer (str): The proposed answer to validate.
        """
        super().__init__(
            action_name="ValidateAnswer",
            action_desc="Check the answer's correctness and provide structured feedback.",
            params_doc={
                "question": "The original question.",
                "answer": "The proposed answer to validate."
            }
        )

    def __call__(self, question: str, answer: str) -> str:
        prompt = (
            f"You are a Validator Agent. Your job is to check the answer provided to a question.\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Evaluate the correctness. Reply in JSON format as follows:\n"
            f"{{\"valid\": true/false, \"feedback\": \"validation comments for feedback\"}}"
        )
        # This assumes the agent's LLM responds to the prompt call
        return prompt

class ValidatorAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(
            name="ValidatorAgent",
            role="Validate responses from individual agents for correctness.",
            actions=[ValidateAnswer()],
            llm=llm
        )

    def respond(self, task_pkg: TaskPackage, **kwargs):
        question = task_pkg.instruction
        answer = task_pkg.answer or ""
        validate_action = self.actions[0]
        prompt = validate_action(question=question, answer=answer)
        llm_response = self.llm(prompt)

        try:
            result = json.loads(llm_response)
            valid = result.get("valid", False)
            feedback = result.get("feedback", "No feedback provided.")
        except json.JSONDecodeError:
            valid = False
            feedback = "ValidatorAgent could not parse the response. Please check formatting."

        # Update task package based on validation
        task_pkg.completion = "validated" if valid else "needs_revision"
        task_pkg.answer = feedback
        return task_pkg



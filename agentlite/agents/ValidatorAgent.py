from .BaseAgent import BaseAgent

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
        super().__init__(
            name=name, 
            role=role
        )
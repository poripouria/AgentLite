import unittest

from agentlite.llm import LLMConfig
from agentlite.llm.agent_llms import LangchainLLM, OpenAIChatLLM, get_llm_backend


class Test_act_match(unittest.TestCase):
    def test_1(self):
        # llm_name = "gpt-4-32k"
        # llm_config_dict = {
        #     "llm_name": llm_name,
        #     "temperature": 0.1,
        #     "context_len": 4096,
        # }
        llm_config_dict = {
            "llm_name": "gemma3:4b", # "deepseek-r1:8b"
            "temperature": 0.9,
            "provider": "ollama",
            "base_url": "http://localhost:11434"
        }
        llm_config = LLMConfig(llm_config_dict)
        # llm = OpenAIChatLLM(llm_config)
        llm = get_llm_backend(llm_config)
        prompt = "what is the founder of microsoft?"
        a = llm(prompt)
        print(a)

if __name__ == "__main__":
    Test_act_match().test_1()
    # unittest.main()
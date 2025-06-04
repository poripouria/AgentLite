"""functions or objects shared by agents"""

import re
import json

from agentlite.actions.BaseAction import BaseAction


def name_checking(name: str):
    """ensure no white space in name"""
    white_space = [" ", "\n", "\t"]
    for w in white_space:
        if w in name:
            return False
    return True


def act_match(input_act_name: str, act: BaseAction):
    # print(f"Checking action match for: {input_act_name} with {act.action_name}")
    if input_act_name == act.action_name:  # exact match
        # print(f"EXACT matched")
        return True
    input_act_name_cleaned = re.sub(r"[<>{}\[\]().,;:!@#$%^&*]", "", input_act_name).lower()
    act_name_cleaned = re.sub(r"[<>{}\[\]().,;:!@#$%^&*]", "", act.action_name).lower()
    if input_act_name_cleaned == act_name_cleaned:  # cleaned match
        # print(f"CLEANED matched")
        return True
    if input_act_name.lower() in act.action_name.lower() or act.action_name.lower() in input_act_name.lower():
        # fuzzy match, if input_act_name is a substring of act.action_name or vice versa
        # print(f"FUZZY matched")
        return True
    
    ## To-Do More fuzzy match
    return False


def parse_action(string: str) -> tuple[str, dict, bool]:
    """
    Parse an action string into an action type and an argument.
    """

    string = string.strip(" ").strip(".").strip(":").split("\n")[0]
    # Adjusted regex to better capture action name and arguments
    pattern = r"^(?:Action:)?\s*(\w+)\[(.+)\]$"
    match = re.match(pattern, string)
    PARSE_FLAG = True

    if match:
        action_type = match.group(1).strip()
        arguments_str = match.group(2).strip()
        try:
            # Ensure arguments are treated as a JSON string
            if not arguments_str.startswith('{') and not arguments_str.startswith('['):
                # Attempt to fix common non-JSON-compliant string issues if possible,
                # or handle as a simple string if it's not meant to be JSON.
                # For now, we'll assume it should be JSON and try to parse.
                # If it's a simple string like "some thought", it might fail here.
                # Consider if the logic should accommodate non-JSON params for some actions.
                arguments = json.loads(arguments_str)
            else:
                arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            # Fallback or error handling if arguments are not valid JSON
            # This might happen if the LLM doesn't format it as expected.
            # Depending on requirements, could try to wrap in quotes or return as raw string.
            # For now, marking as parse error.
            PARSE_FLAG = False
            return string, {}, PARSE_FLAG # Return original string and empty dict on error
        return action_type, arguments, PARSE_FLAG
    else:
        # Handle cases where the string does not match the expected Action[Arguments] format
        # This could be a simple thought or a malformed action string.
        # Consider returning the string as the action_type if it's a simple thought.
        # For now, consistent with original behavior on mismatch.
        PARSE_FLAG = False
        return string, {}, PARSE_FLAG


AGENT_CALL_ARG_KEY = "Task"
NO_TEAM_MEMEBER_MESS = (
    """No team member for manager agent. Please check your manager agent team."""
)
ACION_NOT_FOUND_MESS = (
    """"This is the wrong action to call. Please check your available action list."""
)

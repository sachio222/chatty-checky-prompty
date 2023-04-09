# File: chat_gan.py
# Description: This file implements the Chatty, Checky, Prompty algorithm, which
# uses any three LLM models to collaboratively generate and validate prompts.
# The algorithm involves an interaction between the three models, where Chatty
# generates prompts, Checky validates them, and Prompty suggests improvements.

# Copyright (c) 2023, Jake Krajewski.

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
#########################################
# PYTHON VERSION IS NOT TESTED.         #
# THIS IS PROVIDED AS A STARTING POINT. #
#########################################
# Constants
character_match = """Chatty cannot see the desired output. Chatty only has your prompt to help them.
      Chatty does not know how close they are, or how far they are, what they did right or what they did wrong.
      Tell them what to do, and what not to do. Tell them what is right, also tell them what is wrong about their suggestion.
      The fewer attempts Chatty needs to get it right, the better you all fare.
      The more exact they match the desired output, the better you fare.
      Think about things like "what is the right length?", "what is the right format?", "what is the right tone?".
      Say things like, make it shorter, longer, etc. etc.
      You must achieve a direct match. Use any language or symbols that you want.
      This is directed to an AI and does not have to be human readible.
      The shorter your instructions, the better.
      Be Creative. Be Specific. You may model your prompt after your own instructions in order to guide chatty.
      Use your history to determine what has worked and what has not.
      The level of incorrectness is determined by how many characters are different between the desired output and the actual output.
      The lower the number of characters that are different, the lower the score.
      The desire is to reach 0 incorrectness.
      Chatty has no memory."""  # Truncated for readability
concept_match = """Chatty cannot see the desired output. Chatty only has your prompt to help them.
      Chatty does not know how close they are, or how far they are, what they did right or what they did wrong.
      Tell them what to do, and what not to do. Tell them what is right, also tell them what is wrong about their suggestion.
      The fewer attempts Chatty needs to get it right, the better you all fare.
      The more exact they match the desired output, the better you fare.
      Think about things like "what is the right length?", "what is the right format?", "what is the right tone?".
      Say things like, make it shorter, longer, etc. etc.
      You must achieve a direct match. Use any language or symbols that you want.
      This is directed to an AI and does not have to be human readible.
      The shorter your instructions, the better.
      Be Creative. Be Specific. You may model your prompt after your own instructions in order to guide chatty.
      Use your history to determine what has worked and what has not.
      The desire is to reach 0 incorrectness.
      Chatty has no memory."""  # Truncated for readability

chatty = ChatOpenAI(temperature=0.7)
checky = ChatOpenAI(temperature=0.0)
prompty = ChatOpenAI(temperature=0.8)

MAX_ATTEMPTS = 15

app = FastAPI()

@app.get("/api/langchain/gan")
async def handler(desired_output: str):
    initial_prompt = "generate something"
    prompt_history = []

    correct_prompt = await find_correct_prompt(initial_prompt, 0, desired_output, prompt_history)

    if correct_prompt:
        return {"correctPrompt": correct_prompt}
    else:
        raise HTTPException(status_code=400, detail="No suitable prompt found.")


async def find_correct_prompt(chatty_prompt: str, attempts: int, desired_output: str, prompt_history: List[str]) -> Optional[str]:
    if attempts >= MAX_ATTEMPTS:
        return None

    chatty_output = await chatty.call([
        SystemMessage("You are Chatty. You are an AI guesser. You receive information from Prompty. Who tells you what to do."),
        SystemMessage("Respond only with the completion you want Checky to check. Checky will tell Prompty if you are right or wrong, and Prompty will suggest how to improve your output."),
        AIMessage(chatty_prompt)
    ])

    print(f"CHATTY:\n[Attempt {attempts + 1}]:", chatty_output.text)

    validation_result = await checky.call(format_checky_input(chatty_output.text, desired_output, True))

    print(f"CHECKY:\nValidation result [Attempt {attempts + 1}]:", validation_result.text)

    validation_result_json = json.loads(validation_result.text)
    validation_action = validation_result_json["action"]
    validation_is_correct = validation_result_json["isCorrect"]

    print("DO:\n", validation_action)

    if validation_is_correct:
        return chatty_prompt
    else:
        prompt_history.append(chatty_prompt)

        action = validation_action
        if action == "suggest_prompt":
            prompty_output = await prompty.call(format_prompty_input(chatty_output.text, desired_output, prompt_history))

            print(f"PROMPTY:\n[Attempt {attempts + 1}]:", prompty_output.text)
            return await find_correct_prompt(prompty_output.text, attempts + 1, desired_output, prompt_history)
        elif action == "retry":
            prompty_output_retry = await prompty.call(format_prompty_input(chatty_output.text, desired_output, prompt_history))

            print("PROMPTY", prompty_output_retry.text)
            return await find_correct_prompt(prompty_output_retry.text, attempts + 1, desired_output, prompt_history)
        elif action == "error":
            print(f"Error in checky: {validation_result.reason}")
            return None
        else:
                        print(f"Unknown action from checky: {action}")
            return None

def format_checky_input(chatty_output: str, desired_output: str, analyze: bool):
    return [
        SystemMessage("instructions: 'Checky, your role is to validate whether the generated output by Chatty '{}' matches the desired output: '{}'. Please follow these rules:'".format(chatty_output, desired_output)),
        SystemMessage(json.dumps(checky_validation_rules(desired_output, analyze))),
        SystemMessage("Only reply with the appropriate action in the appropriate format. Do not explain, or add any other text.")
    ]

def format_prompty_input(chatty_output: str, desired_output: str, prompt_history: List[str]):
    return [
        SystemMessage("instructions: 'Prompty, your role is to get Chatty, your fellow AI, to do as you ask. Get them to generate the desired output: {}.'".format(desired_output)),
        AIMessage("prompt history: {}".format(prompt_history)),
        SystemMessage(json.dumps({
            "content": "suggest_prompt",
            "data": {
                "instructions": "Based on the generated output, the desired output: {}, and Chatty's current response: {} guide Chatty to achieve an exact match between the desired output and actual output.".format(desired_output, chatty_output)
            }
        })),
        SystemMessage(concept_match),
        SystemMessage("Respond only with text directed at Chatty [Incorrectness score: <incorrectness score>].")
    ]

def checky_validation_rules(desired_output: str, analyze: bool):
    return {
        "data": {
            "rules": [
                {
                    "if": "output is correct",
                    "return": {
                        "isCorrect": True
                    }
                },
                {
                    "if": "output doesn't meet criteria and a prompt suggestion is needed",
                    "return": {
                        "isCorrect": False,
                        "action": "suggest_prompt",
                        "reason": "The generated output does not meet the desired criteria."
                    }
                },
                {
                    "if": "output is close but needs slight modification",
                    "return": {
                        "isCorrect": False,
                        "action": "retry",
                        "reason": "The generated output is close but needs slight modification."
                    }
                },
                {
                    "if": "an error occurs during validation",
                    "return": {
                        "isCorrect": False,
                        "action": "error",
                        "reason": "An error occurred during the validation process."
                    }
                },
                {
                    # Default case added
                    "return": {
                        "isCorrect": False,
                        "action": "error",
                        "reason": "Unable to determine the appropriate action."
                    }
                }
            ]
        }
    }


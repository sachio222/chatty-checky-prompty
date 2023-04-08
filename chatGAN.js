// @/pages/api/langchain/gan.js

// File: chatGan.js
// Description: This file implements the Chatty, Checky, Prompty algorithm, which
// uses any three LLM models to collaboratively generate and validate prompts.
// The algorithm involves an interaction between the three models, where Chatty
// generates prompts, Checky validates them, and Prompty suggests improvements.

// Copyright (c) 2023, Jake Krajewski.

// Import required modules and dependencies
import { ChatOpenAI } from 'langchain/chat_models';
import {
  HumanChatMessage,
  SystemChatMessage,
  AIChatMessage
} from 'langchain/schema';

// Constants
const characterMatch = `Chatty cannot see the desired output. Chatty only has your prompt to help them.
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
      Chatty has no memory.`; // Truncated for readability
const conceptMatch = `Chatty cannot see the desired output. Chatty only has your prompt to help them.
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
      Chatty has no memory.`; // Truncated for readability

const chatty = new ChatOpenAI({ temperature: 0.7 });
const checky = new ChatOpenAI({ temperature: 0.0 });
const prompty = new ChatOpenAI({ temperature: 0.8 });

const MAX_ATTEMPTS = 15;

// Handler function
export default async function handler(req, res) {
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  if (!OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not defined.');
  }

  const { desiredOutput } = req.query;

  const initialPrompt = 'generate something';
  
  
  
  const MAX_ATTEMPTS = 15;
  const promptHistory = [];

  const correctPrompt = await findCorrectPrompt(initialPrompt, 0, desiredOutput, promptHistory, req);

  if (correctPrompt) {
    res.status(200).json({ correctPrompt });
  } else {
    res.status(400).json({ error: 'No suitable prompt found.' });
  }
}

// Helper functions
async function findCorrectPrompt(chattyPrompt, attempts, desiredOutput, promptHistory, req) {
  let chattyOutput;
  

  if (attempts >= MAX_ATTEMPTS) {
    return null;
  }


  chattyOutput = await chatty.call([
    new SystemChatMessage(`You are Chatty. You are an AI guesser.
      You receive information from Prompty. Who tells you what to do.
      `),
    new SystemChatMessage(`Respond only with the completion you want Checky to check.
    Checky will tell Prompty if you are right or wrong, and Prompty will suggest
    how to improve your output.`),
    new AIChatMessage(chattyPrompt)
  ]);

  console.log(`CHATTY:\n[Attempt ${attempts + 1}]:`, chattyOutput.text);

  const validationResult = await checky.call(
    formatCheckyInput(chattyOutput.text, desiredOutput, true)
  );

  console.log(
    `CHECKY:\nValidation result [Attempt ${attempts + 1}]:`,
    validationResult.text
  );

  const validationResultJson = JSON.parse(validationResult.text);
  const validationAction = validationResultJson.action;
  const validationIsCorrect = validationResultJson.isCorrect;

  console.log('DO:\n', validationAction);

  if (validationIsCorrect) {
    return chattyPrompt;
  } else {
    promptHistory.push(chattyPrompt);

    const action = validationAction;
    switch (action) {
      case 'suggest_prompt':
        const promptyOutput = await prompty.call(
          formatPromptyInput(chattyOutput.text, desiredOutput, promptHistory)
        );

        console.log(`PROMPTY:\n[Attempt ${attempts + 1}]:`, promptyOutput.text);
        return findCorrectPrompt(promptyOutput.text, attempts + 1, desiredOutput, promptHistory, req);

      case 'retry':
        const promptyOutputRetry = await prompty.call(
          formatPromptyInput(chattyOutput.text, desiredOutput, promptHistory)
        );

        console.log('PROMPTY', promptyOutputRetry.text);
        return
        findCorrectPrompt(promptyOutputRetry.text, attempts + 1, desiredOutput, promptHistory, req);
        break;

      case 'error':
        console.error(`Error in checky: ${validationResult.reason}`);
        return null;

      default:
        console.error(`Unknown action from checky: ${action}`);
        return null;
    }
  }
}

const formatCheckyInput = (chattyOutput, desiredOutput, analyze) => {
  return [
    new SystemChatMessage(`instructions: 'Checky, your role is to validate whether
    the generated output by Chatty '${chattyOutput}'
    matches the desired output: '${desiredOutput}'. Please follow these rules:'`),
    new SystemChatMessage(
      JSON.stringify(checkyValidationRules(desiredOutput, analyze))
    ),
    new SystemChatMessage(
      'Only reply with the appropriate action in the appropriate format. Do not explain, or add any other text.'
    )
  ];
};

const formatPromptyInput = (chattyOutput, desiredOutput, promptHistory) => {
  return [
    new SystemChatMessage(`instructions: 'Prompty, your role is to get Chatty,
    your fellow AI, to do as you ask. Get them to generate the desired output: ${desiredOutput}.'`),
    new AIChatMessage(`prompt history: ${promptHistory}`),
    new SystemChatMessage(
      JSON.stringify({
        content: 'suggest_prompt',
        data: {
          instructions: `Based on the generated output, the desired output: ${desiredOutput},
          and Chatty's current response: ${chattyOutput} guide Chatty to achieve an exact match between the desired
          output and actual output..
          `
        }
      })
    ),
    new SystemChatMessage(conceptMatch),
    new SystemChatMessage(
      `Respond only with text directed at Chatty [Incorrectness score: <incorrectness score>] .`
    )
  ];
};

const checkyValidationRules = (desiredOutput, analyze) => {
  return {
    data: {
      rules: [
        {
          if: 'output is correct',
          return: {
            isCorrect: true
          }
        },
        {
          if: "output doesn't meet criteria and a prompt suggestion is needed",
          return: {
            isCorrect: false,
            action: 'suggest_prompt',
            reason: 'The generated output does not meet the desired criteria.'
          }
        },
        {
          if: 'output is close but needs slight modification',
          return: {
            isCorrect: false,
            action: 'retry',
            reason:
              'The generated output is close but needs slight modification.'
          }
        },
        {
          if: 'an error occurs during validation',
          return: {
            isCorrect: false,
            action: 'error',
            reason: 'An error occurred during the validation process.'
          }
        },
        {
          // Default case added
          return: {
            isCorrect: false,
            action: 'error',
            reason: 'Unable to determine the appropriate action.'
          }
        }
      ]
    }
  };
};

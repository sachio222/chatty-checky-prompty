# Chatty, Checky, Prompty Algorithm

The Chatty, Checky, Prompty Algorithm works with three LLM models to generate, validate, and improve prompts in a collaborative manner. The algorithm establishes an interactive loop between the three models, with Chatty generating prompts, Checky validating them, and Prompty suggesting improvements.

## Dependencies

- `langchain/chat_models`
- `langchain/schema`
- `NextJS`

## Usage

1. Drop into @/pages/api folder
2. Start dev server
3. Send query as in example below.

### Example

http://localhost:3000/chatGAN?=userQuery="I'm singing in the rain, just singing in the rain"


The model can try to have Chatty either match the user input (use characterMatching) or match the concept that is trying to be produced (use conceptMatching).

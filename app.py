from flask import Flask, render_template, request
import tiktoken
import openai
import os

app = Flask(__name__)

# Set up OpenAI credentials and configuration
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # Your Azure OpenAI resource's endpoint value.
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

# Conversation setup
system_message = {"role": "system", "content": "You are a helpful assistant and you will end each response with a random emoji"}
max_response_tokens = 400  #100 tokens ~ 75 words so max_response_tokens = 2.5*75 = 200 words
token_limit = 4096 # (4096/100)*75= = 3072 words leaving 2800 words for prompt+history prompts and responses
conversation = []
conversation.append(system_message)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        if user_input == 'q':
            return render_template('index.html', chat_history=conversation)

        conversation.append({"role": "user", "content": user_input})
        conv_history_tokens = num_tokens_from_messages(conversation)

        while conv_history_tokens + max_response_tokens >= token_limit:
            del conversation[1]
            conv_history_tokens = num_tokens_from_messages(conversation)

        response = openai.ChatCompletion.create(
            engine="MyChatGPT35Turbo",  # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
            messages=conversation,
            temperature=0.7,
            max_tokens=max_response_tokens,
        )

        assistant_response = response['choices'][0]['message']['content']
        conversation.append({"role": "assistant", "content": assistant_response})

        return render_template('index.html', chat_history=conversation)
    else:
        return render_template('index.html', chat_history=conversation)

if __name__ == '__main__':
    app.run()

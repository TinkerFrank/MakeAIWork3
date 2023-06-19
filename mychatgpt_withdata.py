import torch
import gradio as gr
from torch import nn 
import cv2
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import tiktoken
import openai
import os
openai.api_type = "azure"
openai.api_version = "2023-05-15"
# Your Azure OpenAI resource's endpoint value .
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

system_message = {"role": "system", "content": "You are a helpful assistant."}
max_response_tokens = 250
token_limit = 4096
conversation = []
conversation.append(system_message)

AQLinfo = "x is the sum of the amount of 'Blotch Apple','Rot Apple' and 'Scab Apple'"
conversation.append({"role": "system", "content": AQLinfo})

AQLdata = {
  "AQL Class I": "x=0 the entire batch is classified as 'Normal Apple'",
  "AQL Class II ": "1<x<8 apples in the batch are NOT 'Normal Apple'",
  "AQL Class III": "8>x>15 apples in the batch are NOT 'Normal Apple'",
  "AQL CLass IV": "x>15 apples in the batchare NOT 'Normal Apple'"
}
conversation.append({"role": "system", "content": "{}".format(AQLdata)})

AQLinfo = "the accuracy of the classifier used is 80%"
conversation.append({"role": "system", "content": AQLinfo})

# Loading the classification model
modelresnet = torch.load('apple_resnet_classifier.pt',  map_location=torch.device('cpu'))
modelresnet.to(device)
modelresnet.eval()

# print('Enter image location:')
# image_url = input()
# img = Image.open(image_url)

# User input of sample location
print('Enter folder location:')
folder_url = input()
#folder_url = r"D:\apple_50sample"

transform_img_normal = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


def predict(folder_path):
    class_labels = ['Blotch Apple', 'Normal Apple', 'Rot Apple', 'Scab Apple']
    class_counts = [0,0,0,0]

    dataset = ImageFolder(folder_path, transform=transform_img_normal)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)

    with torch.no_grad():
        for data in dataset_loader:
            inputs, labels = data

            #weights in the loaded model are cuda casted 
            #cast the inputs also to cuda to make it work 
            inputs = inputs.to(device)

            out = modelresnet(inputs).to(device)
            _, predicted = torch.max(out.data, 1)
            for p in predicted:
                if p.item() == 0:
                    class_counts[0] += 1
                elif p.item() == 1:
                    class_counts[1] += 1
                elif p.item() == 2:
                    class_counts[2] += 1
                else:
                    class_counts[3] += 1

        label_counts_dict = {label: count for label, count in zip(class_labels, class_counts)}
        return label_counts_dict


# def predict(image):
#     img = image.resize((224, 224))
#     img = ToTensor()(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         out = modelresnet(img)
#         _, predicted = torch.max(out.data, 1)
#         probabilities = torch.nn.functional.softmax(out, dim=1)[0]
#         class_labels = ['Bad Apple', 'Normal Apple',
#                         'Rot Apple', 'Scab Apple']
#         values, indices = torch.topk(probabilities, 4)
#         confidences = {class_labels[i]: v.item() for i, v in zip(indices, values)}
#         print(confidences)
#         return confidences

def append_data(table):
    table_input = {"role": "system", "content": "{}".format(table)}
    print(table_input)
    conversation.append(table_input)
    print('table appended')

#x = predict(img)
x = predict(folder_url)
append_data(x)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


while (True):
    print('Input:')

    user_input = input("")

    if (user_input == 'q'):
        break

    conversation.append({"role": "user", "content": user_input})
    conv_history_tokens = num_tokens_from_messages(conversation)

    while (conv_history_tokens+max_response_tokens >= token_limit):
        del conversation[1]
        conv_history_tokens = num_tokens_from_messages(conversation)

    response = openai.ChatCompletion.create(
        # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        engine="MyChatGPT35Turbo",
        messages=conversation,
        temperature=.7,
        max_tokens=max_response_tokens,
    )

    conversation.append(
        {"role": "assistant", "content": response['choices'][0]['message']['content']})
    print("\n" + response['choices'][0]['message']['content'] + "\n")

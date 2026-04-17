import json
import os
import random

import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader,TensorDataset


nltk.download('punkt')
nltk.download('wordnet')

intents_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["hi", "hello", "hey"],
            "response": ["Hello!", "Hi there!", "Hey!"]
        },
        {
            "tag": "stocks",
            "patterns": ["stocks", "give me stocks"],
            "response": ["Here are some stock suggestions"]
        }
    ]
}

# Save file if it doesn't exist
if not os.path.exists("intents.json"):
    with open("intents.json", "w") as f:
        json.dump(intents_data, f, indent=4)

class ChatbotModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(ChatbotModel, self).__init__()
        self.fc1=nn.Linear(input_size,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x=self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x=self.fc3(x)
        return x

class ChatbotAssistant:
    def __init__(self,intents_path,function_mappings=None):
        self.intents_path = intents_path
        self.function_mapping = function_mappings

        self.documents=[]
        self.vocabulary=[]
        self.intents=[]
        self.intents_response=[]

        self.x=None
        self.y=None
        self.model=None

    def tokenize_and_lemmatize(self,text):
        lemmatizer=WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return[lemmatizer.lemmatize(word.lower()) for word in words]

    def bags_of_words(self,words):
        return[1 if word in words else 0 for word in self.vocabulary]
    def parse_intents(self):
        with open(self.intents_path) as f:
            intents_data=json.load(f)
            for intent in intents_data['intents']:
                tag=intent['tag']
                if tag not in self.intents:
                    self.intents.append(tag)
                    self.intents_response.append(intent['response'])

                    for pattern in intent['patterns']:
                        pattern_words=self.tokenize_and_lemmatize(pattern)
                        self.vocabulary.extend(pattern_words)
                        self.documents.append((pattern_words,tag))

                        self.vocabulary=sorted(set(self.vocabulary))

                    def prepare_data(self):
                        bags=[]
                        indices=[]

                        for words,tag in self.documents:
                            bag=self.bag_of_words(words)
                            intents_inddex=self.intents.index(tag)

                            bags.append(bag)
                            indices.append(intents_inddex)

                            self.x=np.array(bags)
                            self.y=np.array(indices)
                def train_model(self,epochs=100):
                    x_tensor=torch.tensor(self.x,dtype=torch.float)
                    y_tensor=torch.tensor(self.y,dtype=torch.long)

                    dataset=TensorDataset(x_tensor,y_tensor)
                    loader=DataLoader(dataset,batch_size=64,shuffle=True)

                    self.model=ChatbotModel(self.x.shape[1],len(self.intents))

                    criterion=nn.CrossEntropyLoss()
                    optimizer=optim.Adam(self.model.parameters(),lr=0.001)

                    for epoch in range(epochs):
                        total_loss=0

                        for batch_x,batch_y in loader:
                            optimizer.zero_grad()
                            outputs=self.model(batch_x)
                            loss=criterion(outputs,batch_y)
                            loss.backward()
                            optimizer.step()

                            total_loss+=loss.item()
                            print(f"Epoch {epoch+1}:Loss{total_loss/len(loader):4f}")

    def process_message(self,message):
        words=word_tokenize(message)
        bag=self.tokenize_and_lemmatize(message)

        bag_tensor=torch.tensor(bag,dtype=torch.float)

        with torch.no_grad():
            predictions=self.model.forward(bag_tensor)

            index=torch.argmax(predictions,dim=1)
            intent=self.intents_response[index]

            if self.function_mapping and intent in self.function_mapping:
                self.function_mapping[intent]()

                responses=self.intents_response.get[index]
                return random.choice(responses)
    def get_stocks():
                stocks = ['AAPL', 'META', 'NVDA', 'GS', 'MSFT']
                print("Stock picks:", random.sample(stocks, 3))

if __name__ == "__main__":
    assistant = ChatbotAssistant(
            "intents.json",
            function_mappings={"stocks": get_stocks}
        )

        assistant.parse_intents()
        assistant.prepare_data()
        assistant.train_model()

        while True:
            msg = input("You: ")

            if msg == "/quit":
                break

reply = assistant.process_message(msg)
print("Bot:", reply)

















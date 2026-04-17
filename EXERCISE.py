import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')


# ---------------- DATA INSIDE CODE ----------------
intents_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["hi", "hello", "hey", "good morning"],
            "responses": [
                "Hey 👋 I’m here to help!",
                "Hello 😊 What can I do for you?",
                "Hi there! Ask me anything."
            ]
        },
        {
            "tag": "goodbye",
            "patterns": ["bye", "goodbye", "see you"],
            "responses": [
                "Goodbye 👋",
                "See you later 😄",
                "Take care!"
            ]
        },
        {
            "tag": "stocks",
            "patterns": ["stocks", "give me stocks"],
            "responses": [
                "Here are some stock suggestions 📊"
            ]
        },
        {
            "tag": "fallback",
            "patterns": ["random", "????", "idk"],
            "responses": [
                "I didn’t understand that 😅",
                "Can you rephrase it?"
            ]
        }
    ]
}


# ---------------- MODEL ----------------
class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------- CHATBOT ----------------
class ChatbotAssistant:
    def __init__(self, intents):
        self.intents_data = intents

        self.lemmatizer = WordNetLemmatizer()

        self.vocabulary = []
        self.tags = []
        self.documents = []

        self.model = None

    # ---------- NLP ----------
    def tokenize(self, text):
        return word_tokenize(text)

    def lemmatize(self, words):
        return [self.lemmatizer.lemmatize(w.lower()) for w in words]

    def bag_of_words(self, words):
        return [1 if w in words else 0 for w in self.vocabulary]

    # ---------- PROCESS INTENTS ----------
    def prepare(self):
        for intent in self.intents_data["intents"]:
            tag = intent["tag"]

            if tag not in self.tags:
                self.tags.append(tag)

            for pattern in intent["patterns"]:
                tokens = self.lemmatize(self.tokenize(pattern))
                self.vocabulary.extend(tokens)
                self.documents.append((tokens, tag))

        self.vocabulary = sorted(set(self.vocabulary))

    # ---------- TRAIN ----------
    def train(self, epochs=100):
        X, y = [], []

        for words, tag in self.documents:
            X.append(self.bag_of_words(words))
            y.append(self.tags.index(tag))

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        self.model = ChatbotModel(len(self.vocabulary), len(self.tags))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0

            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # ---------- RESPONSE ----------
    def get_response(self, tag):
        for intent in self.intents_data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

        return "I didn’t understand that."

    # ---------- PREDICT ----------
    def predict(self, message):
        words = self.lemmatize(self.tokenize(message))
        bow = self.bag_of_words(words)

        x = torch.tensor([bow], dtype=torch.float32)

        with torch.no_grad():
            out = self.model(x)
            idx = torch.argmax(out).item()

        tag = self.tags[idx]
        return self.get_response(tag)


# ---------------- RUN ----------------
if __name__ == "__main__":
    bot = ChatbotAssistant(intents_data)

    bot.prepare()
    bot.train(epochs=50)

    print("\n🤖 Chatbot Ready! Type /quit to exit\n")

    while True:
        msg = input("You: ")

        if msg == "/quit":
            break

        reply = bot.predict(msg)
        print("Bot:", reply)
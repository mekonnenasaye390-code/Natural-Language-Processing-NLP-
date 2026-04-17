import nltk
import spacy
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

text="what shall i do####,@#%"
tokens=nltk.word_tokenize(text)
print('tokens:',tokens)
tags=nltk.pos_tag(tokens)
print('tags:',tags)
lower=text.lower()
print('lower:',lower)
clean_text=re.sub(r'[^\w\s]','',lower)
print('clean_text:',clean_text)

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ['eating', 'eats', 'ate']

stems = [stemmer.stem(word) for word in words]

print('stems:', stems)

import spacy
nlp=spacy.load('en_core_web_sm')
doc=nlp(text)
print('doc',doc)
print(type(doc))
for token in doc:
    print(token.text,token.pos_)
for sent in doc.sents:
    print(sent.text)
for ent in doc.ents:
    print(ent.text,ent.label_)
    print(nlp.analyze_pipes())
from transformers import BertTokenizer,BertModel
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model=BertModel.from_pretrained('bert-base-uncased')

text="Life is a full of problem"
inputs=tokenizer(text,return_tensors="pt")
print("tokens",inputs["input_ids"])
print('attention mask',inputs["attention_mask"])
outputs=model(**inputs)
last_hidden_states=outputs.last_hidden_state
print("outputs",last_hidden_states.shape)

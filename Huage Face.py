from transformers import pipeline
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
classifier=pipeline("sentiment-analysis")
result=classifier("I love learning NLP")
#print(result)
model_name="distilbert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForSequenceClassification.from_pretrained(model_name)
#print(tokenizer)
#print(model)
text=("I love AI  coding so how to the code")
inputs=tokenizer(text,return_tensors="pt")
#print(inputs)
outputs=model(**inputs)
#print(outputs)
probabilities=torch.nn.functional.softmax(outputs.logits,dim=-1)
print(probabilities)
generator=pipeline("text_generation")
print(generator("AI will change the world",max_length=30))

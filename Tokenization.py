import spacy
from spacy import displacy
nlp=spacy.load('en_core_web_sm')
print(nlp.pipe_names)

with open(r"C:\Users\hp\OneDrive\NLP Codinng or CHATBOAT\wiki_us.txt","r") as f:
    text=f.read()
    print(text)
    doc=nlp(text)
    print('doc',doc)
    print(len(doc))
    print(len(text))
for token in text[0:10]:
    print(token)
for token in doc[0:10]:
    print(token)
for sent in doc.sents:
    print(sent)
sentence1=list(doc.sents)[0]
print(sentence1)
#print(displacy.render(sentence1))

for ent in doc.ents:
    print(ent.text,ent.label_)
#print(displacy.render(doc,style="ent"))

doc1=nlp("life is a full of problem")
doc2=nlp("problem iss not permanent")
#print(doc1,doc2,doc1.similarity(doc2))
print(nlp.analyze_pipes())


import re
text="What is coding and studing an Example!!!1234"
text=re.sub(r'[^\w\s]','',text).lower()
print(text)
doc=nlp(text)
tokens=[token.lemma_ for token in doc if not token.is_stop]
print(tokens)





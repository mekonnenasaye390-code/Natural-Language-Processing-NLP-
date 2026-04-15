import nltk
import nltk.corpus
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

text="I Love coding AI, but sometime very hard !!!"
clean_text=re.sub('[^a-zA-Z]',' ',text)
print('clean_text',clean_text)
text_lower=text.lower()
print('text_lower',text_lower)
word_tokens=word_tokenize(clean_text)
print('word_tokens',word_tokens)
stop_words=set(stopwords.words('english') )
print('stop_words',stop_words)
filtered=[word for word in word_tokens if word not in stop_words]
print('filtered',filtered)
lemmatizer=WordNetLemmatizer()
lemmas=[lemmatizer.lemmatize(word) for word in filtered]
print('lemmas',lemmas)
vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(lemmas)
print('x',x)
print(x.toarray())
print(vectorizer.get_feature_names_out())
model=MultinomialNB().fit(x,lemmas)
print('model',model)
test=vectorizer.transform([text])
print('test',test)
y_true=[1,0,1]
y_pred=model.predict(x)
print('y_pred',y_pred)
tags=nltk.pos_tag(lemmas)
print('tags',tags)








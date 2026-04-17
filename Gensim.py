from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
texts=[
    "Life is a full",
    "what shall i do",
    "I love writing AI code",

]
labels=[1,0,1,0]
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(texts)
print(x.shape)
print(vectorizer.vocabulary_)
print(x.toarray())
x_train,x_test,y_train,y_test=train_test_split(x,labels,test_size=0.2,random_state=42)
model=MultinomialNB()
print(model.fit(x_train,y_train))
sample = ["I love it"]
sample_vec = vectorizer.transform(sample)

print(sample_vec.toarray())
prediction = model.predict(sample_vec)
print(prediction)
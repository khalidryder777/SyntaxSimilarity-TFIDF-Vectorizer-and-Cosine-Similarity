

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#TfidfVectorizer class imported from Scikit learn to compute TFIDF values from a set of docs

docs=[ "Sachin is considered to be one of the greatest cricket players",
          "Federer is considered one of the greatest tennis players",
          "Nadal is considered one of the greatest tennis players",
          "Virat is the captain of the  Indian cricket team"
          
         ]


##

#creating Vectorizer object from TfidfVectorizer class by passing the appro parameters
# 
vectorizer = TfidfVectorizer(analyzer='word',norm=None, use_idf=True,smooth_idf=True)
tfIdfMat  = vectorizer.fit_transform(docs)

feature_names = sorted(vectorizer.get_feature_names())

#passing the list of docs in the corpus to the fit transform method of the vectorizer object
#fit transform method does internal computation of tfidf values

docList=['Doc 1','Doc 2','Doc 3','Doc 4']
skDocsTfIdfdf = pd.DataFrame(tfIdfMat.todense(),index=sorted(docList),  columns=feature_names)
print(skDocsTfIdfdf)


#compute cosine similarity to compute the similarity between the docs in corpus
csim = cosine_similarity(tfIdfMat,tfIdfMat) #passing TFIDF values of the docs from corpus

csimDf = pd.DataFrame(csim,index=sorted(docList),columns=sorted(docList)) #Designing the table

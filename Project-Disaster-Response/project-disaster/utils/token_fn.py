import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download(['punkt','stopwords','wordnet'])
lemmatizer=WordNetLemmatizer()
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    '''
    Reads table from db file
    INPUT - text- sentence
    OUTPUT - 
            tokens - list of tokens from the cleansified sentence
    '''
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    text=re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    tokens=word_tokenize(text)
    new_tokens=[]
    for token in tokens:
        if token not in stopwords.words("english"):
            #new_tokens.append(stemmer.stem(token))
            new_tokens.append(lemmatizer.lemmatize(token,pos='n').lower().strip())
    new_tokens2=[]
    for token in new_tokens:
        new_tokens2.append(lemmatizer.lemmatize(token,pos='v').lower().strip())
    
    return new_tokens2
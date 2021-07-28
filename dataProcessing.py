import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re,string,unicodedata

imdb_data=pd.read_csv('Data/IMDB_Dataset.csv')

tokenizer=ToktokTokenizer()
stopword_list=stopwords.words('english')
stop=set(stopword_list)

def cleaner(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'[^a-zA-z\s]','',text)
    tokens = tokenizer.tokenize(text)
    tokens = [token.lower().strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    text = ' '.join(filtered_tokens)
    ps=nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    #wordnet_lemmatizer = WordNetLemmatizer()
    #text =  ' '.join([wordnet_lemmatizer.lemmatize(word) for word in tokens])
    return text

imdb_data['review']=imdb_data['review'].apply(cleaner)

imdb_data.to_csv('data/cleanedReviews.csv',index=False)
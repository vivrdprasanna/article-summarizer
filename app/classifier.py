import re, json, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()
file = r'train.json'
with open(file) as train_file:
    dict_train = json.load(train_file)

# DATA CLEANING
    
id_ = []
cuisine = []
ingredients = []
for i in range(len(dict_train)):
    id_.append(dict_train[i]['id'])
    cuisine.append(dict_train[i]['cuisine'])
    ingredients.append(dict_train[i]['ingredients'])
    
import pandas as pd
df = pd.DataFrame({'id':id_, 
                   'cuisine':cuisine, 
                   'ingredients':ingredients})

new = []
for s in df['ingredients']:
    s = ' '.join(s)
    new.append(s)    
    
df['ing'] = new

def list_ingredients_to_string(lst):
    string = ''
    for element in lst:
        string += element + ' '
    return string

def clean_ingredients_string(s, debug = False):
    s = re.sub(r'[^\w\s]', '' ,s) # Remove punctuations
    s = re.sub(r"(\d)",  "", s)  # Remove digits
    s = re.sub(r'\([^)]*\)', '', s) # Remove content inside paranthesis
    s = re.sub(u'\w*\u2122', '', s) # Remove Brand Name   
    s = s.lower()  #Convert to lowercase
    # Stemming
    words = word_tokenize(s)
    word_ps = []
    for w in words:
        word_ps.append(ps.stem(w))
    s=' '.join(word_ps)
    return s


ingredients_cleaned = []
counter = 1
for s in df['ing']:
    cleaned_string = clean_ingredients_string(s)
    ingredients_cleaned.append(cleaned_string)
    if counter % (39000/6) == 0:
        print(counter, sep = ' ')
    counter += 1
df['ingredients_cleaned'] = ingredients_cleaned

df = df[['cuisine', 'ingredients_cleaned']]
print(df.head(5))
print("CLEANING DONE")

# VECTORIZING

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ingredients_cleaned'])

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df['cuisine'])
Y = label_encoder.transform(df['cuisine']) 

cuisine_map = {'0':'brazilian', '1':'british', '2':'cajun_creole', '3':'chinese', '4':'filipino', '5':'french', '6':'greek', '7':'indian', '8':'irish', '9':'italian', '10':'jamaican', '11':'japanese', '12':'korean', '13':'mexican', '14':'moroccan', '15':'russian', '16':'southern_us', '17':'spanish', '18':'thai', '19':'vietnamese'}

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)

# DEFINE/TRAIN THE MODEL
from sklearn.svm import SVC
from sklearn import svm

lin_clf = svm.LinearSVC(C=1, verbose = 1)
print("FITTING MODEL")
lin_clf.fit(X_train, y_train)
y_pred = lin_clf.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test,y_pred)*100, 4))
result = pd.DataFrame({'Actual Cuisine':y_test, 'Predicted Cuisine':y_pred})

def predict(single_input):
    single_input_cleaned = clean_ingredients_string(single_input, True)
    vectorized_string = vectorizer.transform([single_input_cleaned])
    return "prediction: " + cuisine_map[str(lin_clf.predict(vectorized_string[0])[0])]
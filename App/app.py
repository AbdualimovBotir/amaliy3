import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Download the NLTK stopwords if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model from the current directory
tfidf = pickle.load(open('vectorizer2.pkl','rb'))
model = pickle.load(open('model2.pkl','rb'))

# Streamlit app interface
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input text
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the text
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict the class
    result = model.predict(vector_input)[0]
    
    # 4. Display the result
    if result == 1:
        st.header("Spam")
    elif result == 0:
        st.header("Not Spam")

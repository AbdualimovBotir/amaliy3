import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# PorterStemmer ni inicializatsiya qilish
ps = PorterStemmer()

# Agar NLTK resurslari (stopwords va punkt) hali yuklanmagan bo'lsa, ularni yuklash
nltk.download('punkt')
nltk.download('stopwords')

# Matnni oldindan ishlash va transformatsiya qilish uchun funksiya
def transform_text(text):
    text = text.lower()  # Matnni kichik harflarga o'zgartirish
    text = nltk.word_tokenize(text)  # Matnni so'zlarga ajratish

    y = []
    for i in text:
        if i.isalnum():  # Faqat alifbo va raqamlar qoldiriladi
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # To'liq stop so'zlar va punktuatsiya belgilaridan tozalash
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Har bir so'zni stemlash

    return " ".join(y)  # Olingan so'zlarni birlashtirish va qaytarish

# Model va vektorizerni yuklash (models papkasi ichida joylashgan)
tfidf = pickle.load(open('models/vectorizer2.pkl', 'rb'))  # Vektorizerni yuklash
model = pickle.load(open('models/model2.pkl', 'rb'))  # Modelni yuklash

# Streamlit interfeysi
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")  # Foydalanuvchidan matn kiritish

if st.button('Predict'):  # Tugma bosilganda bashorat qilish
    # 1. Matnni oldindan ishlash
    transformed_sms = transform_text(input_sms)
    
    # 2. Matnni vektorizatsiya qilish
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Natija chiqarish
    result = model.predict(vector_input)[0]
    
    # 4. Natijani ko'rsatish
    if result == 1:
        st.header("Spam")  # Agar natija 1 bo'lsa, Spam deb belgilaydi
    elif result == 0:
        st.header("Not Spam")  # Agar natija 0 bo'lsa, Spam emas deb belgilaydi

import nltk

# for deployment at streamlit server
nltk.download('punkt_tab')
nltk.download('stopwords') 
nltk.download('wordnet')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

# Title for the Streamlit app
st.title("Welcome to NLP Text Preprocessing")

# Input field for the corpus
st.subheader("Corpus!!! ")
corpus = st.text_area("Enter corpus (paragraph):")

# Tokenization options in the sidebar
st.sidebar.subheader("Select Options ")
tokenization_type = st.sidebar.radio("",
                 ('Sentence Tokenization','Word Tokenization', 'Stemming', 'Lemmatization', 'Stopword Removal','Bag Of Words'))


# Check if the corpus is not empty
if corpus:

    # Sentence Tokenization
    if tokenization_type == "Sentence Tokenization": 
        sentence_tokens = sent_tokenize(corpus)
        st.subheader("Sentence Tokens:")
        for i, tokens in enumerate(sentence_tokens, 1):
            st.write(f"{i}. {tokens}")

    # Word Tokenization
    if tokenization_type == "Word Tokenization":
        word_tokens = word_tokenize(corpus)
        st.subheader("Word Tokens:")
        for i, tokens in enumerate(word_tokens, 1):
            st.write(f"{i}. {tokens}")

    # Stemming
    if tokenization_type == "Stemming":
        word_tokens = word_tokenize(corpus)
        stemmer = PorterStemmer()
        st.subheader("Stemming:")
        for i, tokens in enumerate(word_tokens, 1):
            st.write(f"{i}.  {tokens}  --->  {stemmer.stem(tokens)}")

    # Lemmatization
    if tokenization_type == "Lemmatization":
        word_tokens = word_tokenize(corpus)
        lemmatizer = WordNetLemmatizer()
        st.subheader("Lemmatization:")
        for i, tokens in enumerate(word_tokens, 1):
            st.write(f"{i}.  {tokens}  --->  {lemmatizer.lemmatize(tokens)}")

    # Stopwords Removal
    if tokenization_type == "Stopword Removal":
        stop_words = set(stopwords.words('english'))  # Get the list of stopwords
        word_tokens = word_tokenize(corpus)  # Tokenize the corpus
        filtered_words = [word for word in word_tokens if word.lower() not in stop_words]  # Remove stopwords

        # Join filtered words to form the new paragraph
        filtered_paragraph = " ".join(filtered_words)

        # Display the paragraph after removing stopwords
        st.subheader("Paragraph After Stopword Removal:")
        st.write(filtered_paragraph)

    # Bag Of Words
    if tokenization_type == "Bag Of Words":
        # Tokenize the corpus into sentences
        sentences = sent_tokenize(corpus)
        
        vectorizer = CountVectorizer(lowercase=True, analyzer='word', max_features=None)
        vectorizer.fit([corpus])  
        
        # Display the features (unified vocabulary)
        features = vectorizer.get_feature_names_out()
        st.subheader("Unified Vocabulary:")
        st.write(f"Features: {features}")
        
        # Process each sentence
        st.subheader("Bag of Words for Each Sentence:")
        for i, sent in enumerate(sentences, 1):
            vector = vectorizer.transform([sent])
            st.write(f"Sentence {i}: {sent}")
            st.write(f"Vector: {vector.toarray()}")

else:
    st.warning("Please enter a paragraph to tokenize.")

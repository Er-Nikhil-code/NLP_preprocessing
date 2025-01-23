import nltk

# for deployment at streamlit server
nltk.download('punkt_tab')
nltk.download('stopwords') 
nltk.download('wordnet')

# importing necessary libraries
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

# Options for pre-processing
tokenization_type = st.selectbox(
    "Select pre-processing type",
    ['Select an option', 'Sentence Tokenization', 'Word Tokenization', 'Stemming', 'Lemmatization', 'Stopword Removal', 'Bag Of Words']
)

submit_button = st.button('Submit')

# Check if the corpus is not empty and submit button is pressed
if corpus and submit_button:

    # Sentence Tokenization
    if tokenization_type == "Sentence Tokenization": 
        sentence_tokens = sent_tokenize(corpus)
        st.subheader("Sentence Tokens:")
        for i, tokens in enumerate(sentence_tokens, 1):
            st.write(f"{i}. {tokens}")
        st.subheader(f"About {tokenization_type}")
        st.write("Sentence tokenization is the process of splitting text into individual sentences.")

    # Word Tokenization
    if tokenization_type == "Word Tokenization":
        word_tokens = word_tokenize(corpus)
        st.subheader("Word Tokens:")
        for i, tokens in enumerate(word_tokens, 1):
            st.write(f"{i}. {tokens}")
        st.subheader(f"About {tokenization_type}")
        st.write("Word tokenization is the process of splitting a given text into individual words or terms, which are referred to as tokens.")

    # Stemming
    if tokenization_type == "Stemming":
        word_tokens = word_tokenize(corpus)
        stemmer = PorterStemmer()
        st.subheader("Stemming:")
        for i, tokens in enumerate(word_tokens, 1):
            st.write(f"{i}.  {tokens}  --->  {stemmer.stem(tokens)}")
        st.subheader(f"About {tokenization_type}")
        st.write("Stemming in Natural Language Processing (NLP) is the process of reducing a word to its base or root form by removing prefixes or suffixes.")

    # Lemmatization
    if tokenization_type == "Lemmatization":
        word_tokens = word_tokenize(corpus)
        lemmatizer = WordNetLemmatizer()
        st.subheader("Lemmatization:")
        for i, tokens in enumerate(word_tokens, 1):
            st.write(f"{i}.  {tokens}  --->  {lemmatizer.lemmatize(tokens,pos="v")}") # POS is set to verb
        st.subheader(f"About {tokenization_type}")
        st.write("Lemmatization in Natural Language Processing (NLP) is the process of transforming a word into its canonical or dictionary form, known as its lemma. Unlike stemming, which may simply truncate words to their base forms without considering their actual meaning, lemmatization takes into account the morphological analysis of the word, including its part of speech and context, to ensure that the resulting lemma is a valid word.")

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
        st.subheader(f"About {tokenization_type}")
        st.write("""Stop word removal in Natural Language Processing (NLP) is the process of eliminating common words from a text that carry little to no meaningful information or semantic value for a specific task. These words, known as stop words, are typically high-frequency words such as articles, prepositions, and conjunctions (e.g., "is," "the," "and," "of"). Removing them helps reduce noise in the data and focuses analysis on more significant terms.""")

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
        st.subheader(f"About {tokenization_type}")
        st.write("""The Bag of Words (BoW) model in Natural Language Processing (NLP) is a simple and widely used method for text representation. It converts text into a numerical format by treating a document as a collection (or "bag") of words, disregarding grammar, word order, and syntax, while focusing solely on word frequency or presence.""")
    if tokenization_type == "Select an option":
        st.warning("Select the valid choices!!")
elif not corpus.split():
    st.warning("Please enter a paragraph to tokenize!!")

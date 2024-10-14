from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD


nltk.download('stopwords')
stop_words = stopwords.words('english')

app = Flask(__name__)

#Fetch dataset, initialize vectorizer and LSA here

# A simple function to preprocess text
def preprocess_text(text):
    return ' '.join([word.lower() for word in text.split() if word.lower() not in stop_words])
#fetching the dataset
newsgroups = fetch_20newsgroups(subset='all')
# Apply the preprocessing function to the newsgroups data... come back here this could be an ish
processed_data = [preprocess_text(text) for text in newsgroups['data']]
#initializing the vectorizer
vectorizer = TfidfVectorizer(max_df=0.05, max_features=1000, ngram_range=(1, 2))
#Creating the term-document matrix... same ish here with processed data
term_doc_matrix = vectorizer.fit_transform(processed_data)
#Get the terms (feature names) from the vectorizer
terms = vectorizer.get_feature_names_out()

#SVD to the matrix to reduce the dimensionality
#Define the number of components
num_components = 50

svd_model = TruncatedSVD(n_components=num_components)

#Fit the SVD model
svd_matrix = svd_model.fit_transform(term_doc_matrix)




def search_engine(query,top_n=5):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    #Implement search engine here
    # return documents, similarities, indices 

    # Step 1: Transform the query into a TF-IDF vector
    query_tfidf = vectorizer.transform([query])

    # Step 2: Compute cosine similarity between the query and all documents
    cosine_sim = cosine_similarity(query_tfidf, term_doc_matrix).flatten()

    # Step 3: Get the indices of the top N similar documents.... ish here?
    top_n_indices = np.argsort(cosine_sim)[-top_n:][::-1]  # Sort in descending order

    # Step 4: Get the top N documents and their similarity scores
    top_n_docs = [processed_data[i] for i in top_n_indices]
    top_n_sims = [cosine_sim[i] for i in top_n_indices]
    
    return top_n_docs, top_n_sims, top_n_indices


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)

# Example usage:
query = "machine learning algorithms"
top_docs, top_sims, top_indices = search_documents(query, vectorizer, term_doc_matrix, newsgroups['data'])
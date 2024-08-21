from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.feature_extraction import text
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import spacy

OPENAI_APIKEY = st.secrets['OPENAI_APIKEY']
TAGALOG_STOP_WORDS = set("applause nga ug eh yun yan yung kasi ko akin aking ako alin am amin aming ang ano anumang apat at atin ating ay bababa bago bakit bawat bilang dahil dalawa dapat din dito doon gagawin gayunman ginagawa ginawa ginawang gumawa gusto habang hanggang hindi huwag iba ibaba ibabaw ibig ikaw ilagay ilalim ilan inyong isa isang itaas ito iyo iyon iyong ka kahit kailangan kailanman kami kanila kanilang kanino kanya kanyang kapag kapwa karamihan katiyakan katulad kaya kaysa ko kong kulang kumuha kung laban lahat lamang likod lima maaari maaaring maging mahusay makita marami marapat masyado may mayroon mga minsan mismo mula muli na nabanggit naging nagkaroon nais nakita namin napaka narito nasaan ng ngayon ni nila nilang nito niya niyang noon o pa paano pababa paggawa pagitan pagkakaroon pagkatapos palabas pamamagitan panahon pangalawa para paraan pareho pataas pero pumunta pumupunta sa saan sabi sabihin sarili sila sino siya tatlo tayo tulad tungkol una walang ba eh kasi lang mo naman opo po si talaga yung".split())
EMBEDDING_MODEL = 'text-embedding-3-large'  # 'text-embedding-3-small'
SPACY_MODEL = spacy.load(os.path.join(os.getcwd(), 'en_core_web_sm/en_core_web_sm-3.7.1')) # 'en_core_web_lg'
ENTITY_LABELS = ['PERSON', 'EVENT', 'DATE', 'GPE', 'ORG', 'FAC', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL']
CANDIDATE_LABELS = ['Economic Growth', 'Healthcare Reform', 'Education Initiatives', 'Infrastructure Development', 'Environmental Policies', 'Agricultural Support', 'Employment and Labor', 'Social Welfare Programs', 'Foreign Relations', 'Public Safety and Security']
SUPPORTED_LANGUAGES_T = ['Finnish', 'Tagalog', 'Cebuano', 'Ilocano', 'Hiligaynon']

APP_NAME = 'SENTINEL: Semantic Evaluation and Natural Text Intelligence Learning System'
APP_DESC = ' `by @Team IntelliNews`'
ABOUT_SENTINEL_1 = """SENTINEL is a powerful document analysis and comparison tool, driven by cutting-edge Large Language Models (LLMs) and advanced Natural Language Processing (NLP) technologies. It excels in conducting semantic evaluations to uncover similarities, differences, and nuanced relationships within textual data. Whether analyzing documents for investigative journalism, content comparison, or extracting key insights, SENTINEL delivers precise and actionable results."""
ABOUT_SENTINEL_2 = """Ideal for newsrooms and investigative journalists, SENTINEL enhances research capabilities by swiftly identifying patterns, sentiments, and critical information buried within extensive text corpora. Its intelligent learning system continuously refines accuracy, ensuring reliable and efficient analysis across diverse document types and sources. SENTINEL empowers users to uncover hidden connections and trends, making it an indispensable tool for driving informed decisions and impactful storytelling."""
K = 10
APP_DOCS = os.path.join(os.getcwd(), 'data/documents')
DF_CSV = os.path.join(os.getcwd(), 'data/intellinews.csv')
DB_PATH = os.path.join(os.getcwd(), 'data/intellinews.db')
COLLECTION_NAME = "intellinews"

        #     Recommendations: Recommendations / Action items
        # """
COLOR_BLUE = "#0c2e86"
COLOR_RED = "#a73c07"
COLOR_YELLOW = "#ffcd34"
COLOR_GRAY = '#f8f8f8'

scroll_back_to_top_btn = f"""
<style>
    .scroll-btn {{
        position: absolute;
        border: 2px solid #31333f;
        background: {COLOR_GRAY};
        border-radius: 10px;
        padding: 2px 10px;
        bottom: 0;
        right: 0;
    }}

    .scroll-btn:hover {{
        color: #ff4b4b;
        border-color: #ff4b4b;
    }}
</style>
<a href="#government-processease">
    <button class='scroll-btn'>
        Back to Top
    </button>
</a>
"""

FEEDBACK_FACES = {
    "😀": ":smiley:", "🙂": ":sweat_smile:", "😐": ":neutral_face:", "🙁": ":worried:", "😞": ":disappointed:"
}


#######################################################################################################

# @st.cache_data()
def init_data():
    df = pd.DataFrame(columns=['url', 'title', 'speech'])
    try:
        df = pd.read_csv(DF_CSV)
    except:
        pass
    return df
#######################################################################################################

def get_openai_client():
    client = OpenAI(api_key=OPENAI_APIKEY)
    return client
#######################################################################################################

def init_chroma_db(collection_name, db_path):
    # Create a Chroma Client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Create an embedding function
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_APIKEY, model_name=EMBEDDING_MODEL)

    # Create a collection
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    return collection
#######################################################################################################

def semantic_search(Q, k=K, collection=None,  titles=[]):
    n_K = len(titles) * k
    results = collection.query(
        query_texts=[Q], # Chroma will embed this for you
        n_results=n_K, # how many results to return,
        where={ 'title': {'$in': titles} }
    )
    return results

# def semantic_search(Q, k=5, collection=None):
#     # Query the collection
#     results = collection.query(
#         query_texts=[Q], # Chroma will embed this for you
#         n_results=k # how many results to return
#     )
#     return results
#######################################################################################################

def upsert_documents_to_collection(collection, documents):
    # Every document needs an id for Chroma
    last_idx = len(collection.get()['ids'])
    ids = list(f'id_{idx+last_idx:010d}' for idx, _ in enumerate(documents))
    docs = list(map(lambda x: x.page_content, documents))
    mets = list(map(lambda x: x.metadata, documents))

    # Update/Insert some text documents to the db collection
    collection.upsert(ids=ids, documents=docs,  metadatas=mets)
#######################################################################################################

def generate_response(task, prompt, llm, temperature=0.):
    response = llm.chat.completions.create(
        # model='gpt-3.5-turbo',
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': f"Perform the specified task: {task}"},
            {'role': 'user', 'content': prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content
#######################################################################################################

def generate_summarization(doc, llm):
    task = 'Text Summarization'
    prompt = f"Summarize this document:\n\n{doc}"
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_translation(doc, target_lang, llm):
    task = 'Text Translation'
    prompt = f"Translate this document from English to {target_lang}:\n\n{doc}\n\n\nOnly respond with the translation."
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_topic_labels(doc, llm, top_k=K):
    task = 'Topic Modeling or keyword extraction'
    prompt = f"Extract and list the top {top_k} main keywords in this document:\n\n{doc}"
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_sentiment_analysis(doc, llm):
    task = 'Sentiment Analysis'
    prompt = f"Classify the sentiment analysis of this document:\n\n{doc}\n\n\n Use labels: Positive, Negative, Neutral, Mixed"
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_document_analysis(Q, titles, texts, llm, advanced_prompt):
    task = 'Document analysis and comparison'

    doc_input = 'Each document has a title and content and is delimited by triple backticks.'
    for i in range(len(titles)):
        doc_input += f"""
        
        ```Document title: {titles[i]} Content: {texts[i]}```
        """

    prompt = advanced_prompt.replace('{{QUESTION}}', Q).replace('{{DOCUMENTS}}', doc_input).replace('{{TITLE_0}}', titles[0]).replace('{{TITLE_1}}', titles[1])
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_response_to_question(Q, text, titles, llm):
    """Generalized function to answer a question"""
    prompt = f"""
    Provide the answer on {Q} based on {', '.join(titles)} given this document:\n\n{text}.

    You should only respond based on the given documents. if you don't know the answer, just respond you don't know the answer. Don't give more than what is asked. Only answer the questions directly related to the {', '.join(titles)} and the given report. If not directly stated in the report, say that and don't give assumptions.

    For the answer, you would include a reference to a phrase where you have found the answer. e.g. "Source: Document 0 Title: {titles[0]}" or "Sources: Document 0 and 1; Titles: {titles[0]} and {titles[1]}".
    """
    response = generate_response(Q, prompt, llm)
    return response
#######################################################################################################

def ask_query(Q, titles, llm, k=15, collection=None):
    """Function to go from question to query to proper answer"""
    # Get related documents
    results = semantic_search(Q, k, collection, titles=titles)

    # Get the text of the documents
    # text = query_result['documents'][0][0] TODO
    text = ''
    for t in results['documents'][0]:
        text += t

    # Pass into GPT to get a better formatted response to the question
    response = generate_response_to_question(Q, text, titles, llm=llm)
    # return Markdown(response)
    return response
#######################################################################################################

def plot_wordcloud(df, column):
    # Data with filled of additonal stop words
    my_stop_words = list(text.ENGLISH_STOP_WORDS.union(TAGALOG_STOP_WORDS))

    # Fit vectorizers
    count_vectorizer = CountVectorizer(stop_words=my_stop_words)
    cv_matrix = count_vectorizer.fit_transform(df[column])

    tfidf_vectorizer = TfidfVectorizer(stop_words=my_stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])

    # Create dictionaries for word cloud
    count_dict = dict(zip(count_vectorizer.get_feature_names_out(),
                                cv_matrix.toarray().sum(axis=0)))

    tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(),
                                tfidf_matrix.toarray().sum(axis=0)))

    # Create word cloud and word frequency visualization
    count_wordcloud = (WordCloud(width=800, height=400, background_color='black')
                    .generate_from_frequencies(count_dict))

    tfidf_wordcloud = (WordCloud(width=800, height=400, background_color='black')
                    .generate_from_frequencies(tfidf_dict))

    # Plot the word clouds and word frequency visualizations
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(count_wordcloud, interpolation='bilinear')
    plt.title('Count Vectorizer Word Cloud')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(tfidf_wordcloud, interpolation='bilinear')
    plt.title('TF-IDF Vectorizer Word Cloud')
    plt.axis("off")

    plt.tight_layout()
    plt.show();
    return fig
#######################################################################################################

def save_uploadedfile(uploadedfile):
    if not os.path.exists(APP_DOCS):
        os.makedirs(APP_DOCS)
    file_path = uploadedfile.name

    with open(os.path.join(APP_DOCS, file_path), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path
#######################################################################################################

def _submit_feedback(feedback, *args, **kwargs):
    question = kwargs['question']
    llm_response = kwargs['llm_response']
    documents = kwargs['documents']
    feedback_sent = kwargs["feedback_time"]
    try:
        with open("data/feedback.txt", "a") as f:
            line = '================================================================================================================================================\n\n'
            reax = FEEDBACK_FACES.get(feedback['score'])
            text = f"{line}*Feedback was sent at {feedback_sent}*\n\n**Documents:** {documents}\n\n**QUESTION:** {question}\n\n**RESPONSE:** {llm_response}\n\n**FEEDBACK:**\n> Reaction: {reax}\n\n> Comment: {feedback['text']}\n\n"
            f.write(text)
            st.toast("Thank you for the feedback!", icon='🎉')
    except:
        pass
#######################################################################################################

def display_feedback():
    if os.path.exists("data/feedback.txt"):
        with open("data/feedback.txt", "r") as f:
            st.markdown(f.read())

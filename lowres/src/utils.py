__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#### Uncomment the code above in Production Environment
#######################################################################################################

import streamlit as st
from streamlit_feedback import streamlit_feedback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import spacy
import spacy_streamlit
from wordcloud import WordCloud
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from src.prompts import *

# # Development Environment
# load_dotenv()
# OPENAI_APIKEY = os.environ['OPENAI_APIKEY']
# IS_PROD = False

# Production Environment
OPENAI_APIKEY = st.secrets['OPENAI_APIKEY']
IS_PROD = True

PROJ_DIR = 'lowres/' if IS_PROD else ''
EMBEDDING_MODEL = 'text-embedding-3-large'
SPACY_MODEL = spacy.load(os.path.join(os.getcwd(), f'{PROJ_DIR}en_core_web_sm/en_core_web_sm-3.7.1')) # change to 'en_core_web_lg' if hosted on a server with more resources
ENTITY_LABELS = ['PERSON', 'EVENT', 'DATE', 'GPE', 'ORG', 'FAC', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL']
CANDIDATE_LABELS = ['Economic Growth', 'Healthcare Reform', 'Education Initiatives', 'Infrastructure Development', 'Environmental Policies', 'Agricultural Support', 'Employment and Labor', 'Social Welfare Programs', 'Foreign Relations', 'Public Safety and Security']
SUPPORTED_LANGUAGES_T = ['English', 'Finnish', 'Tagalog', 'Cebuano', 'Ilocano', 'Hiligaynon']
TAGALOG_STOP_WORDS = set("applause nga ug eh yun yan yung kasi ko akin aking ako alin am amin aming ang ano anumang apat at atin ating ay bababa bago bakit bawat bilang dahil dalawa dapat din dito doon gagawin gayunman ginagawa ginawa ginawang gumawa gusto habang hanggang hindi huwag iba ibaba ibabaw ibig ikaw ilagay ilalim ilan inyong isa isang itaas ito iyo iyon iyong ka kahit kailangan kailanman kami kanila kanilang kanino kanya kanyang kapag kapwa karamihan katiyakan katulad kaya kaysa ko kong kulang kumuha kung laban lahat lamang likod lima maaari maaaring maging mahusay makita marami marapat masyado may mayroon mga minsan mismo mula muli na nabanggit naging nagkaroon nais nakita namin napaka narito nasaan ng ngayon ni nila nilang nito niya niyang noon o pa paano pababa paggawa pagitan pagkakaroon pagkatapos palabas pamamagitan panahon pangalawa para paraan pareho pataas pero pumunta pumupunta sa saan sabi sabihin sarili sila sino siya tatlo tayo tulad tungkol una walang ba eh kasi lang mo naman opo po si talaga yung".split())
APP_NAME = 'SENTINEL: Semantic Evaluation and Natural Text Intelligence Learning System'
APP_DESC = ' `by @Team IntelliNews`'
ABOUT_SENTINEL_1 = """SENTINEL is a powerful document analysis and comparison tool, driven by cutting-edge Large Language Models (LLMs) and advanced Natural Language Processing (NLP) technologies. It excels in conducting semantic evaluations to uncover similarities, differences, and nuanced relationships within textual data. Whether analyzing documents for investigative journalism, content comparison, or extracting key insights, SENTINEL delivers precise and actionable results."""
ABOUT_SENTINEL_2 = """Ideal for newsrooms and investigative journalists, SENTINEL enhances research capabilities by swiftly identifying patterns, sentiments, and critical information buried within extensive text corpora. Its intelligent learning system continuously refines accuracy, ensuring reliable and efficient analysis across diverse document types and sources. SENTINEL empowers users to uncover hidden connections and trends, making it an indispensable tool for driving informed decisions and impactful storytelling."""
K = 10
DEFAULT_NUM_INPUT = 10
MAX_DOCS = 5
EXP_MAX_DOCS = 30
MIN_RESULTS_PER_DOC = 5
MAX_RESULTS_PER_DOC = 50
MIN_NUM_KWORDS = 5
MAX_NUM_KWORDS = 20
COLLECTION_NAME = "intellinews"
APP_DOCS = os.path.join(os.getcwd(), f'{PROJ_DIR}data/documents')
DF_CSV = os.path.join(os.getcwd(), f'{PROJ_DIR}data/intellinews.csv')
DB_PATH = os.path.join(os.getcwd(), f'{PROJ_DIR}data/intellinews.db')
FEEDBACK_FILE = os.path.join(os.getcwd(), f'{PROJ_DIR}data/feedback.csv')
FEEDBACK_FACES = {
    "😀": ":smiley:", "🙂": ":sweat_smile:", "😐": ":neutral_face:", "🙁": ":worried:", "😞": ":disappointed:"
}

COLOR_BLUE = "#0c2e86"
COLOR_RED = "#a73c07"
COLOR_YELLOW = "#ffcd34"
COLOR_GRAY = '#f8f8f8'

SCROLL_BACK_TO_TOP_BTN = f"""
<style>
    .scroll-btn {{
        position: absolute;
        border: 2px solid #31333f;
        background: #31333f;
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
<a href="#start">
    <br />
    <button class='scroll-btn'>
        New Query
    </button>
</a>
"""

#######################################################################################################

def get_openai_client():
    client = OpenAI(api_key=OPENAI_APIKEY)
    return client
#######################################################################################################

def init_chroma_db(collection_name, db_path=DB_PATH):
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

def semantic_search_separated_documents(Q, k=K, collection=None, titles=[]):
    results = {
        'ids': [[]],
        'distances': [[]],
        'metadatas': [[]],
        'documents': [[]]
    }
    for title in titles:
        title_results = collection.query(
            query_texts=[Q],
            n_results=k, 
            where={'title': title}
        )
        for key in results:
            if key in title_results and isinstance(results[key], list) and isinstance(title_results[key], list):
                results[key][0].extend(title_results[key][0])
    return results
#######################################################################################################

def semantic_search_expanded(Q, expanded_queries, k=K, collection=None, titles=[], separate_documents=False, llm=None):
    expanded_queries.append(Q)
    
    results = {
        'ids': [[]],
        'distances': [[]],
        'metadatas': [[]],
        'documents': [[]]
    }
        
    for query in expanded_queries:
        if separate_documents:
            partial_results = semantic_search_separated_documents(query, k, collection, titles)
        else:
            partial_results = semantic_search(query, k, collection, titles)
            
        for key in results:
            if key in partial_results and isinstance(results[key], list) and isinstance(partial_results[key], list):
                results[key][0].extend(partial_results[key][0])
                    
    # Remove duplicates from documents and corresponding metadata, distances, and ids
    seen_documents = set()
    unique_results = {
        'ids': [[]],
        'distances': [[]],
        'metadatas': [[]],
        'documents': [[]]
    }
    
    for i, doc in enumerate(results['documents'][0]):
        if doc not in seen_documents:
            seen_documents.add(doc)
            unique_results['documents'][0].append(doc)
            unique_results['ids'][0].append(results['ids'][0][i])
            unique_results['distances'][0].append(results['distances'][0][i])
            unique_results['metadatas'][0].append(results['metadatas'][0][i])
    return unique_results
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

def expand_query(Q, nr_queries=4, llm=None, temperature=1):
    task = 'Query Expansion'
    prompt = f"""You are an AI language model assistant. Your task is to generate different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Handle complex queries by splitting them into simpler, more focused sub-queries.

    Instructions:

    - Analyze this query delimited by backticks: ```{Q}```. Identify whether the query contains multiple components or aspects that can be logically separated.
    
    - Split Complex Queries: if the query is complex, break it down into distinct sub-queries. Each sub-query should focus on a specific aspect of the original query.
    
    - Perform Query Expansion: For each sub-query, generate {nr_queries} different expanded versions. These expanded versions should rephrase the sub-query using synonyms or alternative wording.
    
    - Output Format: Provide the expanded queries in a list format, with each query on a new line. Don't write any extra text except the queries, don't number the queries and don't divide the queries by empty lines.
    
    Answer in Finnish."""
    response = generate_response(task, prompt, llm, temperature)
    expanded_queries = response.split('\n')  # Assuming each variation is on a new line
    expanded_queries = [query.strip() for query in expanded_queries]
    filtered_queries = [s for s in expanded_queries if s and len(s) <= 500]
    print(filtered_queries)
    return filtered_queries

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

def generate_focused_summarization(Q, doc, newsroom, llm):
    task = 'Text Summarization'
    if newsroom == 'Helsingin Sanomat':
        language_prompt = "Answer in Finnish."
    else:
        language_prompt = "Answer in English."
    prompt = f"""Make a summary of the document below with a focus on answering the following research question delimited by double quotes: "{Q}". 
    Please extract and condense the key points and findings relevant to this question, highlighting any important data, conclusions, or implications. 
    Justify your insights with evidence from the documents. Format your references as follows:
    - Source: [Document Title]
    - Excerpt: [Approximately 100 words from the document that supports your claim]
    {language_prompt}
    Here is the document to analyse delimited by three backticks:
    ```{doc}```"""
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

def generate_document_analysis(Q, df, llm, advanced_prompt):
    task = 'Document analysis and comparison'
    titles = df['title'].to_list()
    documents = df['documents'].to_list()
    doc_input = ''
    for i in range(len(df)):
        doc_input += f"""
        Document {i} Title: {titles[i]}
        Document {i} Content: {documents[i]}
        """
    prompt = advanced_prompt.replace('{{QUESTION}}', Q).replace('{{DOCUMENTS}}', doc_input).replace('{{TITLE_0}}', titles[0]).replace('{{TITLE_1}}', titles[1])
    response = generate_response(task, prompt, llm)
    return response
#######################################################################################################

def generate_document_analysis_hs(Q, titles, texts, llm, advanced_prompt): # HS_ANALYSIS
    task = 'Document analysis and comparison'
    doc_input = 'Each document has a title and content and is delimited by triple backticks.'
    for i in range(len(titles)):
        doc_input += f"""
        ```Document title: {titles[i]} Content: {texts[i]}```
        """

    prompt = advanced_prompt.replace('{{QUESTION}}', Q).replace('{{DOCUMENTS}}', doc_input)
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

def ask_query(Q, titles, llm, collection, k=15):
    """Function to go from question to query to proper answer"""
    # Get related documents
    results = semantic_search(Q, k=K, collection=collection, titles=titles)

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

# @st.cache_data()
def init_data():
    df = pd.DataFrame(columns=['url', 'title', 'speech'])
    try:
        df = pd.read_csv(DF_CSV)
    except:
        pass
    return df
#######################################################################################################

def save_uploadedfile(uploadedfile):
    if not os.path.exists(APP_DOCS):
        os.makedirs(APP_DOCS)
    file_path = uploadedfile.name

    with open(os.path.join(APP_DOCS, file_path), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path
#######################################################################################################

def submit_feedback(feedback, *args, **kwargs):
    score = feedback['score']
    reax = FEEDBACK_FACES.get(score)
    newsroom = kwargs["newsroom"]
    fb_type = kwargs['type']
    question = kwargs['question']
    prompt = kwargs['prompt']
    llm_response = kwargs['llm_response']
    documents = kwargs['documents']
    feedback_sent = kwargs["feedback_time"]
    comment = feedback['text']

    fbk_from = kwargs['fbk_from']
    if fbk_from == 'rag_chatbot_fbk':
        st.session_state['ragchat_feedback_sent'] = True
    elif fbk_from == 'document_analyzer_fbk':
        st.session_state['docanalyzer_feedback_sent'] = True
    try:
        fb_data = {
            'timestamp': feedback_sent,
            'type': fb_type,
            'newsroom': newsroom,
            'documents': documents,
            'question': question,
            'prompt': prompt,
            'response': llm_response,
            'reaction': reax,
            'score': score,
            'comment': comment
        }
        new_df = pd.DataFrame([fb_data])
        if os.path.exists(FEEDBACK_FILE):
            fb_df = pd.read_csv(FEEDBACK_FILE)
            df = pd.concat([fb_df, new_df], axis=0).reset_index(drop=True)
            df.to_csv(FEEDBACK_FILE, index=False)
        else:
            new_df.to_csv(FEEDBACK_FILE, index=False)

    except Exception as ex:
        st.error(ex)
#######################################################################################################

def display_feedback():
    if os.path.exists(FEEDBACK_FILE):
        st.dataframe(pd.read_csv(FEEDBACK_FILE), height=750, width=1400)
    else:
        st.error('No feedback submitted yet.')
#######################################################################################################

def scroll_to_top():
    js = '''
    <script>
        var body = window.parent.document.querySelector("#start");
        console.log(body);
        body.scrollTop = 0;
    </script>
    '''
    st.html(js)

#######################################################################################################

def reset_document_analyzer():
    for key in st.session_state.keys():
        if 'newsroom' != key:
            del st.session_state[key]
#######################################################################################################

def reset_ragchatbot():
    reset_document_analyzer()
#######################################################################################################

def blank(lines=1):
    for _ in range(lines):
        st.markdown('<div></div>', unsafe_allow_html=True)
#######################################################################################################

@st.dialog("What newsroom are you affiliated with?")
def set_newsroom():
    newsrooms = ['Helsingin Sanomat', 'GMA Network']
    newsroom = st.radio("Select Newsroom:", newsrooms)
    # name = st.text_input("Enter your name:")
    # position = st.text_input("Enter your position:")
    st.container(height=10, border=0)
    if st.button("Select"):
        st.session_state.newsroom = {"newsroom": newsroom} #, "name": name, "position": position}
        st.rerun()
#######################################################################################################

def set_docanalyzer_feedback(Q, prompt, document_analysis, QDOCS, newsroom):
    if 'docanalyzer_feedback_sent' not in st.session_state:
        st.session_state.docanalyzer_feedback_sent = False
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fbk_from = 'document_analyzer_fbk'
    if streamlit_feedback(
        feedback_type="faces",
        optional_text_label="Please provide some more information here...",
        align='flex-start',
        kwargs={"fbk_from":fbk_from, "type": "document_analyzer", "newsroom": newsroom, "question": Q, "prompt": prompt, "llm_response": document_analysis, "documents": ', '.join(QDOCS), "feedback_time": current_date},
        on_submit=submit_feedback
    ):
        st.session_state.docanalyzer_feedback_sent = True
#######################################################################################################

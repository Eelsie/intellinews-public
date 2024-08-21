__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from streamlit_feedback import streamlit_feedback
import spacy_streamlit

from src.utils import save_uploadedfile, _submit_feedback, display_feedback, plot_wordcloud, get_openai_client, init_chroma_db, semantic_search, generate_summarization, generate_translation, generate_sentiment_analysis, generate_topic_labels, generate_document_analysis, ask_query, init_data, upsert_documents_to_collection
from src.utils import APP_DOCS, APP_NAME, APP_DESC, ABOUT_SENTINEL_1, ABOUT_SENTINEL_2, SUPPORTED_LANGUAGES_T, ENTITY_LABELS, CANDIDATE_LABELS, K, scroll_back_to_top_btn, SPACY_MODEL, DF_CSV, COLLECTION_NAME, DB_PATH
from src.prompts import DOC_ANALYSIS_BASE_PROMPT, HS_PROMPT


# Create a Streamlit app
st.set_page_config(layout="wide", page_icon='📰', page_title=APP_NAME)
st.title(APP_NAME)
st.write(APP_DESC)

# Initialize chroma db
collection = init_chroma_db(collection_name=COLLECTION_NAME, db_path=DB_PATH)

# Initialize OpenAI client
llm = get_openai_client()

# Load the dataset
df = init_data()


options = st.sidebar.radio("", ["🏠 Home", "📊 The Dataset", "📚 Document Analysis", "🎫 Feedback Page"])
if options == "🏠 Home":
    st.write('___')
    st.subheader("About")
    st.write(ABOUT_SENTINEL_1)
    st.write(ABOUT_SENTINEL_2)
    st.write('___')
    st.caption("For more information, visit our GitHub repository: [SENTINEL](https://github.com/journalismAI-intellinews/intellinews)")

if options == "📊 The Dataset":
    st.write('___')
    st.write("##### Upload a document to add to the dataset:")
    pdfs = st.file_uploader('Upload PDF files', accept_multiple_files=True, type=['pdf'], label_visibility='hidden')
    btn_upload = st.button("Upload")
    if pdfs and btn_upload:
        for pdf in pdfs:
            file_path = save_uploadedfile(pdf)
            st.toast(f"File uploaded successfully: {file_path}")
            loader = PyPDFLoader(f'{APP_DOCS}/{file_path}')
            docs = loader.load_and_split()

            metadata = {'url':f'file://{file_path}', 'title':file_path}

            # for dataframe
            doc_input = ''
            for i_tmp, doc in enumerate(docs):
                doc_input += str(doc.page_content)
                doc.metadata = metadata
            upsert_documents_to_collection(collection, docs) # NOTE: Run Once to insert the documents to the vector database
            new_df = pd.DataFrame([{'url': f'file://{file_path}', 'title': file_path, 'speech': doc_input}])
            df = init_data()
            df = pd.concat([df, new_df], axis=0).reset_index(drop=True)
            df.to_csv(DF_CSV, index=False)

    st.write("___")
    df = init_data()
    c1, c2 = st.columns([2, 1])
    c2.subheader("Word Count:")
    c2.write(f"{df['speech'].apply(lambda x: len(x.split())).sum(): ,}")
    c1.subheader("The Dataset:")
    c1.write(f"The dataset contains {len(df)} documents.")
    display_df = df.rename(columns={'speech': 'content'}).copy()
    st.dataframe(display_df, height=750, width=1400)

if options == "🎫 Feedback Page":
    display_feedback()

if options == "📚 Document Analysis":
    st.sidebar.container(height=100, border=0)
    st.sidebar.write('___')
    chk_HS_prompt = st.sidebar.checkbox('Use HS prompt', value=False, key='chk_HS_prompt')
    chk_advanced_prompt = st.sidebar.checkbox('Show Advanced Prompt Engineering Option for "Document Analyzer"', value=False, key='chk_advanced_prompt')
    base_prompt = HS_PROMPT if chk_HS_prompt else DOC_ANALYSIS_BASE_PROMPT
    advanced_prompt = ''
    if chk_advanced_prompt:
        st.sidebar.warning('**Note:** _{{QUESTION}}_, _{{DOCUMENTS}}_, _{{TITLE_0}}_, and _{{TITLE_1}}_ are placeholders for the actual values.')
        advanced_prompt = st.sidebar.text_area("Prompt:", placeholder="Type your prompt here...", value=base_prompt.strip(), height=800, max_chars=5000)
    else:
        advanced_prompt = base_prompt

    if len(df) < 2:
        st.error('Please upload at least two documents in the "📊 The Dataset" page to start the comparison analysis.')
    else:
        tab1, tab2 = st.tabs(["Document Analyser", "RAG ChatBot"])

        with tab1:
            with st.form(key='query_form'):
                Q = ''
                QA = ''
                QT = []
                QDOCS = []
                cf1, _, cf2 = st.columns([11, 1, 4])
                with cf1:
                    QA = st.text_area("Ask a Question:", placeholder="Type your question here...", height=100, max_chars=5000)
                    _, center, _ = st.columns([5, 1, 5])
                    center.subheader('OR')
                    QT = st.multiselect("Select a Topic:", CANDIDATE_LABELS)
                    st.markdown('<div></div>', unsafe_allow_html=True)
                    _, center, _ = st.columns(3)
                    center.subheader('FROM DOCUMENTS')
                    QDOCS = st.multiselect("Select Documents:", df['title'].unique(), max_selections=30)
                with cf2:
                    st.markdown('<div></div>', unsafe_allow_html=True)
                    st.write("###### Output Options:")
                    st.markdown('<div></div>', unsafe_allow_html=True)
                    st.write("Select Translation Language(s):")
                    for lang in SUPPORTED_LANGUAGES_T:
                        st.checkbox(lang, value=False, key=f'chk_{lang.lower()}')
                    st.markdown('<div></div>', unsafe_allow_html=True)
                    st.markdown('<div></div>', unsafe_allow_html=True)
                    st.markdown('<div></div>', unsafe_allow_html=True)
                    K = st.number_input('Number of Results(k) per Document:', min_value=5, max_value=50, value=K, step=5)
                    st.markdown('<div></div>', unsafe_allow_html=True)
                    with st.expander("Advanced Options:", expanded=False):
                        st.checkbox('Show Wordclouds', value=False, key='chk_wordcloud')
                        st.checkbox('Show Sentiment Analysis', value=False, key='chk_sentiment')
                        st.checkbox('Extract Keywords', value=False, key='chk_keywords')
                        _, c_extract = st.columns([1, 15])
                        c_extract.number_input('Top Keywords:', min_value=5, max_value=20, value=10, step=5, key='top_keywords')
                        st.checkbox('Show Name Entity Recognition (NER)', value=False, key='chk_ner')

                if len(QT) > 0:
                    Q = ', '.join(QT)
                    QA = ''
                else:
                    Q = QA

                cf1.markdown('___', unsafe_allow_html=True)
                btn_ask = cf1.form_submit_button("Analyze Documents")

            if btn_ask and Q.strip() != '':
                if len(QDOCS) <= 1:
                    st.error("Please select at least two documents for comparison.")
                else:
                    # Semantic Search Results
                    results = semantic_search(Q, k=K, collection=collection, titles=QDOCS)

                    # Inspect Results
                    data_dict = {
                        'ids': results['ids'][0],
                        'distances': results['distances'][0],
                        'documents': results['documents'][0],
                        'title': [eval(str(m))['title'] for m in results['metadatas'][0]],
                        'url': [eval(str(m))['url'] for m in results['metadatas'][0]],
                        'metadata': results['metadatas'][0]
                    }

                    results_df = pd.DataFrame(data_dict)
                    with st.expander("Semantic Data Analysis:", expanded=True):
                        st.subheader('Query:')
                        st.write(Q)
                        st.subheader(f'Sources({results_df["title"].nunique()}):')
                        st.write('; '.join(results_df['title'].unique()))
                        st.subheader(f'Semantic Search Results Data (k={len(results_df)}):')
                        st.dataframe(results_df)
                        if st.session_state['chk_wordcloud']:
                            st.subheader('Word Clouds:')
                            st.pyplot(plot_wordcloud(results_df, 'documents'))

                    cols = st.columns(results_df['title'].nunique())
                    unique_titles = results_df['title'].unique()
                    texts = []

                    for i in range(len(cols)):
                        with cols[i]:
                            title = unique_titles[i]
                            tmp_df = results_df[results_df['title'] == title]
                            source = ''
                            text = ''
                            
                            for x in range(tmp_df.shape[0]):
                                source = f"Source: {tmp_df['url'].iloc[x]}"
                                text += '... ' + tmp_df['documents'].iloc[x] + '...\n\n'
                            
                            texts.append(text)

                            for lang in SUPPORTED_LANGUAGES_T:
                                if st.session_state[f'chk_{lang.lower()}']:
                                    st.subheader(f'Summary: *({lang})*')
                                    st.write(generate_translation(summary, lang, llm))
                                    st.write('___')

                            if st.session_state['chk_sentiment']:
                                st.subheader('Sentiment Analysis:')
                                st.write(generate_sentiment_analysis(text, llm))
                                st.write('___')

                            if st.session_state['chk_keywords']:
                                st.subheader('Keywords:')
                                top_k = st.session_state['top_keywords'] or 10
                                st.write(generate_topic_labels(text, llm, top_k=top_k))
                                st.write('___')

                            if st.session_state['chk_ner']:
                                st.subheader('Name Entity Recognition *(NER)*:')
                                doc = SPACY_MODEL(text)
                                spacy_streamlit.visualize_ner(
                                    doc,
                                    labels = ENTITY_LABELS,
                                    show_table = False,
                                    title = '',
                                    key=f'ner{i}'
                                )

                    document_analysis = generate_document_analysis(Q, unique_titles, texts, llm, advanced_prompt)
                    st.write('___')
                    st.header('SENTINEL Document Analysis:')
                    # st.markdown(document_analysis.replace('```markdown', '').replace('```', ''))
                    st.markdown(document_analysis)
                    st.write('___')
                    # current_date = str(datetime.utcnow())
                    # st.caption('Was the analysis helpful?')
                    # streamlit_feedback(
                    #     feedback_type="faces",
                    #     key = f'document_analysis_{current_date}',
                    #     optional_text_label="Please provide some more information here...",
                    #     # max_text_length=1500,
                    #     align='flex-start',
                    #     kwargs={"question": Q, "llm_response": document_analysis, "documents": ', '.join(QDOCS), "feedback_time": current_date},
                    #     on_submit=_submit_feedback
                    # )


        with tab2:
            QDOCS = []
            lang = 'English'
            languages = [lang] + SUPPORTED_LANGUAGES_T
            cf1, _, cf2 = st.columns([20, 1, 5])
            with cf1:
                QDOCS = st.multiselect("Select Documents:", df['title'].unique(), max_selections=5)

            with cf2:
                st.markdown('<div></div>', unsafe_allow_html=True)
                st.write("###### Chat Options:")
                st.markdown('<div></div>', unsafe_allow_html=True)
                lang = st.selectbox("Language Option:", languages, index=0)
                st.markdown('<div></div>', unsafe_allow_html=True)
                K = st.number_input('Number of Results(k) per Document:', min_value=5, max_value=50, value=K, step=5)
                st.markdown('<div></div>', unsafe_allow_html=True)

            if len(QDOCS) > 0:
                cf1.caption(f'Sources: {", ".join(QDOCS)}')

            if len(QDOCS) <= 1:
                cf1.markdown("*Note:* Please select at least two documents for comparison so you could start the chat.")

            else:
                if lang != "English":
                    # titles = list(map(lambda x: generate_translation(x, llm, "English", lang), orig_titles))
                    feedback_caption = generate_translation("Was this helpful?", lang, llm)
                    user_prompt = generate_translation("Type your questions here...", lang, llm)
                    load_response = generate_translation("Loading a response...", lang, llm)
                else:
                    # titles = orig_titles
                    feedback_caption = 'Was this helpful?'
                    user_prompt = "Type your questions here..."
                    load_response = "Loading a response..."

                # Initialize title
                if "titles" not in st.session_state:
                    st.session_state['titles'] = None

                if "feedback" not in st.session_state:
                    st.session_state['feedback'] = None

                # Initiliaze spoken
                if "spoken" not in st.session_state:
                    st.session_state['spoken'] = False

                # Initialize total number of responses
                if "total_responses" not in st.session_state:
                    st.session_state['total_responses'] = 0

                with cf1:
                    st.markdown('<div></div>', unsafe_allow_html=True)
                    st.markdown('<div></div>', unsafe_allow_html=True)
                    st.write("###### Chat:")

                    # Initialize chat history or reset history if you change documents
                    if "messages" not in st.session_state or st.session_state['titles'] != QDOCS:
                        try:
                            if len(st.session_state.messages) > 0:
                                st.session_state.total_responses += len(st.session_state.messages)
                        except:
                            pass
                        st.session_state.messages = []
                        st.session_state['titles'] = QDOCS
                        st.session_state['feedback'] = None

                    # Display chat messages from history
                    for i, message in enumerate(st.session_state.messages):
                        with st.chat_message(message['role']):
                            st.markdown(message['content'])
                            if message['role'] == "assistant":
                                titles = st.session_state['titles']
                                documents = ', '.join(titles)
                                question = st.session_state.messages[i-1]['content']
                                current_date = str(datetime.utcnow())
                                idx = st.session_state.total_responses

                                # # Text to Speech the response (if enabled)
                                # if st.session_state['speak'] and (i == len(st.session_state.messages) - 1) and not(st.session_state['spoken']):
                                #     st.session_state['spoken'] = True
                                #     if st.session_state['lang'] == 'English':
                                #         text_to_speech(message['content'], lang='en')
                                #     else:
                                #         # tl = Filipino (the only one available)
                                #         text_to_speech(message['content'], lang='tl')

                                st.caption(feedback_caption)
                                streamlit_feedback(
                                    feedback_type="faces",
                                    key = f'comment_{i+idx}_{len(message)}_{"-".join(titles)}',
                                    optional_text_label="Please provide some more information here...",
                                    # max_text_length=1500,
                                    align='flex-start',
                                    kwargs={"question": question, "llm_response": message['content'], "documents": documents, "feedback_time": current_date},
                                    on_submit=_submit_feedback
                                )

                    # Accept user input
                    if prompt := st.chat_input(user_prompt):
                        # Reset feedback
                        # st.session_state.feedback_update = None
                        st.session_state.feedback = None

                        # Translate the user prompt to english
                        if lang != "English":
                            prompt = generate_translation(prompt, lang, llm)

                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})

                        # Display user message
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        # Display response
                        with st.chat_message("assistant"):
                            with st.spinner(load_response):
                                # Semantic Search Results
                                response = ask_query(prompt, QDOCS, llm, k=15, collection=collection)

                                # Translate the response if the language of choice is not in English
                                if lang != "English":
                                    response = generate_translation(response, lang, llm)

                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.markdown(response)


                            # # Text to Speech the response (if enabled)
                            # if speak:
                            #     if lang == 'English':
                            #         text_to_speech(response, lang='en')
                            #     else:
                            #         # tl = Filipino (the only one available)
                            #         text_to_speech(response, lang='tl')

                        # Add a scroll back to top button
                        st.markdown(scroll_back_to_top_btn, unsafe_allow_html=True)
                        st.rerun()


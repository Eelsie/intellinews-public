from src.utils import *
#######################################################################################################
#######################################################################################################

# Create a Streamlit app
st.set_page_config(layout="wide", page_icon='📰', page_title=APP_NAME)
st.markdown('<div id="start"></div>', unsafe_allow_html=True)
st.title(APP_NAME)
st.write(APP_DESC)

# Initialize newsroom selection
if "newsroom" not in st.session_state:
    set_newsroom()

# Initialize chroma db
collection = init_chroma_db(collection_name=COLLECTION_NAME, db_path=DB_PATH)

# Initialize OpenAI client
llm = get_openai_client()

# Load the dataset
df = init_data()

if "newsroom" in st.session_state:
    newsroom = st.session_state.newsroom["newsroom"]
    st.sidebar.markdown(f"Newsroom:  &nbsp; **{newsroom}**")
    st.sidebar.button(":newspaper: &nbsp; Change", on_click=set_newsroom)
    st.sidebar.write('___')

st.sidebar.container(height=20, border=0)
options = st.sidebar.radio("Menu", ["🏠 Home", "📊 The Dataset", "📚 Document Analysis", "🎫 Feedback Page"], label_visibility='hidden')
st.sidebar.container(height=40, border=0)
st.sidebar.write('___')


if options == "🏠 Home":
    st.write('___')
    st.subheader("About")
    st.write(ABOUT_SENTINEL_1)
    st.write(ABOUT_SENTINEL_2)
    st.write('___')
    st.caption("For more information, visit our GitHub repository: [SENTINEL](https://github.com/journalismAI-intellinews/intellinews)")
    reset_document_analyzer()

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
    reset_document_analyzer()

if options == "🎫 Feedback Page":
    display_feedback()
    reset_document_analyzer()

if options == "📚 Document Analysis":
    newsroom = ''
    if 'newsroom' in st.session_state:
        newsroom = st.session_state.newsroom["newsroom"]
    st.sidebar.container(height=5, border=0)
    st.sidebar.caption(':gear: &nbsp;&nbsp; Experimental Options for "Document Analyzer":')
    num_experimental_max_docs = st.sidebar.number_input('Maximum Number of Selected Documents', min_value=MAX_DOCS, max_value=EXP_MAX_DOCS, step=1, key='num_experimental_max_docs')

    if num_experimental_max_docs > MAX_DOCS:
        st.sidebar.warning(f'**Note:** Output options such as _Text Summarization with Translation_, _Sentiment Classification_, _Keyword Extraction_, and _Name Entity Recognition (NER)_  is currently not supported if the number documents to be selected is more than {MAX_DOCS}.')

    st.sidebar.container(height=5, border=0)
    chk_advanced_prompt = st.sidebar.checkbox('Show Advanced Prompt Engineering', value=False)
    prompt_template = ''
    if newsroom == NEWSROOM_HS:
        if num_experimental_max_docs > MAX_DOCS:
            prompt_template = HS_PROMPT_MANY_DOCUMENTS
        else:
            prompt_template = HS_PROMPT
    else:
        prompt_template = DOC_ANALYSIS_BASE_PROMPT

    advanced_prompt = ''
    if chk_advanced_prompt:
        if newsroom == NEWSROOM_HS:
            st.sidebar.warning('**Note:** _{{QUESTION}}_, and _{{DOCUMENTS}}_ are placeholders for the actual values.')
        else:
            st.sidebar.warning('**Note:** _{{QUESTION}}_, _{{DOCUMENTS}}_, _{{TITLE_0}}_, and _{{TITLE_1}}_ are placeholders for the actual values.')
        advanced_prompt = st.sidebar.text_area("Prompt Template:", placeholder="Type your prompt here...", value=prompt_template.strip(), height=800, max_chars=5000)
    else:
        advanced_prompt = prompt_template

    if len(df) < 2:
        st.error('Please upload at least two documents in the "📊 The Dataset" page to start the comparison analysis.')
    else:
        tab1, tab2 = st.tabs(["Document Analyser", "RAG ChatBot"])

        # Document Analyzer
        with tab1:
            with st.form(key='query_form'):
                Q = ''
                QA = ''
                QT = []
                QDOCS = []
                qa_val = ''
                mselect_qt = []
                mselect_qdocs = []
                if 'key_txt_qa' in st.session_state:
                    qa_val = st.session_state['key_txt_qa']
                if 'key_mselect_qt' in st.session_state:
                    mselect_qt = st.session_state['key_mselect_qt']
                if 'key_mselect_qdocs' in st.session_state:
                    mselect_qdocs = st.session_state['key_mselect_qdocs']

                cf1, _, cf2 = st.columns([11, 1, 4])
                with cf1:
                    QA = st.text_area("Ask a Question:", placeholder="Type your question here...", height=100, max_chars=5000, value=qa_val)
                    st.session_state['key_txt_qa'] = QA
                    _, center, _ = st.columns([5, 1, 5])
                    center.subheader('OR')
                    QT = st.multiselect("Select a Topic:", CANDIDATE_LABELS, default=mselect_qt)
                    st.session_state['key_mselect_qt'] = QT
                    blank()
                    _, center, _ = st.columns(3)
                    center.subheader('FROM DOCUMENTS')
                    QDOCS = st.multiselect("Select Documents:", df['title'].unique(), max_selections=num_experimental_max_docs, default=mselect_qdocs)
                    st.session_state['key_mselect_qdocs'] = QDOCS
                with cf2:
                    blank()
                    st.write("###### Output Options:")
                    blank()
                    if num_experimental_max_docs <= MAX_DOCS:
                        st.caption("Text Summarization:")
                        st.checkbox('Show Summary', value=False, key=f'chk_show_summary')
                        blank()
                    # st.write("Select Translation Language(s) you want to translate the summary to:")
                    # for lang in SUPPORTED_LANGUAGES_T:
                    #     st.checkbox(lang, value=False, key=f'chk_{lang.lower()}')
                    SUPPORTED_LANGUAGES_T.insert(0, 'No translation')
                    doc_lang = st.selectbox("Select Translation Language Option:", SUPPORTED_LANGUAGES_T, index=(0 if newsroom == NEWSROOM_HS else 1))
                    blank()
                    K = st.number_input('Number of Results(k) per Document:', min_value=MIN_RESULTS_PER_DOC, max_value=MAX_RESULTS_PER_DOC, value=K, step=5, key='number_input_k')
                    st.checkbox('Extract results(k) separately for each document', value=False, key='separate_documents_in_semantic_search')
                    st.checkbox('Expand queries', value=False, key='expand_queries')

                    if num_experimental_max_docs <= MAX_DOCS:
                        blank()
                        with st.expander("Advanced Options:", expanded=False):
                            st.checkbox('Show Wordclouds', value=False, key='chk_wordcloud')
                            st.checkbox('Show Sentiment Analysis', value=False, key='chk_sentiment')
                            st.checkbox('Extract Keywords', value=False, key='chk_keywords')
                            _, c_extract = st.columns([1, 15])
                            c_extract.number_input('Top Keywords:', min_value=MIN_NUM_KWORDS, max_value=MAX_NUM_KWORDS, value=10, step=5, key='top_keywords')
                            st.checkbox('Show Name Entity Recognition (NER)', value=False, key='chk_ner')

                if len(QT) > 0:
                    Q = ', '.join(QT)
                    QA = ''
                else:
                    Q = QA

                cf1.markdown('___', unsafe_allow_html=True)
                if 'doc_analyzer_docanalysis' in st.session_state:
                    btn_ask = cf1.form_submit_button("Analyze Documents", disabled=True)
                else:
                    btn_ask = cf1.form_submit_button("Analyze Documents")

            if btn_ask and Q.strip() != '':
                if len(QDOCS) <= 1:
                    st.error("Please select at least two documents for comparison.")
                else:
                    if 'doc_analyzer_query' not in st.session_state:
                        st.session_state['doc_analyzer_query'] = Q

                    # Semantic Search Results
                    if 'expand_queries' in st.session_state and st.session_state['expand_queries']:
                        expanded_queries = expand_query(Q, newsroom, doc_lang, 4, llm, 1)
                        st.session_state['expanded_queries'] = expanded_queries
                        if 'separate_documents_in_semantic_search' in st.session_state and st.session_state['separate_documents_in_semantic_search']:
                            results = semantic_search_expanded(Q, expanded_queries, k=K, collection=collection, titles=QDOCS, separate_documents=True)
                        else:
                            results = semantic_search_expanded(Q, expanded_queries, k=K, collection=collection, titles=QDOCS, separate_documents=False)
                    elif 'separate_documents_in_semantic_search' in st.session_state and st.session_state['separate_documents_in_semantic_search']:
                        results = semantic_search_separated_documents(Q, k=K, collection=collection, titles=QDOCS)
                    else:
                        results = semantic_search(Q, k=K, collection=collection, titles=QDOCS)
                    if 'doc_analyzer_result' not in st.session_state:
                        st.session_state['doc_analyzer_result'] = results

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
                    cols = st.columns(results_df['title'].nunique())
                    unique_titles = results_df['title'].unique()

                    texts = [] # for HS analysis using joined chunks
                    for i in range(len(cols)):
                        with cols[i]:
                            title = unique_titles[i]
                            tmp_df = results_df[results_df['title'] == title]
                            source = ''
                            text = ''

                            for x in range(tmp_df.shape[0]):
                                source = f"Source: {tmp_df['url'].iloc[x]}"
                                text += '... ' + tmp_df['documents'].iloc[x] + '...\n\n'

                            texts.append(text) # for HS analysis using joined chunks

                            if 'chk_show_summary' in st.session_state and st.session_state['chk_show_summary']:
                                summary = ''
                                for il, lang in enumerate(SUPPORTED_LANGUAGES_T):
                                    if doc_lang == lang: #st.session_state[f'chk_{lang.lower()}']:
                                        if summary == '':
                                            summary = generate_summarization(text, llm)
                                            if 'doc_analyzer_summary' not in st.session_state:
                                                st.session_state[f'doc_analyzer_col{i}_summary'] = summary
                                        if doc_lang != 'No translation':
                                            translation = generate_translation(summary, lang, llm)
                                        if f'doc_analyzer_{lang}_translation' not in st.session_state:
                                            st.session_state[f'doc_analyzer_col{i}_{lang}_translation'] = translation
                                        break

                            if 'chk_sentiment' in st.session_state and st.session_state['chk_sentiment']:
                                sentiment_analysis = generate_sentiment_analysis(text, llm)
                                if 'doc_analyzer_sentiment_analysis' not in st.session_state:
                                    st.session_state[f'doc_analyzer_col{i}_sentiment_analysis'] = sentiment_analysis

                            if 'chk_keywords' in st.session_state and st.session_state['chk_keywords']:
                                top_k = st.session_state['top_keywords'] or 10
                                topic_labels = generate_topic_labels(text, llm, top_k=top_k)
                                if 'doc_analyzer_topic_labels' not in st.session_state:
                                    st.session_state[f'doc_analyzer_col{i}_topic_labels'] = topic_labels

                    document_analysis = ''
                    if newsroom  == NEWSROOM_HS:
                        if 'num_experimental_max_docs' in st.session_state and st.session_state['num_experimental_max_docs'] > MAX_DOCS:
                            summaries = []
                            for text in texts:
                                summaries.append(generate_focused_summarization(Q, text, llm, newsroom, doc_lang))
                            texts = summaries
                        document_analysis = generate_document_analysis_hs(Q, unique_titles, texts, llm, advanced_prompt, doc_lang)
                    else:
                        document_analysis = generate_document_analysis(Q, results_df, llm, advanced_prompt)

                    # Translate the document analysis using the current user session's language of choice
                    if doc_lang != 'No translation':
                        document_analysis = generate_translation(document_analysis, doc_lang, llm)

                    if 'doc_analyzer_docanalysis' not in st.session_state:
                        st.session_state['doc_analyzer_docanalysis'] = document_analysis

                    st.rerun()

            # Document Analyzer's feedback to be submitted and states has been updated
            if 'doc_analyzer_query' in st.session_state:
                Q = st.session_state['doc_analyzer_query']

            # Semantic Search Results
            if 'doc_analyzer_result' in st.session_state:
                results = st.session_state['doc_analyzer_result']

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
                    if 'expanded_queries' in st.session_state:
                        st.subheader('Expanded queries:')
                        queries_str = ''
                        for query in st.session_state['expanded_queries'][:-1]:
                            queries_str += f"{query}<br>"
                        st.write(f"""
                        {queries_str}
                        """, unsafe_allow_html=True)
                    st.subheader(f'Sources({results_df["title"].nunique()}):')
                    st.write('; '.join(results_df['title'].unique()))
                    st.subheader(f'Semantic Search Results Data (k={len(results_df)}):')
                    title_counts = results_df['title'].value_counts()
                    title_count_str = ''
                    for title, count in title_counts.items():
                        title_count_str += f"{title} (k={count})<br>"

                    st.write(f"""
                    {title_count_str}
                    """, unsafe_allow_html=True)
                    st.dataframe(results_df)
                    if 'chk_wordcloud' in st.session_state and st.session_state['chk_wordcloud']:
                        st.subheader('Word Clouds:')
                        st.pyplot(plot_wordcloud(results_df, 'documents'))

                cols = st.columns(results_df['title'].nunique())
                unique_titles = results_df['title'].unique()

                for i in range(len(cols)):
                    with cols[i]:
                        title = unique_titles[i]
                        tmp_df = results_df[results_df['title'] == title]
                        source = ''
                        text = ''

                        for x in range(tmp_df.shape[0]):
                            source = f"Source: {tmp_df['url'].iloc[x]}"
                            text += '... ' + tmp_df['documents'].iloc[x] + '...\n\n'

                        if newsroom != NEWSROOM_HS:
                            st.header(title)
                            st.write(f"Document Result Index: {i}")
                            st.caption(f"Source: {results_df['url'].iloc[i]}")

                        summary = ''
                        for il, lang in enumerate(SUPPORTED_LANGUAGES_T):
                            if doc_lang == lang: #st.session_state[f'chk_{lang.lower()}']:
                                if f'doc_analyzer_col{i}_{lang}_translation' in st.session_state:
                                    st.write('___')
                                    st.subheader(f'Summary: *({lang})*')
                                    translation = st.session_state[f'doc_analyzer_col{i}_{lang}_translation']
                                    st.write(translation)
                                break

                        if 'chk_sentiment' in st.session_state and st.session_state['chk_sentiment']:
                            st.subheader('Sentiment Analysis:')
                            if f'doc_analyzer_col{i}_sentiment_analysis' in st.session_state:
                                sentiment_analysis = st.session_state[f'doc_analyzer_col{i}_sentiment_analysis']
                                st.write(sentiment_analysis)
                                st.write('___')

                        if 'chk_keywords' in st.session_state and st.session_state['chk_keywords']:
                            st.subheader('Keywords:')
                            top_k = st.session_state['top_keywords'] or 10
                            if f'doc_analyzer_col{i}_topic_labels' in st.session_state:
                                topic_labels = st.session_state[f'doc_analyzer_col{i}_topic_labels']
                                st.write(topic_labels)
                                st.write('___')

                        if 'chk_ner' in st.session_state and st.session_state['chk_ner']:
                            st.subheader('Name Entity Recognition *(NER)*:')
                            doc = SPACY_MODEL(text)
                            spacy_streamlit.visualize_ner(
                                doc,
                                labels = ENTITY_LABELS,
                                show_table = False,
                                title = '',
                                key=f'ner{i}'
                            )

                if 'doc_analyzer_docanalysis' in st.session_state:
                    document_analysis = st.session_state['doc_analyzer_docanalysis']
                    st.write('___')
                    st.header('SENTINEL Document Analysis:')
                    st.markdown(document_analysis.replace('```markdown', '').replace('```', ''))

                    blank(3)
                    st.caption("Was this helpful?")
                    set_docanalyzer_feedback(newsroom=newsroom, Q=Q, prompt=advanced_prompt, document_analysis=document_analysis, QDOCS=QDOCS)
                    st.write('___')

                    blank(2)
                    _, c_reset, _ = st.columns([3, 3, 2])
                    c_reset.button("Reset Document Analyzer", on_click=reset_document_analyzer)
                    if "docanalyzer_feedback_sent" in st.session_state and st.session_state.docanalyzer_feedback_sent:
                        st.toast('Document Analyzer feedback submitted successfully!', icon='🎉')
                        st.session_state.docanalyzer_feedback_sent = False

        # RAG ChatBot
        with tab2:
            QDOCS = []
            cf1, _, cf2 = st.columns([20, 1, 5])
            with cf1:
                QDOCS = st.multiselect("Select Documents:", df['title'].unique(), max_selections=num_experimental_max_docs)

            with cf2:
                blank()
                st.write("###### Chat Options:")
                blank()
                lang = st.selectbox("Language Option:", SUPPORTED_LANGUAGES_T, index=0)

                blank()
                K = st.number_input('Number of Results(k) per Document:', min_value=MIN_RESULTS_PER_DOC, max_value=MAX_RESULTS_PER_DOC, value=DEFAULT_NUM_INPUT, step=5)
                blank()

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
                    blank(2)
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
                                current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                idx = st.session_state.total_responses
                                newsroom = st.session_state.newsroom["newsroom"]

                                st.caption(feedback_caption)
                                fbk_from = 'rag_chatbot_fbk'
                                streamlit_feedback(
                                    feedback_type="faces",
                                    key = f'comment_{i+idx}_{len(message)}_{"-".join(titles)}',
                                    optional_text_label="Please provide some more information here...",
                                    # max_text_length=1500,
                                    align='flex-start',
                                    kwargs={"fbk_from":fbk_from, "type": "rag_chatbot", "newsroom": newsroom, "question": question, "prompt": f"[same as the chat question]\n\n{question}", "llm_response": message['content'], "documents": documents, "feedback_time": current_date},
                                    on_submit=submit_feedback
                                )


                    # Accept user input
                    if prompt := st.chat_input(user_prompt):

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
                                response = ask_query(prompt, QDOCS, llm, collection, k=15)

                                # Translate the response if the language of choice is not in English
                                if lang != "English":
                                    response = generate_translation(response, lang, llm)

                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.markdown(response)


                        st.rerun()
                    blank(2)
                    st.write('___')
                    if len(st.session_state.messages) > 0:
                        _, c_reset_chat, _ = st.columns([3, 3, 2])
                        c_reset_chat.button("Reset Chat History", on_click=reset_ragchatbot)

                    if 'ragchat_feedback_sent' in st.session_state and st.session_state.ragchat_feedback_sent:
                        st.toast('RAG Chat feedback submitted successfully!', icon='🎉')
                        st.session_state.ragchat_feedback_sent = False


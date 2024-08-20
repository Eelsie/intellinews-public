import streamlit as st

import os
from dotenv import load_dotenv

import regex as re #import re
from time import sleep
import textwrap
from annotated_text import annotated_text, parameters
from random import randint

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from langchain_community.document_loaders import PyPDFLoader
import uuid

parameters.LABEL_FONT_SIZE="1.0rem"
parameters.PALETTE = ["#ff4b4b", "#ffa421", "#ffe312", "#21c354", "#00d4b1", "#00c0f2", "#1c83e1", "#803df5",
    "#808495",'#db6131', '#ce3d89', '#bc7911', '#52305d', '#3113cb', '#0e9ece', '#67ae1a', '#6c6820', '#c49c3d', '#8909c0']


APP = './app'
APP_DOCS = f'{APP}/documents'
if not os.path.exists(APP_DOCS):
    os.makedirs(APP_DOCS)

IS_DEBUG = False
RETRY_COUNT = 12

PAGE_PER_REQUEST = 9
TOPIC_COUNT = 23
MIN_SENT_COUNT = 1
MAX_SENT_COUNT = 10

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

load_dotenv()
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
genai.configure(api_key=GEMINI_API_KEY)

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

def debug(response):
      # If the response doesn't contain text, check if the prompt was blocked.
      feedback = response.prompt_feedback
      # Also check the finish reason to see if the response was blocked.
      reason = response.candidates[0].finish_reason
      # If the finish reason was SAFETY, the safety ratings have more details.
      safety_settings = response.candidates[0].safety_ratings
      print(f'Feedback: {feedback}\n\nReason: {reason}\n\nSafety Settings: {safety_settings}')

def to_markdown(text):
    try:
      text = text.replace('•', '  *')
      return textwrap.indent(text, '> ', predicate=lambda _: True)
    except:
      return text

def get_supported_models():
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

def get_gemini_model():
    return genai.GenerativeModel('gemini-1.5-pro-latest')
    # return genai.GenerativeModel('gemini-1.0-pro')

def get_gemini_text_response(prompt):
  try:
    response = get_gemini_model().generate_content(
       prompt, request_options={"timeout": 600}, safety_settings=SAFETY_SETTINGS)
    if IS_DEBUG:
      debug(response)
    if response.text:
      return response.text
  except:
    pass

def get_gemini_vision_model():
    return genai.GenerativeModel('gemini-1.0-pro-vision-latest')

def get_gemini_vision_response(prompt, img):
  try:
    response = get_gemini_vision_model().generate_content(
       [prompt, img], request_options={"timeout": 600}, safety_settings=SAFETY_SETTINGS)
    if IS_DEBUG:
      debug(response)
    if response.text:
      return response.text
  except:
    pass

# get_supported_models()
def get_list_of_random_color(N=20):
    colors = []
    for _ in range(N):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors

def get_doc_query_prompt(doc_query_context, doc_input, country='Finland', other_task=None):
  prompt = f"""
  You are an unbias, fair, honest, intelligent, and an expert jounalist-researcher that is very knowledgeable in different domain of expertise encompassing investigative journalism.
  You are not representing any party or organization and you would treat the documents as research materials for intelligent context searching for you to be able to report the similiraties and differences of what has been written in those documents.
  Your main task is to compare sets of documents that discuss several topics in the country of {country}.

  To accomplish this, you would first list down key points from the given research document: '{doc_query_context}' as these key points will serve as the context of queries that you would search in another research document: {doc_input}.
  If the keypoints are already in a list format, make sure that it will be included in your final list of keypoints.

  Then, for each keypoint item relative to the search result that you have found given the same context, it is important to describe they're differences and similarities in terms of how was it align to its original context. If no similar context found, just note that keypoint was not found in the document but still include the keypoint in the item list.
  Likewise, for each keypoint item, you would include a reference to a phrase where you have found the keypoint.

  More importantly, you to always provide a final summary of the results from your findings where in you would highlight the overall similarities and differences of each keypoint and make a final recommendation or action items as necessary.

  The final output should be in the following format:
    Title: Title
    Keypoints:
      Keypoint 1 Title: Keypoint 1 Title
        Keypoint 1 Context: Context Summary
        Keypoint 1 Similarities: Similarities
        Keypoint 1 Differences: Differences
        Keypoint 1 Reference: Reference phrase
      Keypoint 2 Title: Keypoint 2 Title
        Keypoint 2 Context: Context Summary
        Keypoint 2 Similarities: Similarities
        Keypoint 2 Differences: Differences
        Keypoint 2 Reference: Reference phrase
    Executive Summary: Executive Summary
    Recommendations: Recommendations / Action items
  """
  if other_task:
    prompt += f'And {other_task}.'
  return prompt

def make_doc_query_request(pages_query_context, pages_input, country='Finland', other_task=None):
  doc_query_context = ''
  for page in pages_query_context:
      doc_query_context += str(page.page_content)
  doc_input = ''
  for page in pages_input:
      doc_input += str(page.page_content).replace('\n', '')
  return get_doc_query_prompt(doc_query_context, doc_input, country, other_task)


def get_doc_topic_prompt(topic_count, doc_input):
    # prompt = f"""
    #     You are an unbias, fair, honest, intelligent, and an expert jounalist-researcher that is very knowledgeable in different domain of expertise encompassing investigative journalism.
    #     You are not representing any party or organization and you would treat the documents as research materials for intelligent context searching for you to be able to report the similiraties and differences of what has been written in those documents.
    #     Your main task is to list down topic labels with a maximum of {topic_count} topics using topic modeling techniques that you have learned from Natural Language Processing (NLP) and Machine Learning (ML) to identify the main topics that are being discussed in the documents that you have provided.
    #     List down just the topic labels and no need to provide the context of the topics.
    #     The final output should be in the json list format.
    #     It is important that you would provide at least 1 topic label for each document that you have provided. None or empty json output is bad and is not acceptable.
        
    #     Below is an article:
    #     {doc_input}
    #     Please, identify the main topics mentioned in these document.
    # """

    # prompt = f"""
    #     You are an unbias, fair, honest, intelligent, and an expert jounalist-researcher that is very knowledgeable in different domain of expertise encompassing investigative journalism.
    #     You are not representing any party or organization and you would treat the document as research material for intelligent context searching for you to be able to report the similiraties and differences of what has been written in the document.
    prompt = f"""
        You are a Large Language Model fine-tuned for document retrieval and content analysis. Given any kind of document or set of documents from your research, you would need to understand the document and use it for intelligent context searching for you to be able to make a comprehensive analysis about the similiraties and differences of topics being discussed in the document.
        There are cases where you have to review and analyze document that are very lengthy, multiligual, and complex. Likewise, there might be documents that might not be suitable for community standards or might contain sensitive information but you have to make sure that you disregard those information so would still be able to provide a fair and unbiased machine learning analysis of the document for the output.
        
        For initial preprocessing of text, you would need to remove any non-ascii characters and disregard formatting symbols and make sure that the text is clean and ready for topic modeling. You would also need to have English topic labels for paragraph or group of sentences or at least a sentence that are written in other language than English.

        Your main task is to list down topic labels based on the summary of the document. It very is important that you would provide at least 3 topic labels and a maximum of {topic_count} topic labels using topic modeling techniques in Natural Language Processing (NLP) and Machine Learning (ML) to identify the key points that are being discussed in the document.

        Below is the document that you have to review and analyze:

            {doc_input}
        
        You would need to identify the main topics in english that are being discussed in the document.
        List down just the topic labels and no need to provide the context of the topics.
        The final output should be in the json-formatted list:
        Example:
        [
            "Topic 1",
            "Topic 2",
            "Topic 3"
        ]
    """ 
    return prompt

def get_annots_request(doc_input, s_topics):
    # annot_request = f'''Given this text: {st.session_state['doc_inputs'][s_doc]}
    # Add annotations to paragraphs or at least a group of sentences that discuss about the topic: {selected_topic} by adding the pair of the following tags: [{selected_topic}] and [/{selected_topic}].
    
    # Example:
    #     Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    #     Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    
    # Result:
    #     [{selected_topic}]Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    #     Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.[/{selected_topic}] Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    # '''
    annot_request = f'''Given this text: {doc_input}
    Add annotations to paragraphs or group of sentences or at least a sentence that discuss about a particular topic only from these set of topics: {", ".join(s_topics)} by adding the pair of the following tags: e.g. [{s_topics[0]}] and [/{s_topics[0]}].
    It is important that for each paragraphs or group of sentences or at least a sentence that will be annotated should at least contain {MIN_SENT_COUNT} sentence and at most {MAX_SENT_COUNT} sentences.
    And if there are cases that the paragraph or group of sentences is more than {MAX_SENT_COUNT} sentences, it is important to iteratively split the paragraph or group of sentences or at least a sentence into multiple paragraphs or multiple group of sentences to achieve the required {MAX_SENT_COUNT} length of sentences and then annotate each of them.
    You can also split individual paragraphs or group of sentences by its existing bullet or numbering format.
    
    There are cases where you have to review and analyze document that are very lengthy, multiligual, and complex. Likewise, there might be documents that might not be suitable for community standards or might contain sensitive information but you have to make sure that you disregard those information so would still be able to provide a fair and unbiased machine learning analysis of the document for the output.
    
    For initial preprocessing of text, you would need to remove any non-ascii characters and disregard formatting symbols and make sure that the text is clean and ready for topic modeling. You would also need to have English topic labels for paragraph or group of sentences or at least a sentence that are written in other language than English.
    
    Make sure that you have applied the maximum limit {MAX_SENT_COUNT} of sentences to each paragraph orgroup of sentences or at least a sentence that you will annotate to have a readable and understandable context of the topic.
    In the same manner, make sure that you have applied the text cleaning by removing any non-ascii characters and disregard formatting symbols to make sure that the text is clean and does not contain any unwanted characters that might affect the css styling of the text or markdown format of the text.
    All paragraphs or group of sentences or at least a sentence should be annotated. If there are cases that the paragraph or group of sentences or at least a sentence is not related to any of the topics, just annotate the paragraph or group of sentences or at least a sentence with a pair of [NO_TOPIC] and [/NO_TOPIC].
    
    
    Example:
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    
    Result:
        [{s_topics[1]}]Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.[/{s_topics[1]}] [NO_TOPIC]Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.[/NO_TOPIC] [{s_topics[0]}]Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.[/{s_topics[0]}]
    '''
    return annot_request

def get_key_topics_prompt(topics):
    prompt = f'''For each topic in these topics: {', '.join(topics)} that you have identified, please select the topics that are in the same context and combine them into one topic label.
            If one topic is in different context, just include that topic label to your final list of topics with empty list for its subtopics.
                Example:
                "Economic Growth and Prosperity" and "Economy and Employment" and "Economic Policy" are all related to "Economic Development". Therefore, one topic label should be "Economic Development" and the previous three topics will become  subitems or subtopics in json list format as well.
            Your main task is to provide a final list of topics based on the original topics provided.
            The final output should be in the json list format:
            ''' + '''[
                {"topic": "Topic 1", "subtopics": ["Subtopic 1", "Subtopic 2", "Subtopic 3"]},
                {"topic": "Topic 2", "subtopics": ["Subtopic 1", "Subtopic 2"]},
                {"topic": "Topic 3", "subtopics": ["Subtopic 1", "Subtopic 2", "Subtopic 3"]},
                {"topic": "Topic 4", "subtopics": []},
                {"topic": "Topic 5", "subtopics": ["Subtopic 1", "Subtopic 2", "Subtopic 3", "Subtopic 4"]}
            ]
            '''
    return prompt

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]', ' ', text)

def save_uploadedfile(uploadedfile):
    # file_id = str(uuid.uuid4())
    file_name_only = uploadedfile.name.split('.')[0]
    file_ext_only = uploadedfile.name.split('.')[1]
    # file_path = f'{file_name_only}_{file_id}.{file_ext_only}'
    file_path = uploadedfile.name

    with open(os.path.join(APP_DOCS, file_path), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path


#################### STREAMLIT APP ####################
st.set_page_config(
    page_title="IntelliNews",
    # page_icon="🇵🇭",
    layout="wide"
)

st.header("IntelliNews")
# st.markdown('___')

#################### HELPER FUNCTIONS ####################
def clear_cache():
    # Delete all the items in Session state
    for key in st.session_state.keys():
        del st.session_state[key]
    # # Delete all the documents in directory
    # for file in os.listdir(APP_DOCS):
    #     os.remove(f'{APP_DOCS}/{file}')


#################### TOPIC MODELING ####################
doc_dicts = []
doc_inputs = {}
file_paths = []
topics = []
sub_topics = []
select_topics = []

with st.sidebar:
    # st.header("Documents:")
    pdfs = st.file_uploader('Upload PDF files', type=['pdf'], accept_multiple_files=True, label_visibility='hidden')
    if pdfs:
        clear_cache()
        btn_find = st.button("Find Topics", use_container_width=True)
        if btn_find:
            for idx, pdf in enumerate(pdfs):
                file_path = save_uploadedfile(pdf)
                file_paths.append(file_path)
                st.session_state['file_paths'] = file_paths
                loader = PyPDFLoader(f'{APP_DOCS}/{file_path}')
                doc = loader.load_and_split()

                doc_input = ''
                tmp_doc_input = ''

                for i_tmp, page in enumerate(doc):
                    doc_input += str(page.page_content)
                    tmp_doc_input += remove_non_ascii(str(page.page_content))
                    if i_tmp % PAGE_PER_REQUEST == 0 or (len(doc) == i_tmp + 1):
                        response = None
                        cnt_topic = 0
                        while response is None:
                            sleep(.1)
                            request = get_doc_topic_prompt(TOPIC_COUNT, tmp_doc_input)
                            response = get_gemini_text_response(request)
                            if response is not None:
                                response = response.replace('```json', '').replace('```', '')
                                l = list(eval(response))
                                doc_dict = {'document': file_path, 'topics': l}
                                doc_dicts.append(doc_dict)
                                st.session_state['doc_dicts'] = doc_dicts
                                topics.extend(l)
                                # st.write(f'l to be extended: {l}')
                                # st.write(f'topics.extend: {topics}')
                                break
                            else:
                                cnt_topic += 1
                                if cnt_topic == RETRY_COUNT:
                                    # st.error(f'Failed to process "{file_path}". Please try again.')
                                    break
                        tmp_doc_input = ''

                doc_inputs[file_path] = remove_non_ascii(doc_input).replace('\n\n', '\n').replace('\n', ' ').replace('  ', ' ')
                st.session_state['doc_inputs'] = doc_inputs

            key_topics_prompt = get_key_topics_prompt(topics)
            new_response = None
            cnt_common_topic = 0
            while new_response is None:
                sleep(.1)
                new_response = get_gemini_text_response(key_topics_prompt)
                if new_response is not None:
                    # print(new_response)
                    json_topics = eval(new_response.replace('```json', '').replace('```', ''))
                    key_topics = [t['topic'] for t in json_topics]
                    st.session_state['topics'] = key_topics
                    sub_topics = [t['subtopics'] for t in json_topics]
                    st.session_state['sub_topics'] = sub_topics
                    break
                else:
                    cnt_common_topic += 1
                    if cnt_common_topic == RETRY_COUNT:
                        st.error('Failed to get response. Please try again.')
                        break


        if 'topics' in st.session_state and len(st.session_state['topics']) > 0:
            sts = []
            for it, topic in enumerate(st.session_state['topics']):
                if len(st.session_state['sub_topics'][it]) > 0:
                    sts = st.session_state['sub_topics'][it]
                else:
                    sts.append(topic)
                docs = []
                for doc in st.session_state['doc_dicts']:
                    doc_topics = doc['topics']
                    for doc_t in doc_topics:
                        if doc_t in sts:
                            if doc['document'] not in docs:
                                docs.append(doc['document'])
                            break

                if len(docs) >= 1:
                    select_topics.append({'topic':topic, 'docs':docs})
                    # annotated_text((str(", ".join(docs)), topic.upper()))

                    # st.markdown("""
                    # <style>
                    # .small-font {
                    #     font-size:8px !important;
                    # }
                    # </style>
                    # """, unsafe_allow_html=True)
                    # st.markdown(f'{topic}<p class="small-font">{str(", ".join(docs))}</p>', unsafe_allow_html=True)
                    # st.markdown('___')

            st.session_state['select_topics'] = select_topics
            # colors = get_list_of_random_color(len(select_topics))
            topics_found = [s['topic'] for s in select_topics]
            st.header(f"Topics found ({len(topics_found)}):")
            st.caption(', '.join(topics_found))
            for i, s_topic in enumerate(select_topics):
                # annotated_text((str(", ".join(s_topic['docs'])), s_topic['topic'].upper(), colors[i]))
                annotated_text((str("; \n".join(s_topic['docs'])), s_topic['topic'].upper()))


#################### COMPARER ####################
if 'select_topics' in st.session_state and len(st.session_state['select_topics']) > 0:
    s_topics = [s['topic'] for s in st.session_state['select_topics']]
    s_docs = [s['docs'] for s in st.session_state['select_topics']]
    # select_topic = st.selectbox('Ask a topic:', s_topics)
    # select_topic = str(select_topic)
    # idx = s_topics.index(select_topic)
    question = st.text_area('Ask a question:', placeholder='Type a question here...')
    btn_ask = st.button("Ask")


    cols = st.columns(len(st.session_state['file_paths']))
    for i, f in enumerate(st.session_state['file_paths']):
        with cols[i]:

            loader = PyPDFLoader(f"{APP_DOCS}/{f}")
            doc = loader.load_and_split()
            all_annot_response = ''
            tmp_doc_input = ''
            for i_tmp, page in enumerate(doc):
                tmp_doc_input += remove_non_ascii(str(page.page_content))
                if i_tmp % PAGE_PER_REQUEST == 0 or (len(doc) == i_tmp+1):

                    annot_request = get_annots_request(doc_input=tmp_doc_input, s_topics=s_topics)
                    annot_response = None
                    cnt_annot_topic = 0
                    while annot_response is None:
                        sleep(.1)
                        annot_response = get_gemini_text_response(annot_request)
                        if annot_response is not None:
                            all_annot_response += annot_response
                            break
                        else:
                            cnt_annot_topic += 1
                            if cnt_annot_topic == RETRY_COUNT:
                                # st.error('Failed to get response. Please try again.')
                                all_annot_response += tmp_doc_input
                    tmp_doc_input = ''

            st.markdown('___')
            st.markdown('##### Source:')
            st.caption(f)
            st.markdown('___')


            annot_s_topics = [f'[{s}]' for s in s_topics]
            regex_s = re.compile("(?=(" + "|".join(map(re.escape, annot_s_topics)) + "))")
            # regex_ss = re.findall(regex_s, annot_response)
            annot_e_topics = [f'[/{s}]' for s in s_topics]
            regex_e = re.compile("(?=(" + "|".join(map(re.escape, annot_e_topics)) + "))")
            regex_ee = re.findall(regex_e, all_annot_response)

            annot_all_topics = [f'\[{s}\].*?\[\/{s}\]' for s in s_topics]
            annot_all_topics_regex_f = '|'.join(annot_all_topics)
            regex_all_res = re.findall(annot_all_topics_regex_f, all_annot_response, re.DOTALL)

            annot_texts = []
            all_text = all_annot_response
            for ir, ret in enumerate(regex_all_res):
                # for selected_topic in s_topics:
                # selected_topic = regex_ee[i].replace('[[/', '').replace(']]', '')
                annot = None
                if ret is None or (not isinstance(ret, str)):
                    continue

                selected_topic = re.findall('\[\/.*?\]', ret, re.DOTALL)
                if selected_topic and len(selected_topic)>0:
                    selected_topic = selected_topic[0].replace('[/', '').replace(']', '')
                    
                    new_atext = ret.replace(f'[/{selected_topic}]', '').replace(f'[{selected_topic}]', '')
                    if selected_topic == 'NO_TOPIC':
                        annot_texts.append(new_atext)
                    else:
                        annot_texts.append((new_atext, selected_topic.upper()))
            #         # st.caption(f'selected_topic: {selected_topic}')
            #     # st.caption(f'ret: \n\n{ret}')
            #     # annot = [s for s in all_text.split(ret) if s !='']
            #     annot = all_text.split(ret)
            #     # [st.caption(f'annot[{i}]: {str(a)}') for i,a in enumerate(annot)]
            #     # st.write('___')

            #     prev_text = ''
            #     atext = ''
            #     find_prev = re.findall('\[\/.*?\]', annot[0], re.DOTALL)
            #     if find_prev is None or len(find_prev)==0:
            #         prev_text = annot[0]
            #     atext = ret

            #     if len(prev_text) > 0:
            #         # prev_text = re.sub(r'\[.*?\]', '', prev_text)
            #         # prev_text = re.sub(r'\[\/.*?\]', '', prev_text)
            #         annot_texts.append(prev_text)
            #     if len(atext) > 0:
            #         # new_atext = re.sub(r'\[.*?\]', '', atext)
            #         # new_atext = re.sub(r'\[\/.*?\]', '', new_atext)
            #         new_atext = atext.replace(f'[/{selected_topic}]', '').replace(f'[{selected_topic}]', '')
            #         annot_texts.append((new_atext, selected_topic.upper()))

            #     if len(prev_text) > 0:
            #         all_text = all_text.replace(prev_text, '')
            #     if len(atext) > 0:
            #         all_text = all_text.replace(atext, '')
            #     # all_text = all_text.replace(prev_text, '')
            #     # all_text = all_text.replace(atext, '')

            #     # prev_text = atext
            #     # break

            # # if len(all_text) > 0:
            # if ir == len(regex_all_res)-1 and (find_prev is None or len(find_prev)==0):
            #     try:
            #         annot_texts.append(annot[-1])
            #     except:
            #         pass
            annotated_text([annot_texts])

            if IS_DEBUG:
                st.subheader("Annotated Text:")
                st.write(all_annot_response)




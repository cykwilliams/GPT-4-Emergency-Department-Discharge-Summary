###Clean version of chatgpt_summary_ed220923.ipynb (on RAE)

import os  
import openai  
import tiktoken  
import pandas as pd  
import re
import csv

path = 'pathname'

df = pd.read_parquet(path + 'chatgpt_summary_ed_ed_notes_edprovider_adults_filtered_master_220923.parquet')
df['visit_occurrence_id'] = df['visit_occurrence_id'].astype(str)
df['note_text'] = df['note_text'].astype(str)

####Preprocessing

#Make new admitted column
df.loc[df['hospital_visit_occurrence_id'] == '-1', 'admitted'] = 0
df.loc[df['hospital_visit_occurrence_id'] != '-1', 'admitted'] = 1

print(df.shape)
print(df.hospital_visit_occurrence_id.nunique())
print(df.admitted.value_counts())


##Conduct minimal preprocessing
#Remove '\n' (as upon inspection these appear to be randomly inserted into text)
df["note_text_processed"] = [re.sub(r'\n', '', s) for s in df["note_text"]]
#Confirm this
print(df[df['note_text_processed'].str.contains('\n')].shape, '- should be 0')
#(0, 42)

#Remove extra spaces
def remove_extra_spaces(text):
    # Use regular expressions to replace multiple spaces with a single space
    return re.sub(' +', ' ', text)

df['note_text_processed'] = df['note_text_processed'].apply(remove_extra_spaces)

##Remove duplicate encounters (i.e where duplicate notes exist)
#Check for duplicates
print(len(df[df['visit_occurrence_id'].duplicated(keep = False)]))

#Examine duplicates
pd.set_option('display.max_colwidth', None)
df[df['visit_occurrence_id'].duplicated(keep = False)][['note_datetime','hospital_visit_occurrence_id','note_text']].head(10)
#Many are smaller length attestation notes (with identical note_datetime); others are follow up
#notes (e.g following completion of scans) - hence drop_duplicates keeping first note on note_datetime, than on word count
#(keeping the longest note if note_datetime is the same)

#Sort by note time and then length per above
df['note_datetime'] = pd.to_datetime(df['note_datetime'])
df['note_length_words'] = df['note_text_processed'].str.len()
df = df.sort_values(['visit_occurrence_id', 'note_datetime', 'note_length_words'], ascending=[True, True, False])

#Confirm this
pd.set_option('display.max_colwidth', 40)
df[df.visit_occurrence_id.duplicated(keep = False)].head(14)
#Confirmed

#Drop duplicates
print(df.shape)
df = df.drop_duplicates(subset = 'visit_occurrence_id', keep = 'first')
print(df.shape)


#### Segment note_text_processed

#Examine counts of various section headers within the ED note
df['history_chiefcomplaint'] = df['note_text_processed'].str.contains('History Chief Complaint').apply(lambda x: 'Y' if x else 'N')
df['chiefcomplaint'] = df['note_text_processed'].str.contains('Chief Complaint').apply(lambda x: 'Y' if x else 'N')
df['systemsreview'] = df['note_text_processed'].str.contains('Review of Systems').apply(lambda x: 'Y' if x else 'N')
df['physicalexam'] = df['note_text_processed'].str.contains('Physical Exam').apply(lambda x: 'Y' if x else 'N')
df['edcourse'] = df['note_text_processed'].str.contains('ED Course').apply(lambda x: 'Y' if x else 'N')
df['initialassessment'] = df['note_text_processed'].str.contains('Initial Assessment').apply(lambda x: 'Y' if x else 'N')
df['plan'] = df['note_text_processed'].str.contains('Plan').apply(lambda x: 'Y' if x else 'N')
df['plan2'] = df['note_text_processed'].str.contains('Plan:').apply(lambda x: 'Y' if x else 'N')

for column in ['history_chiefcomplaint', 'chiefcomplaint', 'systemsreview',
       'physicalexam', 'edcourse', 'initialassessment', 'plan', 'plan2']:
    print('\n', column)
    print(df[column].value_counts())


#Create functions to extract History, Physical Examination and Initial Assessment/Plan sections (so can later exclude any notes without all three of these sections)
def extract_text(text, start_pattern, end_pattern):
    start_regex = re.compile('|'.join(start_pattern))
    end_regex = re.compile('|'.join(end_pattern))

    try:
        start_match = start_regex.search(text) 
        end_match = end_regex.search(text) 
        start = start_match.start()
        end = end_match.start()
        result = text[start:end]
    except AttributeError:
        result = 'unable_to_extract'
    return result


def extract_initialassessment_to_end(text, initialassessment, edcourse):
    #Search first for 'Initial Assessment' and if present select text from there to end
    #Otherwise search for 'ED Course' and do the same
    #Otherwise return 'unable_to_extract'
    initialassessment_regex = re.compile('|'.join(initialassessment))
    edcourse_regex = re.compile('|'.join(edcourse))
    
    start_match = initialassessment_regex.search(text)
    if start_match is None:
        start_match = edcourse_regex.search(text)
    if start_match is None:
        return 'unable_to_extract'
    start = start_match.start()
    return text[start:]


# Apply the function and create new columns for each section

#Create list of upper/lower case variations of desired note heading
#Note that e.g all lowercase 'initial assessment' has several false positives, so settle for only 1) First letter caps and 2) all caps
chiefcomplaint = ['Chief Complaint', 'CHIEF COMPLAINT']
physicalexam = ['Physical Exam', 'PHYSICAL EXAM']
initialassessment = ['Initial Assessment', 'INITIAL ASSESSMENT']
edcourse = ['ED Course', 'ED course', 'ED COURSE']

df['history_text'] = df['note_text_processed'].apply(lambda x: extract_text(x, chiefcomplaint, physicalexam)) 

df['examination_text'] = df['note_text_processed'].apply(lambda x: extract_text(x, physicalexam, initialassessment) 
                                               if any(s in x for s in initialassessment) else extract_text(x, physicalexam, edcourse) 
                                               if any(s in x for s in edcourse) else 'unable_to_extract')

df['assessment_plan_text'] = df['note_text_processed'].apply(lambda x: extract_initialassessment_to_end(x, initialassessment, edcourse))

print(df.shape)
print(df.note_text_processed.isnull().sum())
print(df.history_text.isnull().sum())
print(df.examination_text.isnull().sum())
print(df.assessment_plan_text.isnull().sum())

print((df['history_text'] == 'unable_to_extract').sum())
print((df['examination_text'] == 'unable_to_extract').sum())
print((df['assessment_plan_text'] == 'unable_to_extract').sum())

#Count any values of '' (e.g if the second regex pattern comes before the first)
print(df[df['history_text'] == ''].shape)
print(df[df['examination_text'] == ''].shape)
print(df[df['assessment_plan_text'] == ''].shape)

#Count number of null values (shouldn't be any - they should all be 'unable_to_extract')
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print(df[text].isnull().sum())

#### Get token count
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

df['note_text_processed_tokens'] = df['note_text_processed'].apply(lambda x: num_tokens_from_string(x))

#### Create df_refined

###Create df_refined
df['visit_occurrence_id'] = df['visit_occurrence_id'].astype(str)
df['note_text'] = df['note_text'].astype(str)

df_refined = df.copy()
#Remove null values ['history_text', 'examination_text', 'assessment_plan_text']
print(df_refined.shape)
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print('Removing nulls from', text)
    df_refined = df_refined[df_refined[text].notnull()]
    print(df_refined.shape)

#Similarly, remove 'unable_to_extract' from ['history_text', 'examination_text', 'assessment_plan_text']
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print('Removing unable_to_extract from', text)
    df_refined = df_refined[df_refined[text] != 'unable_to_extract']
    print(df_refined.shape)
    
#For completeness, remove '' from ['history_text', 'examination_text', 'assessment_plan_text']
for text in ['history_text', 'examination_text', 'assessment_plan_text']:
    print('Removing unable_to_extract from', text)
    df_refined = df_refined[df_refined[text] != '']
    print(df_refined.shape)
    
print(df_refined[df_refined['note_text_processed_tokens'] >= 3500].shape)
print(df_refined[df_refined['note_text_processed_tokens'] < 3500].shape)

#Hence if excluding notes >3500 tokens in length (full note), will only exclude 2.1%
print(df_refined.shape)
df_refined = df_refined[df_refined['note_text_processed_tokens'] < 3500]
print(df_refined.shape)

###Create slimmed down df
df_notes_only = df_refined[['patientdurablekey', 'visit_occurrence_id', 
       'admitted', 'note_text_processed',
       'note_length_words', 'history_text',
       'examination_text', 'assessment_plan_text',
       'note_text_processed_tokens', 'history_text_tokens',
       'examination_text_tokens', 'assessment_plan_text_tokens']]
print(df_notes_only.shape)
print(df_notes_only.visit_occurrence_id.nunique())
df_notes_only.to_csv(path + 'chatgpt_summary_ed_ed_notes_edprovider_adults_filtered_master_220923_filtered.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')


#######
#### Load final dataset
df_final = pd.read_csv(path + 'chatgpt_summary_ed_ed_notes_edprovider_adults_filtered_master_220923_filtered.csv', index_col = 0)
df_final['visit_occurrence_id'] = df_final['visit_occurrence_id'].astype(str)


### Create discharged-only dataset
print(df_final['admitted'].value_counts())
print(df_final['admitted'].isnull().sum())

df_discharged = df_final[df_final['admitted'] == 0.0]
print(df_discharged.shape)


####Add prompt
prompt = 'You are an Emergency Department physician. Below is the History and Physical Examination note for a patient presenting to the Emergency Department who was subsequently discharged. Write a discharge summary for the patient based on this note. Do not include any additional information not present in the note. \n\n """'
df_discharged['text_summarisation_prompt'] = prompt + df_discharged['note_text_processed'] + '"""'


#### Create n=100 sample from  discharged-only dataset
df_discharged_100 = df_discharged.sample(100, random_state = 13)
print(df_discharged_100.shape)



#### Run OpenAI API
import openai  
import tiktoken  
import pandas as pd  

import os
import re
import json
import base64
import datetime
import requests
import urllib.parse
from dotenv import load_dotenv

from ratelimit import limits, sleep_and_retry


load_dotenv('.env')
API_KEY = os.environ.get('STAGE_API_KEY')
API_VERSION = os.environ.get('API_VERSION')
RESOURCE_ENDPOINT = os.environ.get('RESOURCE_ENDPOINT')

# Sometimes setting these is helpful--sometimes you need to depend on the LLM setting, depending on the library and object
openai.api_type = "azure"  # always use Azure 
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT  # May change depending on which server and Mulesoft key you use
openai.api_version = API_VERSION  # This can be overwritten with an incorrect default if not specified with some langchain objects

##Set temperature (0 = static, 1 = more variable responses)
temperature = 0

# Define the rate limit for the function (e.g. 35 calls per second)
@sleep_and_retry
@limits(calls=295, period=60)
def run_chatgpt_api(prompt, deployment_name):
    print(deployment_name)
    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages = [
                {"role": "user", "content": prompt}
            ],
            n=1,
            stop=None,
            temperature=temperature,
            )
    except:
        response = 'Error_with_API_CYKW'
    return response


def retrieve_content_from_response_json2(x):
    try:
        return json.loads(str(x))['choices'][0]['message']['content']
    except:
        return 'Error_with_API_CYKW'
    
def process_chatgpt_output(df):
    print('Saving temp df')
    import csv
    #Save temp df:
    df.to_csv('df_gender_ed_temp.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
    print('retrieving content')
    df['response_content'] =  df['response_json'].apply(lambda x: retrieve_content_from_response_json2(x))
    #print('retrieving label')
    #df['label'] = df['response_content'].apply(lambda x: retrieve_label(x))
    print('Saving temp df2')
    #Save temp df2
    df.to_csv('df_gender_ed_temp.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
    return df

def combined_lists_to_df_ed(df, index_list, prompt_list, response_list, prompt):
    output_df = pd.DataFrame([index_list, prompt_list, response_list]).transpose()
    output_df = output_df.rename(columns = {0:'index', 1:prompt, 2:'response_json'})
    
    df = df.merge(output_df, left_on = ['index', prompt], right_on = ['index', prompt], how = 'left')
    df = df[df['response_json'].notnull()]
    df = process_chatgpt_output(df)
    df['response_word_count'] = df['response_content'].str.split().str.len()
    return df 

#Function to run df_ed through chatGPT and save output
def chatGPT_run_ed_df(df, deployment_name, prompt_column, suffix):
    print('Running', suffix, 'dataset:')
    df_to_run = df.copy()
    df_to_run['index'] = df_to_run.index
    prompt = prompt_column

    index_list = []
    prompt_list = []
    response_list = []

    for key, value in dict(zip(df_to_run['index'].tolist(), df_to_run[prompt].tolist())).items():
        print(key)
        print(len(response_list))
        index_list.append(key)
        prompt_list.append(value)
        response_list.append(run_chatgpt_api(value, deployment_name))

    df_to_save = combined_lists_to_df_ed(df_to_run, index_list, prompt_list, response_list, prompt)
    print('df.shape (', suffix, '): ', df_to_save.shape)
    df_to_save.to_csv(f'df_summary_ed_{suffix}.csv')
    
    return df_to_save


df_summary_100_gpt35 = chatGPT_run_ed_df(df_discharged_100, 'gpt-35-turbo', 'text_summarisation_prompt', '100_gpt35')
df_summary_100_gpt4 = chatGPT_run_ed_df(df_discharged_100, 'gpt-4', 'text_summarisation_prompt', '100_gpt4')


###Combine both into one df:
#Rename columns
df_summary_100_gpt35 = df_summary_100_gpt35.rename(columns = {'response_json':'response_json_gpt35', 'response_content':'response_content_gpt35', 'response_word_count':'response_word_count_gpt35'})
df_summary_100_gpt4 = df_summary_100_gpt4.rename(columns = {'response_json':'response_json_gpt4', 'response_content':'response_content_gpt4', 'response_word_count':'response_word_count_gpt4'})

#Merge
df_summary_100_results = df_summary_100_gpt35.merge(df_summary_100_gpt4[['visit_occurrence_id', 'index', 'response_json_gpt4', 
                                                                         'response_content_gpt4', 'response_word_count_gpt4']], 
                                                    on = ['visit_occurrence_id', 'index'], how = 'left')
df_summary_100_results['cykw_index'] = range(0, len(df_summary_100_results))

#Save to csv
df_summary_100_results.to_csv(path + 'df_summary_100_results220923.csv', header=True,
          quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')


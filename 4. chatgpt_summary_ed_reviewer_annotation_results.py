##Clean version of chatgpt_ed_summary_RedCap_results.ipynb (local)

#### Import master dataframe (after results have been exported from RedCap + consensus agreement performed + manual categorisation of reviewer comments performed)
import pandas as pd
import csv
path = 'setpath'
df_transposed_concat_binary = pd.read_csv(path + 'df_transposed_concat_binary_all180124_modified.csv', index_col = 0)

#### Report aggregate numbers

### Get counts of scores
print(df_transposed_concat_binary['section_type'].value_counts(), '\n')
#10 categories measured
#There are 600 of each, corresponding to 100 cases x 3 criteria x 2 models

print(df_transposed_concat_binary['final_label_agreed'].value_counts())


#Check reasons
print(df_transposed_concat_binary['final_label_reason'].value_counts(), '\n')
#30 have no_text - check these are for final_label = 0
print(df_transposed_concat_binary[df_transposed_concat_binary['final_label_reason'] == 'no_text']['final_label_agreed'].value_counts())
#They are

##Create Figure 3 data (exported, cleaned and then re-imported to R for Figure)
df_Figure_3 = pd.DataFrame(df_transposed_concat_binary.groupby(['model_type', 'criteria_type', 'section_type'])['final_label_agreed'].sum())
df_Figure_3.to_csv(path + 'df_Figure_3.csv', quoting=csv.QUOTE_ALL)


####Create Table 3 (aggregate of manual categorisation of reviewer explanations for each error)
df_transposed_concat_binary_manual_categories = pd.DataFrame(df_transposed_concat_binary.groupby(['model_type', 'criteria_type'])['final_label_category'].value_counts())
#Save manual categories to csv:
df_transposed_concat_binary_manual_categories.to_csv(path + 'df_transposed_concat_binary_manual_categories180124.csv', quoting=csv.QUOTE_ALL)


### Split into GPT3.5 and GPT4 dfs
pd.set_option('display.max_rows', 300)
print(df_transposed_concat_binary.groupby(['model_type', 'criteria_type'])['final_label_agreed'].value_counts())
print(df_transposed_concat_binary.groupby(['model_type', 'criteria_type'])['final_label_category'].value_counts())

df_transposed_concat_binary_gpt35 = df_transposed_concat_binary[df_transposed_concat_binary['model_type'] == 'gpt35']
df_transposed_concat_binary_gpt4 = df_transposed_concat_binary[df_transposed_concat_binary['model_type'] == 'gpt4']

df_transposed_concat_binary_gpt35_categories = pd.DataFrame(df_transposed_concat_binary_gpt35.groupby(['criteria_type'])['final_label_category'].value_counts())
df_transposed_concat_binary_gpt4_categories = pd.DataFrame(df_transposed_concat_binary_gpt4.groupby(['criteria_type'])['final_label_category'].value_counts())


### Further split by criteria number
df_transposed_concat_binary_gpt35_crit1 = df_transposed_concat_binary_gpt35[df_transposed_concat_binary_gpt35['criteria_type'] == 'crit1']
df_transposed_concat_binary_gpt35_crit2 = df_transposed_concat_binary_gpt35[df_transposed_concat_binary_gpt35['criteria_type'] == 'crit2']
df_transposed_concat_binary_gpt35_crit3 = df_transposed_concat_binary_gpt35[df_transposed_concat_binary_gpt35['criteria_type'] == 'crit3']

df_transposed_concat_binary_gpt4_crit1 = df_transposed_concat_binary_gpt4[df_transposed_concat_binary_gpt4['criteria_type'] == 'crit1']
df_transposed_concat_binary_gpt4_crit2 = df_transposed_concat_binary_gpt4[df_transposed_concat_binary_gpt4['criteria_type'] == 'crit2']
df_transposed_concat_binary_gpt4_crit3 = df_transposed_concat_binary_gpt4[df_transposed_concat_binary_gpt4['criteria_type'] == 'crit3']

##Count number of cases where there is (separately) no inaccuracy/hallucination/omission
df_transposed_concat_binary_gpt35_crit1_combined = pd.DataFrame(df_transposed_concat_binary_gpt35_crit1.groupby('case_number')['final_label_agreed'].sum())
df_transposed_concat_binary_gpt35_crit2_combined = pd.DataFrame(df_transposed_concat_binary_gpt35_crit2.groupby('case_number')['final_label_agreed'].sum())
df_transposed_concat_binary_gpt35_crit3_combined = pd.DataFrame(df_transposed_concat_binary_gpt35_crit3.groupby('case_number')['final_label_agreed'].sum())

df_transposed_concat_binary_gpt4_crit1_combined = pd.DataFrame(df_transposed_concat_binary_gpt4_crit1.groupby('case_number')['final_label_agreed'].sum())
df_transposed_concat_binary_gpt4_crit2_combined = pd.DataFrame(df_transposed_concat_binary_gpt4_crit2.groupby('case_number')['final_label_agreed'].sum())
df_transposed_concat_binary_gpt4_crit3_combined = pd.DataFrame(df_transposed_concat_binary_gpt4_crit3.groupby('case_number')['final_label_agreed'].sum())

for df in [df_transposed_concat_binary_gpt35_crit1_combined, df_transposed_concat_binary_gpt35_crit2_combined, df_transposed_concat_binary_gpt35_crit3_combined,
            df_transposed_concat_binary_gpt4_crit1_combined, df_transposed_concat_binary_gpt4_crit2_combined, df_transposed_concat_binary_gpt4_crit3_combined]:
    print(len(df[df['final_label_agreed'] == 0]))

#64
#36
#50
#90
#58
#53

##Count number of cases where there is not ANY of inaccuracy/hallucination/omission:
df_transposed_concat_binary_gpt35_combined = pd.DataFrame(df_transposed_concat_binary_gpt35.groupby('case_number')['final_label_agreed'].sum())
df_transposed_concat_binary_gpt4_combined = pd.DataFrame(df_transposed_concat_binary_gpt4.groupby('case_number')['final_label_agreed'].sum())

print(len(df_transposed_concat_binary_gpt35_combined[df_transposed_concat_binary_gpt35_combined['final_label_agreed'] == 0]))
print(len(df_transposed_concat_binary_gpt4_combined[df_transposed_concat_binary_gpt4_combined['final_label_agreed'] == 0]))
#10
#33

def pivot_table_by_section_type(df):
    df_select = df[['case_number', 'section_type', 'final_label_agreed']].reset_index(drop = True)
    df_output = df_select.pivot_table(index='case_number', columns='section_type', values='final_label_agreed', aggfunc='first').reset_index()
    df_output = df_output[['case_number', 'PC', 'HPC', 'PMH', 'allergies', 'ROS', 'PE', 'labs', 'imaging', 'plan', 'other']]
    df_output['total'] = df_output[['PC', 'HPC', 'PMH', 'allergies', 'ROS', 'PE', 'labs', 'imaging', 'plan', 'other']].sum(axis=1)
    print('Number of cases with errors (any section):', len(df_output[df_output['total'] > 0]))
    return df_output

df_transposed_concat_binary_gpt35_crit1_pivoted = pivot_table_by_section_type(df_transposed_concat_binary_gpt35_crit1)
df_transposed_concat_binary_gpt35_crit2_pivoted = pivot_table_by_section_type(df_transposed_concat_binary_gpt35_crit2)
df_transposed_concat_binary_gpt35_crit3_pivoted = pivot_table_by_section_type(df_transposed_concat_binary_gpt35_crit3)

df_transposed_concat_binary_gpt4_crit1_pivoted = pivot_table_by_section_type(df_transposed_concat_binary_gpt4_crit1)
df_transposed_concat_binary_gpt4_crit2_pivoted = pivot_table_by_section_type(df_transposed_concat_binary_gpt4_crit2)
df_transposed_concat_binary_gpt4_crit3_pivoted = pivot_table_by_section_type(df_transposed_concat_binary_gpt4_crit3)

# Number of cases with errors (any section): 36
# Number of cases with errors (any section): 64
# Number of cases with errors (any section): 50
# Number of cases with errors (any section): 10
# Number of cases with errors (any section): 42
# Number of cases with errors (any section): 47

##These numbers are reported in Figure 2 (see R script).


#### Create Supplementary Figures 3 and 4

### Supplementary Figure 3

df_transposed_concat_binary_gpt4_categories = df_transposed_concat_binary_gpt4_categories.reset_index(level = 'final_label_category')
#Make crit1, 2 and 3 dfs
crit1_rename_columns = {'follow_up_incorrect' : 'Inaccurate follow-up details',
    'reports_interim_plan' : 'Inaccurately reported the interim plan as the follow-up plan',
    'exam_findings_incorrect' : 'Inaccurate examination findings',
    'inaccurate_ed_management' : 'Inaccurately reported patient’s management in ED',
    'positive_imaging_ignored' : 'Inaccurately reported imaging as normal',
    'social_history_missreported' : 'Inaccurate social history reported'}

crit2_rename_columns = {'redaction_hallucinated' : 'Hallucinated redacted information',
    'hall_follow_up_OPD' : 'Hallucinated outpatient follow-up details',
    'hall_follow_up_returnprecautions' : 'Hallucinated ED return precautions',
    'hall_follow_up_instructions' : 'Hallucinated follow-up instructions',
    'hall_follow_up_pcp' : 'Hallucinated primary care physician follow-up details',
    'hall_ed_management' : 'Hallucinated patient’s management in ED',
    'hall_medication_plan' : 'Hallucinated medication plan',
    'hall_symptom' : 'Hallucinated symptoms',
    'hall_symptom_cause' : 'Hallucinated cause of symptoms',
    'hall_diagnosis' : 'Hallucinated patient’s diagnosis'}

crit3_rename_columns = {'omit_exam_findings' : 'Omission of positive physical examination findings',
    'omit_imaging_done' : 'Omission of imaging performed',
    'omit_ed_management' : 'Omission of details of patient’s management in ED',
    'omit_symptom' : 'Omission of symptom reported',
    'omit_negative_examination_finding' : 'Omission of pertinent negative physical examination findings',
    'omit_PMH' : 'Omission of details of patient’s Past Medical History',
    'omit_medication_hx' : 'Omission of details of patient’s medication history',
    'omit_allergies' : 'Omission of details of patient’s allergies',
    'omit_PSH' : 'Omission of details of patient’s Past Surgical History',
    'omit_labs_done' : 'Omission of laboratory tests performed',
    'omit_labs_pertinent_normal' : 'Omission of pertinent normal laboratory test results',
    'omit_follow_up' : 'Omission of follow-up information',
    'omit_exam_declined' : 'Omission that patient declined physical examination',
    'omit_suspicious_injury_report' : 'Omission of suspicious injury report',
    'omit_diagnosis' : 'Omission of diagnosis',
    'omit_code_stroke_activation' : 'Omission of code stroke activation',
    'omit_bedside_imaging_done' : 'Omission of bedside imaging done',
    'omit_symptom_character' : 'Omission of symptom character',
    'omit_symptom_time_course' : 'Omission of symptom time course',
    'omit_ECG_done' : 'Omission of ECG performed',
    'omit_urine_screen' : 'Omission of urinalysis results'}

df_transposed_concat_binary_gpt4_categories_crit1 = df_transposed_concat_binary_gpt4_categories[df_transposed_concat_binary_gpt4_categories.index == 'crit1']
df_transposed_concat_binary_gpt4_categories_crit1['final_label_category'] = df_transposed_concat_binary_gpt4_categories_crit1['final_label_category'].replace(crit1_rename_columns)

df_transposed_concat_binary_gpt4_categories_crit2 = df_transposed_concat_binary_gpt4_categories[df_transposed_concat_binary_gpt4_categories.index == 'crit2']
df_transposed_concat_binary_gpt4_categories_crit2['final_label_category'] = df_transposed_concat_binary_gpt4_categories_crit2['final_label_category'].replace(crit2_rename_columns)

df_transposed_concat_binary_gpt4_categories_crit3 = df_transposed_concat_binary_gpt4_categories[df_transposed_concat_binary_gpt4_categories.index == 'crit3']
df_transposed_concat_binary_gpt4_categories_crit3['final_label_category'] = df_transposed_concat_binary_gpt4_categories_crit3['final_label_category'].replace(crit3_rename_columns)

##Figure S3A
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set figure size to accommodate all x-axis labels
plt.figure(figsize=(10, 6))  # Adjust the dimensions as needed

ax = df_transposed_concat_binary_gpt4_categories_crit1.plot(kind='bar', x='final_label_category', y='count', legend=False)

# Set labels and title for each plot
plt.xlabel('Categories')
plt.ylabel('Count')
#plt.title(f'Bar Chart for {value}')

plt.xticks(rotation=35, ha='right')

# Set y-axis labels as integers only
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Save the plot to a PDF file
plt.savefig(path + 'Supplementary Figure 3A.pdf', bbox_inches='tight')

# Show the plot
plt.show()
print('\n')

##Figure S3B
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set figure size to accommodate all x-axis labels
plt.figure(figsize=(10, 6))  # Adjust the dimensions as needed

ax = df_transposed_concat_binary_gpt4_categories_crit2.plot(kind='bar', x='final_label_category', y='count', legend=False)

# Set labels and title for each plot
plt.xlabel('Categories')
plt.ylabel('Count')
#plt.title(f'Bar Chart for {value}')

plt.xticks(rotation=35, ha='right')

# Set y-axis labels as integers only
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Save the plot to a PDF file
plt.savefig(path + 'Supplementary Figure 3B.pdf', bbox_inches='tight')

# Show the plot
plt.show()
print('\n')


##Figure S3C
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set figure size to accommodate all x-axis labels
plt.figure(figsize=(10, 6))  # Adjust the dimensions as needed

ax = df_transposed_concat_binary_gpt4_categories_crit3.plot(kind='bar', x='final_label_category', y='count', legend=False)

# Set labels and title for each plot
plt.xlabel('Categories')
plt.ylabel('Count')
#plt.title(f'Bar Chart for {value}')

plt.xticks(rotation=35, ha='right')

# Set y-axis labels as integers only
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Save the plot to a PDF file
plt.savefig(path + 'Supplementary Figure 3C.pdf', bbox_inches='tight')

# Show the plot
plt.show()
print('\n')


### Supplementary Figure 4

df_transposed_concat_binary_gpt35_categories = df_transposed_concat_binary_gpt35_categories.reset_index(level = 'final_label_category')
#Make crit1, 2 and 3 dfs
crit1_rename_columns_gpt35 = {'omit_exam_findings' : 'Omission of positive physical examination findings',
    'omit_imaging_done' : 'Omission of imaging performed',
    'omit_ed_management' : 'Omission of details of patient’s management in ED',
    'omit_symptom' : 'Omission of symptom reported',
    'omit_negative_examination_finding' : 'Omission of pertinent negative physical examination findings',
    'omit_PMH' : 'Omission of details of patient’s Past Medical History',
    'omit_medication_hx' : 'Omission of details of patient’s medication history',
    'omit_allergies' : 'Omission of details of patient’s allergies',
    'omit_PSH' : 'Omission of details of patient’s Past Surgical History',
    'omit_labs_done' : 'Omission of laboratory tests performed',
    'omit_labs_pertinent_normal' : 'Omission of pertinent normal laboratory test results',
    'omit_follow_up' : 'Omission of follow-up information',
    'omit_exam_declined' : 'Omission that patient declined physical examination',
    'omit_suspicious_injury_report' : 'Omission of suspicious injury report',
    'omit_diagnosis' : 'Omission of diagnosis',
    'follow_up_incorrect' : 'Inaccurate follow-up details',
    'social_history_missreported' : 'Inaccurate social history reported',
    'exam_findings_incorrect' : 'Inaccurate examination findings',
    'inaccurate_ed_management' : 'Inaccurately reported patient’s management in ED',
    'time_course_incorrect' : 'Inaccurate timeline of events',
    'diagnosis_incorrect' : 'Inaccurately gives list of differential diagnoses when reporting patient’s final diagnosis',
    'symptoms_incorrect' : 'Inaccurately reports symptoms',
    'reports_interim_plan' : 'Inaccurately reported the interim plan as the follow-up plan',
    'missed_PMH_diagnosis' : 'Inaccurate Past Medical History reported',
    'missreported_as_in_ref_range' : 'Inaccurately reported lab test results as normal',
    'missreported_as_no_imaging' : 'Inaccurately reported no imaging performed',
    'inaccurate_imaging_reporting' : 'Inaccurate imaging reporting',
    'inaccurate_allergy_recorded' : 'Allergies inaccurately reported'}

crit2_rename_columns_gpt35 = {'redaction_hallucinated' : 'Hallucinated redacted information',
    'hall_follow_up_pcp' : 'Hallucinated primary care physician follow-up details',
    'hall_follow_up_instructions' : 'Hallucinated follow-up instructions',
    'hall_follow_up_OPD' : 'Hallucinated outpatient follow-up details',
    'hall__no_further_follow_up_necessary' : 'Hallucinated advice that no further follow up necessary',
    'hall_social_hx' : 'Hallucinated social history',
    'hall_follow_up_returnprecautions' : 'Hallucinated ED return precautions',
    'hall_diagnosis' : 'Hallucinated patient’s diagnosis',
    'hall_ed_management' : 'Hallucinated patient’s management in ED',
    'hall_time_course' : 'Hallucinated admission date',
    'hall_medication_plan' : 'Hallucinated medication plan',
    'hall_differential_diagnosis' : 'Hallucinated differential diagnosis',}

crit3_rename_columns_gpt35 = {'omit_symptom' : 'Omission of symptom reported',
    'omit_exam_findings' : 'Omission of positive physical examination findings',
    'omit_ed_management' : 'Omission of details of patient’s management in ED',
    'omit_PMH' : 'Omission of details of patient’s Past Medical History',
    'omit_imaging_done' : 'Omission of imaging performed',
    'omit_labs_abnormal' : 'Omission of abnormal labs',
    'omit_symptom_time_course' : 'Omission of symptom time course',
    'omit_medication_hx' : 'Omission of details of patient’s medication history',
    'omit_negative_examination_finding' : 'Omission of pertinent negative physical examination findings',
    'omit_PSH' : 'Omission of details of patient’s Past Surgical History',
    'omit_allergies' : 'Omission of details of patient’s allergies',
    'omit_follow_up' : 'Omission of follow-up information',
    'omit_labs_pertinent_normal' : 'Omission of pertinent normal laboratory test results',
    'omit_imaging_improvement' : 'Omission of improved imaging finding',
    'omit_social_hx' : 'Omission of social history detail',
    'omit_suspicious_injury_report' : 'Omission of suspicious injury report',
    'omit_symptom_character' : 'Omission of symptom character',
    'omit_diagnosis' : 'Omission of diagnosis',
    'omit_urine_screen' : 'Omission of urinalysis results',
    'omit_labs_done' : 'Omission of laboratory tests performed'}

df_transposed_concat_binary_gpt35_categories_crit1 = df_transposed_concat_binary_gpt35_categories[df_transposed_concat_binary_gpt35_categories.index == 'crit1']
df_transposed_concat_binary_gpt35_categories_crit1['final_label_category'] = df_transposed_concat_binary_gpt35_categories_crit1['final_label_category'].replace(crit1_rename_columns_gpt35)

df_transposed_concat_binary_gpt35_categories_crit2 = df_transposed_concat_binary_gpt35_categories[df_transposed_concat_binary_gpt35_categories.index == 'crit2']
df_transposed_concat_binary_gpt35_categories_crit2['final_label_category'] = df_transposed_concat_binary_gpt35_categories_crit2['final_label_category'].replace(crit2_rename_columns_gpt35)

df_transposed_concat_binary_gpt35_categories_crit3 = df_transposed_concat_binary_gpt35_categories[df_transposed_concat_binary_gpt35_categories.index == 'crit3']
df_transposed_concat_binary_gpt35_categories_crit3['final_label_category'] = df_transposed_concat_binary_gpt35_categories_crit3['final_label_category'].replace(crit3_rename_columns_gpt35)

##Figure S4A
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set figure size to accommodate all x-axis labels
plt.figure(figsize=(10, 6))  # Adjust the dimensions as needed

ax = df_transposed_concat_binary_gpt35_categories_crit1.plot(kind='bar', x='final_label_category', y='count', legend=False)

# Set labels and title for each plot
plt.xlabel('Categories')
plt.ylabel('Count')
#plt.title(f'Bar Chart for {value}')

plt.xticks(rotation=35, ha='right')

# Set y-axis labels as integers only
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Save the plot to a PDF file
plt.savefig(path + 'Supplementary Figure 4A.pdf', bbox_inches='tight')

# Show the plot
plt.show()
print('\n')

##Figure S4B
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set figure size to accommodate all x-axis labels
plt.figure(figsize=(10, 6))  # Adjust the dimensions as needed

ax = df_transposed_concat_binary_gpt35_categories_crit2.plot(kind='bar', x='final_label_category', y='count', legend=False)

# Set labels and title for each plot
plt.xlabel('Categories')
plt.ylabel('Count')
#plt.title(f'Bar Chart for {value}')

plt.xticks(rotation=35, ha='right')

# Set y-axis labels as integers only
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Save the plot to a PDF file
plt.savefig(path + 'Supplementary Figure 4B.pdf', bbox_inches='tight')

# Show the plot
plt.show()
print('\n')

##Figure S4C
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set figure size to accommodate all x-axis labels
plt.figure(figsize=(10, 6))  # Adjust the dimensions as needed

ax = df_transposed_concat_binary_gpt35_categories_crit3.plot(kind='bar', x='final_label_category', y='count', legend=False)

# Set labels and title for each plot
plt.xlabel('Categories')
plt.ylabel('Count')
#plt.title(f'Bar Chart for {value}')

plt.xticks(rotation=35, ha='right')

# Set y-axis labels as integers only
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Save the plot to a PDF file
plt.savefig(path + 'Supplementary Figure 4C.pdf', bbox_inches='tight')

# Show the plot
plt.show()
print('\n')


#### Post-hoc removal of follow-up + redacted hallucination (for Discussion)

print(df_transposed_concat_binary[df_transposed_concat_binary['criteria_type'] == 'crit2']['final_label_category'].value_counts(), '\n')
# final_label_category
# redaction_hallucinated                  37
# hall_follow_up_OPD                      20
# hall_follow_up_pcp                      20
# hall_follow_up_instructions             13
# hall_follow_up_returnprecautions        10
# hall__no_further_follow_up_necessary     6
# hall_ed_management                       5
# hall_medication_plan                     4
# hall_social_hx                           4
# hall_diagnosis                           3
# hall_differential_diagnosis              1
# hall_symptom                             1
# hall_symptom_cause                       1
# hall_time_course                         1

##Hence, remove ['redaction_hallucinated', 'hall_follow_up_OPD', 'hall_follow_up_pcp', 'hall_follow_up_returnprecautions'] 
posthoc_followup_categories = ['redaction_hallucinated', 'hall_follow_up_OPD', 'hall_follow_up_pcp', 'hall_follow_up_returnprecautions'] 
df_transposed_concat_binary_posthoc_minus_followup = df_transposed_concat_binary.copy()
df_transposed_concat_binary_posthoc_minus_followup.loc[df_transposed_concat_binary_posthoc_minus_followup['final_label_category'].isin(posthoc_followup_categories), 'final_label_agreed'] = 0

#Confirm change
print(df_transposed_concat_binary['final_label_agreed'].value_counts())
print(df_transposed_concat_binary_posthoc_minus_followup['final_label_agreed'].value_counts())

df_transposed_concat_binary_posthoc_minus_followup_gpt35 = df_transposed_concat_binary_posthoc_minus_followup[df_transposed_concat_binary_posthoc_minus_followup['model_type'] == 'gpt35']
df_transposed_concat_binary_posthoc_minus_followup_gpt4 = df_transposed_concat_binary_posthoc_minus_followup[df_transposed_concat_binary_posthoc_minus_followup['model_type'] == 'gpt4']

##Post-hoc analysis: Count number of cases where there is not ANY of inaccuracy/hallucination/omission:
##Count number of cases where there is not ANY of inaccuracy/hallucination/omission:
df_transposed_concat_binary_posthoc_minus_followup_gpt35_combined = pd.DataFrame(df_transposed_concat_binary_posthoc_minus_followup_gpt35.groupby('case_number')['final_label_agreed'].sum())
df_transposed_concat_binary_posthoc_minus_followup_gpt4_combined = pd.DataFrame(df_transposed_concat_binary_posthoc_minus_followup_gpt4.groupby('case_number')['final_label_agreed'].sum())

print(len(df_transposed_concat_binary_posthoc_minus_followup_gpt35_combined[df_transposed_concat_binary_posthoc_minus_followup_gpt35_combined['final_label_agreed'] == 0]))
print(len(df_transposed_concat_binary_posthoc_minus_followup_gpt4_combined[df_transposed_concat_binary_posthoc_minus_followup_gpt4_combined['final_label_agreed'] == 0]))
#23
#47
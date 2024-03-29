import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np
import textstat

path = 'pathname'
df_summary_100_results = pd.read_csv(path + 'df_summary_100_results220923.csv', index_col = 0)
#Import original master df (from which n=100 sample was sampled)
df_master = pd.read_parquet(path + 'chatgpt_summary_ed_ed_notes_edprovider_adults_filtered_master_220923.parquet')

#### Encounter-specific summary stats (Table 1)

##Add race_ethnicity column to df_DC_summaries from patient_key_table
patient_key_table = pd.read_parquet(path + 'patient_key_table')
patient_key_table_filtered = patient_key_table[['person_id', 'sex', 'genderidentity', 'race_ethnicity', 'birthdate', 'deathdate']]

#Create df_DC_summaries_demog
df_DC_summaries_demog = df_summary_100_results.merge(df_master[['visit_occurrence_id', 'sex', 'edvisit_age', 'dischargedisposition']], how = 'left', on = 'visit_occurrence_id')


#Add these to df_DC_summaries_demog
df_DC_summaries_demog = df_DC_summaries_demog.merge(patient_key_table_filtered[['person_id', 'race_ethnicity']], on = 'person_id', how = 'left')
print(df_DC_summaries_demog.shape)
#Check no nulls after merge
print(df_DC_summaries_demog['race_ethnicity'].isnull().sum())
#0

##Get summary stats
print(df_DC_summaries_demog['edvisit_age'].describe())
print(df_DC_summaries_demog['sex'].value_counts().head(10), '\n')
print(df_DC_summaries_demog['race_ethnicity'].value_counts().head(10), '\n')
print(df_DC_summaries_demog['dischargedisposition'].value_counts().head(10), '\n')

print(df_DC_summaries_demog['primarychiefcomplaintname'].value_counts().head(10), '\n')
print(df_DC_summaries_demog['acuitylevel'].value_counts().head(10))


#### GPT-specific summary stats
### Examine original text word count
#Get word count
df_DC_summaries['note_length_words'] = df_DC_summaries['note_text_processed'].str.split().str.len()

##Supplementary Figure 1
# Assuming df_DC_summaries is your DataFrame
df_DC_summaries['note_length_words'].hist()

# Adding labels to the axes
plt.xlabel('Note length (words)', labelpad=10)
plt.ylabel('Count', labelpad=10)

# Save the plot to a PDF file
plt.savefig(path + 'Supplementary Figure 1.pdf')

# Show the plot
plt.show()
print(df_DC_summaries['note_length_words'].describe())

### Examine DC summary word count

##Supplementary Figure 2
# Assuming df_DC_summaries is your DataFrame
df_DC_summaries[['response_word_count_gpt35', 'response_word_count_gpt4']].rename(columns = {'response_word_count_gpt35':'a) GPT-3.5-turbo', 'response_word_count_gpt4':'b) GPT-4'}).plot.hist(alpha=0.5, bins=10, color=['blue', 'orange'])

# Adding labels to the axes
plt.xlabel('Note length (words)', labelpad=10)
plt.ylabel('Count', labelpad=10)

# Save the plot to a PDF file
plt.savefig(path + 'Supplementary Figure 2.pdf')

# Show the plot
plt.show()


##Test statistical significance
median_4 = df_DC_summaries['response_word_count_gpt4'].median()
q25_4 = np.percentile(df_DC_summaries['response_word_count_gpt4'], 25)
q75_4 = np.percentile(df_DC_summaries['response_word_count_gpt4'], 75)

median_35 = df_DC_summaries['response_word_count_gpt35'].median()
q25_35 = np.percentile(df_DC_summaries['response_word_count_gpt35'], 25)
q75_35 = np.percentile(df_DC_summaries['response_word_count_gpt35'], 75)

print('GPT-4 median + IQR:', median_4, '(', q25_4, '-', q75_4, ')')
print('GPT-3.5 median + IQR:', median_35, '(', q25_35, '-', q75_35, ')')

statistic, p_value = mannwhitneyu(df_DC_summaries['response_word_count_gpt35'], df_DC_summaries['response_word_count_gpt4'])

print("Mann-Whitney U test: U-statistic =", statistic, "p-value =", p_value)
# GPT-4 median + IQR: 235.0 ( 205.0 - 264.5 )
# GPT-3.5 median + IQR: 369.5 ( 307.75 - 445.0 )
# Mann-Whitney U test: U-statistic = 8978.5 p-value = 2.474853533202504e-22


### Calculate health literacy scores
def get_health_literacy_scores(df_dcsummary):
    df_dcsummary['note_text_processed_FREL'] = df_dcsummary['note_text_processed'].apply(lambda x: textstat.flesch_reading_ease(x))
    df_dcsummary['note_text_processed_FKGL'] = df_dcsummary['note_text_processed'].apply(lambda x: textstat.flesch_kincaid_grade(x))
    print('Next')
    df_dcsummary['response_content_gpt35_FREL'] = df_dcsummary['response_content_gpt35'].apply(lambda x: textstat.flesch_reading_ease(x))
    df_dcsummary['response_content_gpt35_FKGL'] = df_dcsummary['response_content_gpt35'].apply(lambda x: textstat.flesch_kincaid_grade(x))
    print('Next')
    df_dcsummary['response_content_gpt4_FREL'] = df_dcsummary['response_content_gpt4'].apply(lambda x: textstat.flesch_reading_ease(x))
    df_dcsummary['response_content_gpt4_FKGL'] = df_dcsummary['response_content_gpt4'].apply(lambda x: textstat.flesch_kincaid_grade(x))

    
    return df_dcsummary

df_DC_summaries = get_health_literacy_scores(df_DC_summaries)

## Flesch reading ease
median_4_FREL = df_DC_summaries['response_content_gpt4_FREL'].median()
q25_4_FREL = np.percentile(df_DC_summaries['response_content_gpt4_FREL'], 25)
q75_4_FREL = np.percentile(df_DC_summaries['response_content_gpt4_FREL'], 75)

median_35_FREL = df_DC_summaries['response_content_gpt35_FREL'].median()
q25_35_FREL = np.percentile(df_DC_summaries['response_content_gpt35_FREL'], 25)
q75_35_FREL = np.percentile(df_DC_summaries['response_content_gpt35_FREL'], 75)

print('GPT-4 median_FREL + IQR:', median_4_FREL, '(', q25_4_FREL, '-', q75_4_FREL, ')')
print('GPT-3.5 median_FREL + IQR:', median_35_FREL, '(', q25_35_FREL, '-', q75_35_FREL, ')')

statistic_FREL, p_value_FREL = mannwhitneyu(df_DC_summaries['response_content_gpt35_FREL'], df_DC_summaries['response_content_gpt4_FREL'])

print("Mann-Whitney U test_FREL: U-statistic =", statistic_FREL, "p-value =", p_value_FREL)
# GPT-4 median_FREL + IQR: 48.6 ( 41.0075 - 51.989999999999995 )
# GPT-3.5 median_FREL + IQR: 46.725 ( 39.6875 - 49.545 )
# Mann-Whitney U test_FREL: U-statistic = 4325.0 p-value = 0.09932166838486071


##Flesch-Kincaid Grade Level
median_4_FKGL = df_DC_summaries['response_content_gpt4_FKGL'].median()
q25_4_FKGL = np.percentile(df_DC_summaries['response_content_gpt4_FKGL'], 25)
q75_4_FKGL = np.percentile(df_DC_summaries['response_content_gpt4_FKGL'], 75)

median_35_FKGL = df_DC_summaries['response_content_gpt35_FKGL'].median()
q25_35_FKGL = np.percentile(df_DC_summaries['response_content_gpt35_FKGL'], 25)
q75_35_FKGL = np.percentile(df_DC_summaries['response_content_gpt35_FKGL'], 75)

print('GPT-4 median_FKGL + IQR:', median_4_FKGL, '(', q25_4_FKGL, '-', q75_4_FKGL, ')')
print('GPT-3.5 median_FKGL + IQR:', median_35_FKGL, '(', q25_35_FKGL, '-', q75_35_FKGL, ')')

# Assuming 'variable1' and 'variable2' are your two variables
statistic_FKGL, p_value_FKGL = mannwhitneyu(df_DC_summaries['response_content_gpt35_FKGL'], df_DC_summaries['response_content_gpt4_FKGL'])

print("Mann-Whitney U test_FKGL: U-statistic =", statistic_FKGL, "p-value =", p_value_FKGL)
# GPT-4 median_FKGL + IQR: 10.0 ( 9.475 - 11.1 )
# GPT-3.5 median_FKGL + IQR: 10.7 ( 9.7 - 11.65 )
# Mann-Whitney U test_FKGL: U-statistic = 5921.5 p-value = 0.024363055463824813
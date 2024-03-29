import pandas as pd
import os

path = 'setpath'
df_summary_100_results = pd.read_csv(path + 'df_summary_100_results220923.csv', index_col = 0)

###Write to .txt files (for upload to RedCap for reviewer evaluation)
df_summary_100_results['orig_text_export'] = '########## Original note ########## \n\n Index:' + df_summary_100_results['cykw_index'].astype(str) + '\n encounterkey:' + df_summary_100_results['visit_occurrence_id'] + '\n\n' + df_summary_100_results['text_summarisation_prompt'] + '\n\n'
df_summary_100_results['gpt35_export'] = '########## GPT-3.5 ########## \n' + df_summary_100_results['response_content_gpt35'] + '\n\n'
df_summary_100_results['gpt4_export'] = '########## GPT-4 ########## \n' + df_summary_100_results['response_content_gpt4']


def write_selected_columns_to_txt(df, columns_to_write, output_directory):
    """
    Write select columns for each row in the DataFrame to separate .txt files.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    columns_to_write (list): List of column names to write for each row.
    output_directory (str): Directory where the .txt files will be saved.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for index, row in df.iterrows():
        file_name = f"{index}.txt"  # Using the index as the file name
        file_path = os.path.join(output_directory, file_name)

        selected_data = [str(row[col]) for col in columns_to_write]
        selected_data_str = '\n'.join(selected_data)

        with open(file_path, 'w') as file:
            file.write(selected_data_str)

columns_to_write = ['orig_text_export', 'gpt35_export', 'gpt4_export']
output_directory = 'output_dir'
write_selected_columns_to_txt(df_summary_100_results, columns_to_write, output_directory)
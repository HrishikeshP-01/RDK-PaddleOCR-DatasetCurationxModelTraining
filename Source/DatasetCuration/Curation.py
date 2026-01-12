import pandas as pd
import os
from .gen_label import generate_labels_for_training

def curate_dataset(dataset_file_path, label_info, output_file_path):
    df = pd.read_csv(dataset_file_path)
    queries = []
    for key in label_info.keys():
        if label_info[key]['threshold_operator'] == 'Greater than':
            query = f"`{key}` > {label_info[key]['threshold_value']}"
        elif label_info[key]['threshold_operator'] == 'Lesser than':
            query = f"`{key}` < {label_info[key]['threshold_value']}"
        if label_info[key]['threshold_operator'] == 'In range':
            lower_threshold, upper_threshold = label_info[key]['threshold_value'].replace('(', '').replace(')', '').split(',')
            query = f"`{key}` > {lower_threshold} and `{key}` < {upper_threshold}"
        queries.append(query)
    filter_query = ' and '.join(queries)
    print(filter_query)
    filtered_df = df.query(filter_query, engine='python')
    filtered_df.to_csv(output_file_path+'curated_dataset.csv', header=not os.path.exists(output_file_path+'/curated_dataset.csv'))
    df_for_training = filtered_df.loc[:, ['file_path', 'prediction']]
    df_for_training.to_csv(output_file_path+'curated_dataset_for_training.csv', header=False, index=False)
    generate_labels_for_training(output_file_path+'curated_dataset_for_training.csv', output_file_path+'labels_for_training.txt')
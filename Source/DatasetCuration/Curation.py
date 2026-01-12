import pandas as pd

def curate_dataset(dataset_file_path, label_info, output_file_path):
    df = pd.read_csv(file_path)
    queries = ''
    for key in label_info.keys():
        if label_info[key]['threshold_operator'] == 'Greater than':
            query = f"{key} > {label_info[key]['threshold_value']}"
        elif label_info[key]['threshold_operator'] == 'Lesser than':
            query = f"{key} < {label_info[key]['threshold_value']}"
        if label_info[key]['threshold_operator'] == 'In range':
            lower_threshold, upper_threshold = label_info[key]['threshold_value'].replace('(', '').replace(')', '').split(',')
            query = f"{key} > {lower_threshold} and {key} < {upper_threshold}"
        queries.append(query)
    filter_query = 'and'.join(queries)
    filtered_df = df.query(filter_query)
    df.to_csv(output_file_path+'/curated_dataset.csv', mode='a', header=not os.path.exists(args.output_file))
    df.to_csv(output_file_path+'/curated_dataset_for_training.csv', mode='a', header=False))
import os

from const import beautified_model_names


def write_results(results_df, dir_path, parameters_dict=None):
    if results_df is None or dir_path is None:
        return
    description = results_df.describe()
    os.makedirs(dir_path, exist_ok=True)
    if parameters_dict is not None:
        with open(f'{dir_path}/parameters.txt', 'w') as f:
            for key, value in parameters_dict.items():
                f.write(f'{key}: {value}\n')
    description.to_csv(f'{dir_path}/result_statistics.csv')
    results_df.to_csv(f'{dir_path}/predictions.csv', index=False)


def parse_name_to_dict(name):
    parts = name.split('--')
    model_name = beautified_model_names.get(parts[2], parts[2])
    result_dict = {'company': parts[1], 'result_dir': name, 'model_name': model_name}
    for part in parts[3:]:
        key_value = part.split('-')
        if len(key_value) == 2:
            key, value = key_value[0], key_value[1]
            if value.isdigit():
                result_dict[key] = int(value)
            elif value.lower() in ['true', 'false']:
                result_dict[key] = value.lower() == 'true'
            else:
                result_dict[key] = value
    if 'add_paraphrased_definition' not in result_dict:
        result_dict['add_paraphrased_definition'] = False
    if 'add_adversarial_definition' not in result_dict:
        result_dict['add_adversarial_definition'] = False
    if 'top_k' not in result_dict:
        result_dict['top_k'] = None
    if 'top_p' not in result_dict:
        result_dict['top_p'] = None
    if 'do_sample' not in result_dict:
        result_dict['do_sample'] = False
    return result_dict

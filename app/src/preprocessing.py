# Import libraries
import pandas as pd

# Define column types
target_col = 'binary_target'
all_features = ['сумма', 'частота_пополнения', 'доход', 'сегмент_arpu', 'частота', 'объем_данных', 'on_net',
                'продукт_1', 'продукт_2', 'зона_1', 'зона_2', 'секретный_скор', 'pack_freq',
                'сумма_одного_пополнения', 'nan_count']


def import_data(path_to_file):
    # Get input dataframe
    input_df = pd.read_csv(path_to_file)
    return input_df


# Main preprocessing function
def run_preproc(input_df):
    input_df['nan_count'] = input_df.isnull().sum(axis=1)
    input_df['сумма_одного_пополнения'] = input_df['сумма'] / input_df['частота_пополнения']

    return input_df[all_features]

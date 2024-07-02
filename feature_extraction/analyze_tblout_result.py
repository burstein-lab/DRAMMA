import pandas as pd
from decimal import Decimal
import numpy as np


RELEVANT_COLS = ['target_name', 'query_name', 'query_accession', 'eval_full_seq', 'score_full_seq', 'rep']
REG = r"\s{1,}"  # find 1 or more spaces


def filter_by_threshold(df, parameter_limit, by_e_value: bool):
    '''
    removes from the input dataframe all the genes that don't pass the threshold
    :param df: dataframe created from tblout file
    :param parameter_limit: number(int/float/decimal), the wanted threshold
    :param by_e_value: boolean, determines if the threshold is by eVal or score
    :return: DataFrame, the input dataframe without the values that did not pass the threshold
    '''
    if by_e_value:
        output = df.loc[df['eval_full_seq'] <= float(Decimal(parameter_limit))]
    else:
        output = df.loc[df['score_full_seq'] >= float(Decimal(parameter_limit))]
    return output


def count_hits(df: pd.DataFrame):
    '''
    for each gene, counts the number of hits that passed the threshold he has
    :return: int
    '''
    output = df.groupby(['target_name']).size().reset_index(name='hit_threshold_count')
    return output


def get_best_hits(df: pd.DataFrame) -> pd.DataFrame:
    '''
    for each gene, finds the best score he got, and represent only it's information
    :return: dataframe, so each gene appears once
    '''
    local_df = df.sort_values('score_full_seq', ascending=False).drop_duplicates('target_name')
    # local_df.reset_index(inplace=True)  # create a best hits dataframe
    return local_df


def get_headers(type: int = 0):
    headers_dict = {'id_cols': ["target_name", "target_accession", "query_name", "query_accession"],
                    'full_seq_cols': ["eval_full_seq", "score_full_seq", "bias_full_seq"],
                    'best_dom_cols': ["e_val_best_dom", "score_best_dom", "bias_best_dom"],
                    'misc_cols': ["exp", "reg", "clu", "ov", "env", "dom", "rep", "inc", "target_description"]
                    }

    # if type is 1, return a list
    if type == 1:
        return headers_dict['id_cols'] + headers_dict['full_seq_cols'] + headers_dict['best_dom_cols'] + \
               headers_dict['misc_cols']

    # default - return a dictionary
    return headers_dict


def force_data_types(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Transform dataframe datatypes into numeric or string format, based on the data types in the original
    hmmsearch output format.
    :param df: The dataframe containing the data
    :return: A copy of the original dataframe with the new data types
    '''
    headers = get_headers()
    # Force data types for the cells of the DataFrame for later manipulations
    float_cols = headers['full_seq_cols'] + headers['best_dom_cols'] + ['exp']
    int_cols, str_cols = headers['misc_cols'][1:-1], headers['misc_cols'][-1]
    df[float_cols] = df[float_cols].astype(float)
    df[int_cols] = df[int_cols].astype(int)
    df[str_cols] = df[str_cols].astype(str)
    return df


def parse_tblout_to_pd(path: str) -> pd.DataFrame:
    '''
    The method reads a csv tblout file from {path} then turns it into a dataframe.
    assumes csv is in a tblout format -> output of hmmsearch with -tblout.
    lines that start with '#' are ignores, and cols are seprated by 2 or more spaces
    :param path: path to read tblout csv from.
    :return: a dataframe, with same columns as csv in {path} and a row for each row in the file
    '''
    headers = get_headers(1)
    # read file to dataframe with a single column containing all data
    df = pd.read_csv(path, header=None, index_col=0, sep='\t')

    # drop lines starting with # from data
    mask_idx = df.index.map(lambda x: not x.startswith("#"))
    masked_df = df.loc[mask_idx, :].reset_index().rename(columns={0: 'col'})

    # Create a dictionary for column names
    col_dict = dict(zip(np.arange(len(headers)), headers))
    # split cols by spaces
    masked_df = masked_df['col'].str.split(REG, expand=True).rename(columns=col_dict)

    # replace Nones created in the end of each row by extra spaces by empty strings and the rejoin them
    masked_df.iloc[:, 18:] = masked_df.iloc[:, 18:].fillna('')
    if len(masked_df) == 0:  # if dataframe is empty, return an empty df with the relevant columns
        return force_data_types(pd.DataFrame(columns=headers))
    masked_df['target_description'] = masked_df['target_description'].str.cat(masked_df.iloc[:, 19:], sep=' ')
    df = masked_df.iloc[:, :19]
    df = force_data_types(df)
    return df


def process_tblout_file(file_name, threshold, by_e_value, only_higher_than_threshold=True):
    df = parse_tblout_to_pd(file_name)
    df['eval_full_seq'] = df['eval_full_seq'].apply(Decimal)
    df_filtered = df.loc[:, RELEVANT_COLS]
    df_filtered_by_threshold = filter_by_threshold(df_filtered, threshold, by_e_value)
    number_of_hits_df = count_hits(df_filtered_by_threshold)
    if only_higher_than_threshold:
        best_hits_df = get_best_hits(df_filtered_by_threshold)
    else:
        best_hits_df = get_best_hits(df_filtered)
    output = pd.merge(best_hits_df, number_of_hits_df, how='outer', on='target_name')

    if not only_higher_than_threshold:
        output = output.fillna(0)

    return output.rename(columns={'target_name': 'ID', 'score_full_seq': 'best_score', 'eval_full_seq': 'best_eval'})


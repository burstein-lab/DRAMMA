import pandas as pd
import numpy as np
import os
import re


TAX_ID_TO_DESC = pd.read_pickle(os.path.join("..", 'data', 'WGS_genomes_lineage_mapping.pkl'))
WGS_TAX_MAPPING = pd.read_pickle(os.path.join("..", 'data', 'WGS_gca_to_tax_mapping.pkl'))


def get_taxonomy_by_id(ind):
    if "metagenome" in ind.lower():
        return
    gca = re.search("GCA_[0-9]+\.[0-9]", ind)
    if gca is not None and gca.group() in WGS_TAX_MAPPING.index:
        tax_id = WGS_TAX_MAPPING.loc[gca.group(), 'taxid']
        lineage = TAX_ID_TO_DESC.loc[tax_id, 'lineage']
        if 'metagenome' not in lineage:
            return tax_id


def fill_missing_tax_level(row, tax_level):
    if f"tax_level_{tax_level}" not in row.index:
        return np.nan
    if pd.notnull(row[f"tax_level_{tax_level}"]):
        return row[f"tax_level_{tax_level}"]
    return fill_missing_tax_level(row, tax_level-1)


def tax_id_to_lineage(tax_id):
    if pd.isnull(tax_id):
        return []
    if tax_id in TAX_ID_TO_DESC.index:
        return TAX_ID_TO_DESC.loc[tax_id, 'lineage'].split("; ")
    candidates = TAX_ID_TO_DESC[TAX_ID_TO_DESC['species_taxid'] == tax_id]
    if len(candidates) > 0:
        return candidates.iloc[0]['lineage'].split("; ")
    return []


def get_tax_hierarchy(df, filter_euk=False):
    df['tax_lst'] = df['tax_id'].apply(tax_id_to_lineage)
    if filter_euk:
        df['domain'] = df['tax_lst'].apply(lambda lst: lst[1] if len(lst) > 0 else "")
        df = df[df['domain'] != 'Eukaryota'].drop(columns=['domain'])
    df['tax_lst'] = df['tax_lst'].apply(lambda lst: lst[2:] if len(lst) > 0 else lst)
    max_len = df['tax_lst'].apply(len).max()
    new_cols = [f"tax_level_{i + 1}" for i in range(max_len)]
    df[new_cols] = df['tax_lst'].apply(lambda lst: pd.Series(lst + [None] * (max_len - len(lst))))
    return df
import pandas as pd

COlS = ["contig_id", "source", "type", "start_index", "end_index", "score", "strand", "phase", "attributes"]


def create_df_from_gff(file_name):
    df = pd.read_table(file_name, comment='#', names=COlS, dtype=str)
    df = df.loc[df['type'] == 'CDS']
    df = df.drop(['source', "type", 'score', 'strand', 'phase'], axis=1).dropna(0)
    df = add_name_and_description_to_df(df)
    return df


def add_name_and_description_to_df(df):
    genes_name = []
    description = []
    for line in df.attributes:
        words = line.split(';')
        genes_name.append(words[0][3:])  # adds the name of the gene without 'ID=' to the list
        description.append((words[-1][8:]))
    df.insert(0, 'ID', genes_name)
    df.insert(5, 'description', description)
    df.drop_duplicates(subset='ID', keep='first', inplace=True)
    return df.drop(['attributes'], axis=1)

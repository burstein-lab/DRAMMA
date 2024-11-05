import pandas as pd
from functools import reduce
from .gff_to_df import create_df_from_gff
from .analyze_tblout_result import process_tblout_file
from utilities import feature_to_file
DROP_AMR_FAM_LIST = ['vanA', 'vanB', 'vanC', 'vanD', 'vanH', 'vanR', 'vanS', 'vanT', 'vanW', 'vanX', 'vanY', 'vanZ',
                     'Erm23S_rRNA_methyltrans', 'tet_ribosomoal_protect', 'APH3"']
COLS = ['passed_threshold', 'query_name', 'query_accession', 'best_eval', 'best_score',
                'hit_threshold_count', 'contig_total_resistance_genes',
                'closest_gene_by_distance', 'hmm_accession1', 'hmm_query_name1', 'eVal1', 'score1',
                'cls_gene_description', 'closest_gene_by_nucleotide',
                'hmm_accession2', 'hmm_query_name2', 'eVal2', 'score2', 'cls_nuc_description']


def create_hmm_df(gff_df, hmm_file_name, threshold, by_eval):
    hmm_df = process_tblout_file(hmm_file_name, threshold, by_eval)
    hmm_df_filtered = hmm_df[~hmm_df['query_name'].isin(DROP_AMR_FAM_LIST)]
    merged_df = gff_df.set_index('ID').join(hmm_df_filtered.set_index('ID'), how='right').reset_index()
    return merged_df


def expand_gff_cols(gff_df):
    """
    add headers to the data frame, and assigns default values to the missing values
    """
    # orf_window_MISSING = distance_window*2
    # nucleotide_window_MiSSING = nucleotide_window*2

    gff_df_expand = gff_df.assign(passed_threshold=False, query_name=None, query_accession=None, best_eval=None,
                                  best_score=None, hit_threshold_count=None, contig_total_resistance_genes=0,
                                  genes_in_distance_window=0, closest_gene_by_distance=None, gene_distance=None,
                                  hmm_accession1=None, hmm_query_name1=None, eVal1=None, score1=None,
                                  cls_gene_description=None, genes_in_nucleotide_window=0,
                                  closest_gene_by_nucleotide=None, nucleotide_distance=None, hmm_accession2=None,
                                  hmm_query_name2=None, eVal2=None, score2=None, cls_nuc_description=None)
    return gff_df_expand


def create_contig_dict(df):
    """
    goes through the gff data frame, creates a dictionary in which, each key is a contig, and each value is
    a data frame that contains all of the genes in this contig
    :param df: the expanded gff df
    :return: dictionary
    """
    contig_dict = {}
    for contig_id, contig_df in split_df_by_contig(df):
        contig_dict[contig_id] = contig_df
    return contig_dict


def split_df_by_contig(merged_df):
    """
    splits the dataframe into contig's dataframes, so for each contig there is a dataframe contains only it's genes.
    :return: each time the function is being called, will return the next contig's dataframe
    """
    contig_df_list = [(contig_id, frame) for contig_id, frame in merged_df.groupby('contig_id')]
    yield from contig_df_list


def fill_fields_in_contig_dict_by_distance(contig_dict, hmm_df: pd.DataFrame, gene_window, nucleotide_window):
    """
    For each gene in the hmm_df, calls the relevant contig_df (from the dictionary) and updates
    the information in this df. Updates the dictionary with the updated contig_df.
    :param contig_dict: (contig -> contig's df) dictionary
    :return: dictionary with the new data
    """
    for i, row in hmm_df.iterrows():
        contig_id = row['contig_id']
        contig_df = contig_dict.pop(contig_id)
        contig_df = contig_df.apply(check_if_closer, args=(row, gene_window, nucleotide_window), axis=1)
        contig_dict[contig_id] = contig_df
    return contig_dict


def calculate_nucleotide_distance(gene_to_update, resistant_gene):
    return abs(max(int(gene_to_update.start_index) - int(resistant_gene.end_index),
               int(resistant_gene.start_index) - int(gene_to_update.end_index)))


def check_if_in_window(gene_to_update, hmm_gene, gene_distance_window, nucleotide_window):
    """
    checks if the distance between the genes (both gene_distance and nucleotide_distance) are in the window range.
    If so, will update the gene_to_update relevant values.
    :param gene_distance_window: int, given by the user
    :param nucleotide_window: int, given by the user
    """
    gene_distance = abs(int(hmm_gene.ID.split("_")[-1]) - int(gene_to_update.ID.split("_")[-1]))
    if gene_distance <= gene_distance_window:
        gene_to_update.genes_in_distance_window += 1
        gene_to_update = check_closer_by_distance(gene_to_update, hmm_gene, gene_distance)

    nucleotide_distance = calculate_nucleotide_distance(gene_to_update, hmm_gene)
    if nucleotide_distance <= nucleotide_window:
        gene_to_update.genes_in_nucleotide_window += 1
        gene_to_update = check_if_closer_by_nucleotide(gene_to_update, hmm_gene, nucleotide_distance)
    return gene_to_update


def check_if_closer(gene_to_update, hmm_gene, distance_window, nucleotide_window):
    """
    Checks if this resistant_gene is the closer to the gene_to_update than the current information that it contains.
    If yes, will update this resistant_gene as the closest gene.
    :param gene_to_update: the gene from the contig_df, that we want to update
    :param hmm_gene: the current gene from the hmm_df
    :return: the updated gene_to_update (series)
    """
    if gene_to_update.ID == hmm_gene.ID:  # current gene passed the threshold, so we update is according to hmm_gene
        return update_resistant_gene(gene_to_update, hmm_gene)
    else:
        gene_to_update.contig_total_resistance_genes += 1
        gene_to_update = check_if_in_window(gene_to_update, hmm_gene, distance_window, nucleotide_window)
    return gene_to_update


def update_resistant_gene(gene_to_update, gene_from_hmm):
    gene_to_update.passed_threshold = True
    gene_to_update.query_name = gene_from_hmm.query_name
    gene_to_update.query_accession = gene_from_hmm.query_accession
    gene_to_update.best_eval = gene_from_hmm.best_eval
    gene_to_update.best_score = gene_from_hmm.best_score
    gene_to_update.hit_threshold_count = gene_from_hmm.hit_threshold_count
    return gene_to_update


def update_distance(gene_to_update, resistant_gene, distance):
    gene_to_update.gene_distance = distance
    gene_to_update.closest_gene_by_distance = resistant_gene.ID
    gene_to_update.hmm_accession1 = resistant_gene.query_accession
    gene_to_update.hmm_query_name1 = resistant_gene.query_name
    gene_to_update.eVal1 = resistant_gene.best_eval
    gene_to_update.score1 = resistant_gene.best_score
    gene_to_update.cls_gene_description = resistant_gene.description
    return gene_to_update


def check_closer_by_distance(gene_to_update, resistant_gene, new_distance):
    curr_distance = gene_to_update.gene_distance
    if (pd.notnull(curr_distance) and curr_distance > new_distance) or pd.isnull(curr_distance):
        gene_to_update = update_distance(gene_to_update, resistant_gene, new_distance)
    return gene_to_update


def update_nucleotide(gene_to_update, resistant_gene, nucleotide_distance):
    gene_to_update.nucleotide_distance = nucleotide_distance
    gene_to_update.closest_gene_by_nucleotide = resistant_gene.ID
    gene_to_update.hmm_accession2 = resistant_gene.query_accession
    gene_to_update.hmm_query_name2 = resistant_gene.query_name
    gene_to_update.eVal2 = resistant_gene.best_eval
    gene_to_update.score2 = resistant_gene.best_score
    gene_to_update.cls_nuc_description = resistant_gene.description
    return gene_to_update


def check_if_closer_by_nucleotide(gene_to_update, resistant_gene, new_nucleotide_distance):
    curr_nucleotide_distance = gene_to_update.nucleotide_distance
    if (pd.notnull(curr_nucleotide_distance) and curr_nucleotide_distance > new_nucleotide_distance) or pd.isnull(curr_nucleotide_distance):
        gene_to_update = update_nucleotide(gene_to_update, resistant_gene, new_nucleotide_distance)
    return gene_to_update


def rename_df_by_hmm_threshold(curr_df, hmm_db, threshold, distance_window, nucleotide_window):
    columns_dict = {'genes_in_nucleotide_window': f'genes_in_nucleotide_window_{hmm_db}_{threshold}_nucleotide_window={nucleotide_window}',
                    'genes_in_distance_window': f'genes_in_orf_window_{hmm_db}_{threshold}_orf_window={distance_window}',
                    'nucleotide_distance': f'nucleotide_distance_{hmm_db}_{threshold}_nucleotide_window={nucleotide_window}',
                    'gene_distance': f'gene_distance_{hmm_db}_{threshold}_gene_window={distance_window}'}
    columns_dict.update({name: f'{name}_{hmm_db}_{threshold}' for name in COLS})
    curr_df.rename(columns=columns_dict, inplace=True)
    return curr_df


def create_proximity_dfs_list(gff_df, df, hmm_file_name, hmm_db_name, threshold_list, by_eval, distance_window, nucleotide_window):
    frames = []
    for threshold in threshold_list:
        try:
            hmm_df = create_hmm_df(gff_df, hmm_file_name, threshold, by_eval)
            contig_dict = create_contig_dict(df)  # contig_name: df of all genes in contig
            contig_dict = fill_fields_in_contig_dict_by_distance(contig_dict, hmm_df, distance_window, nucleotide_window)
            curr_df = pd.concat(contig_dict.values()).drop(['contig_id', "start_index", 'end_index'], axis=1)
        except KeyError:
            # create an empty df with same columns
            curr_df = pd.DataFrame(columns=df.columns).drop(['contig_id', "start_index", 'end_index'], axis=1)
        curr_df = rename_df_by_hmm_threshold(curr_df, hmm_db_name, threshold, distance_window, nucleotide_window)
        curr_df.set_index('ID', inplace=True)
        frames.append(curr_df)
    return frames


def analyze_dfs_list(dfs_list, threshold_list):
    """
    handle the different cases possible as a result of running a list of thresholds:
        if none of the genes passed the threshold, for all the thresholds, will return an empty DF.
        if genes has passed only one threshold, will return it.
        otherwise, will merge all the DFs created for each threshold
    :param dfs_list: a list of the DFs created, a DF for each threshold that has genes that passed it
    :return: data-frame
    """
    output_df = reduce(lambda x, y: x.join(y, on='ID', lsuffix='_x', sort=False, how='outer'), dfs_list) #, rsuffix='_y'
    try:
        # output_df.drop(['description_y'], axis=1, inplace=True)
        newname = {'description': [f'description {name}' for name in threshold_list]}
        # rename(columns=lambda c: d[c].pop(0) if c in d.keys() else c)
        output_df.rename(columns={'description_x': 'description'}, inplace=True)
        output_df.rename(columns=lambda col: newname[col].pop(0) if col in newname.keys() else col, inplace=True)

    except KeyError:
        pass
    return output_df


def change_missing_values_and_out_of_window(df, distance_window, nucleotide_window):
    '''
    takes the final data frame and changes all the distances that are larger than the window to twise the window size.
    same with Nones = window*2
    '''
    # change the proximity data and fill the Nones
    orf_window_MISSING = distance_window * 2
    nucleotide_distance_missing = nucleotide_window * 2
    df['gene_distance-AMR-1e-20'].fillna(orf_window_MISSING, inplace=True)
    df.loc[df['gene_distance-AMR-1e-20'] > distance_window, 'gene_distance-AMR-1e-20'] = orf_window_MISSING
    df['nucleotide_distance-AMR-1e-20'].fillna(nucleotide_distance_missing, inplace=True)
    df.loc[df['nucleotide_distance-AMR-1e-20'] > nucleotide_window, 'nucleotide_distance-AMR-1e-20'] = nucleotide_distance_missing
    return df


def calculate_gene_proximity(gff_file_name, hmm_file_name, hmm_db_name, threshold_list, by_eval, distance_window, nucleotide_window):
    """
    calculates the proximity of a gene to all close hmm-passed results
    :param hmm_db_name: the name of the DB used to create tblout
    :param gff_file_name: str, *.gff
    :param hmm_file_name: str, tblout file - *.tblout
    :param threshold_list: a list of str/number (both work), the threshold for the tblout file.
    Determines which gene will be count as resistant. Can be either eVal or Score.
    :param by_eval: Boolean, Determines if the threshold is by eVal or score
    :param distance_window: int, counts
    :param nucleotide_window:
    :return: output dataframe.
    """
    gff_df = create_df_from_gff(gff_file_name)
    df = expand_gff_cols(gff_df)
    dfs_list = create_proximity_dfs_list(gff_df, df, hmm_file_name, hmm_db_name, threshold_list, by_eval, distance_window, nucleotide_window)
    final_df = analyze_dfs_list(dfs_list, threshold_list)
    return final_df


@feature_to_file('Gene_Proximity')
def process_to_file(gff_file_name, hmm_file_name, hmm_db_name, threshold_list, by_eval, distance_window,
                    nucleotide_window):
    return calculate_gene_proximity(gff_file_name, hmm_file_name, hmm_db_name, threshold_list, by_eval, distance_window, nucleotide_window)



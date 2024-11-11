import pickle
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import parse_arg_name, parse_query_name, create_fasta_from_df
from candidate_processing.get_taxonomy import get_tax_hierarchy, fill_missing_tax_level, get_taxonomy_by_id
import seaborn as sns
from collections import defaultdict
from Bio import SearchIO
from feature_extraction.create_tblout_file import get_tblout_file
from pathlib import Path


TMP_FASTA = os.path.join(os.getcwd(), 'neg_candidates_seqs.fasta')
DESC_FILE = os.path.join(Path(__name__).parent.absolute(), 'data', 'arg_mech_drugs.tsv')
HMM_PATH = os.path.join(Path(__name__).parent.absolute(), 'data', "Pfam-A.hmm")
DRUG_MAPPING = {'beta_lactam': 'Beta Lactam', 'kasugamycin': 'Aminoglycoside', 'thiostrepton': 'Peptide', 'chloramphenicol': 'Phenicol', 'bacitracin': 'Peptide', 'polymyxin': 'Peptide', 'glycopeptide': 'Peptide'}
with open(os.path.join(Path(__name__).parent.absolute(), 'data', 'id_to_env.json'), 'r') as fin:
    ID_TO_ENVIRONMENT = json.load(fin)


def parse_domtblout(file_path):
    attribs = ['hit_id', 'query_id', 'evalue', 'bitscore', 'env_start', 'env_end']
    hits = defaultdict(list)
    for queryresult in SearchIO.parse(file_path, 'hmmsearch3-domtab'):
        for hit in queryresult.hsps:
            hits['accession'].append(queryresult.accession)
            for attrib in attribs:
                hits[attrib].append(getattr(hit, attrib))
    return pd.DataFrame.from_dict(hits).rename(
        columns={'hit_id': 'ID', 'query_id': 'domain_ID', 'accession': 'domain_Accession', 'evalue': 'e-value'})


def check_ovelap(l_new, u_new, l, u):
    return (l_new <= l <= u_new) or (l <= l_new <= u)


def get_domains_to_drop(group):
    # removing domains overlapping with previous, better hits
    to_drop = []
    intervals = []
    for index, row in group.iterrows():
        if any([check_ovelap(row.env_start, row.env_end, l, u) for (l, u) in intervals]):
            to_drop.append(index)
        else:
            intervals.append((row.env_start, row.env_end))
    return to_drop


def filter_domains(df, e_value):
    df = df[df['e-value'] <= e_value]
    df = df.sort_values('bitscore', ascending=False)
    to_drop = []
    gb = df.groupby('domain_ID')
    for name, group in gb:
        to_drop += get_domains_to_drop(group)
    return df.drop(to_drop)


def get_drug_and_mech(df):
    desc = pd.read_csv(DESC_FILE, sep='\t')[['ARG Name', 'Drug Class', 'Updated Resistance Mechanism']].rename(columns={'Updated Resistance Mechanism': 'Resistance Mechanism'})
    desc['ARG Name'] = desc['ARG Name'].apply(parse_arg_name)
    desc = desc.set_index('ARG Name').to_dict()
    df['query_name_fixed'] = df['query_name'].apply(parse_query_name)
    for col in desc.keys():
        df[col] = df['query_name_fixed'].map(desc[col])
        df[col] = df[col].fillna(' Newly Predicted') # adding category to all negative proteins
    df['Drug Class'] = df['Drug Class'].apply(lambda x: DRUG_MAPPING[x] if x in DRUG_MAPPING else x.title())
    df['Drug Class'] = df['Drug Class'].apply(lambda x: x if pd.isnull(x) else x.split(";"))
    df.drop('query_name_fixed', axis=1)


def get_drugs_to_unite(df):
    drug_sizes = df.explode('Drug Class')['Drug Class'].value_counts(normalize=True)
    drugs_to_unite = [d for d in list(drug_sizes[drug_sizes < 0.01].index) if d != ' Newly Predicted']
    return drugs_to_unite


def get_rel_tax(df, tax_level):
    df = get_tax_hierarchy(df, filter_euk=True)
    df[f"tax_level_{tax_level}"] = df.apply(lambda row: fill_missing_tax_level(row, tax_level), axis=1)
    df[f"tax_level_{tax_level}"] = df[f"tax_level_{tax_level}"].apply(lambda x: x if pd.isnull(x) else x.title().replace("Tack", "TACK").replace("Fcb", "FCB").replace("Pvc", "PVC").replace(" Group", ""))
    return df


def get_environment(prot_id):
    envs = [np.nan] + [ID_TO_ENVIRONMENT[s].replace("human/animal", 'Unclassified / Other').title().replace("_", " ").replace(" / ", "/") for s in ID_TO_ENVIRONMENT.keys() if s in prot_id]
    return envs[-1]


def prepare_genomic_df(gdf, tax_level):
    # creation of series for all negative (non-ARG) proteins from genomes
    all_gdf = gdf['tax_id'].value_counts(normalize=True).reset_index().rename(columns={'tax_id':'prec',"index":'tax_id'}).set_index('tax_id')
    all_gdf = get_rel_tax(all_gdf, tax_level)

    # using more general lineage for taxonomic groups with low frequency
    tax_to_unite = []
    if tax_level > 1:
        tax_sizes = all_gdf.groupby(f"tax_level_{tax_level}")['prec'].sum()
        tax_to_unite = list(tax_sizes[tax_sizes < 0.005].index)
        all_gdf[f"tax_level_{tax_level}"] = all_gdf.apply(lambda r: r[f"tax_level_{tax_level-1}"] if r[f"tax_level_{tax_level}"] in tax_to_unite else r[f"tax_level_{tax_level}"], axis=1)

    # uniting taxonomic groups of low frequency to "Other" category
    tax_sizes = all_gdf.groupby(f"tax_level_{tax_level}")['prec'].sum()
    tax_to_other = list(tax_sizes[tax_sizes < 0.005].index)
    all_gdf[f"tax_level_{tax_level}"] = all_gdf.apply(lambda r: "Other Taxonomy Groups" if r[f"tax_level_{tax_level}"] in tax_to_other else r[f"tax_level_{tax_level}"], axis=1)

    all_gdf = all_gdf.groupby(f"tax_level_{tax_level}")['prec'].sum().reset_index()
    all_gdf = all_gdf.sort_values('prec', ascending=False).set_index(f"tax_level_{tax_level}")
    return all_gdf, tax_to_unite, tax_to_other


def prepare_metagenomic_df(mgdf):
    mgdf['Environment'] = mgdf['ID'].apply(get_environment)
    all_mgdf = mgdf['Environment'].value_counts(normalize=True).reset_index().rename(columns={'Environment':'prec',"index":'Environment'}).set_index('Environment')
    return all_mgdf.sort_values('prec', ascending=False)


def split_gdf_mgdf(df, tax_level):
    # Getting Tax for genomic samples
    df['tax_id'] = df['ID'].apply(get_taxonomy_by_id)
    gdf, mgdf = df[pd.notnull(df['tax_id'])], df[pd.isnull(df['tax_id'])]
    gdf = get_rel_tax(gdf, tax_level)
    return gdf, mgdf


def get_candidate_dfs(df, tax_level, tax_to_unite, tax_to_other):
    gdf, mgdf = split_gdf_mgdf(df, tax_level)
    gdf[f"tax_level_{tax_level}"] = gdf.apply(lambda r: r[f"tax_level_{tax_level - 1}"] if r[f"tax_level_{tax_level}"] in tax_to_unite else r[f"tax_level_{tax_level}"], axis=1)
    gdf[f"tax_level_{tax_level}"] = gdf.apply(lambda r: "Other Taxonomy Groups" if r[f"tax_level_{tax_level}"] in tax_to_other else r[f"tax_level_{tax_level}"], axis=1)

    # Getting env for metagenomic samples
    mgdf['Environment'] = mgdf['ID'].apply(get_environment)
    return gdf, mgdf


def create_bar_plots(df, col_to_plot, groups_col, tax_level, to_unite=()):
    tmp_df = df
    if col_to_plot == 'Drug Class':
        tmp_df = df.explode('Drug Class')
        tmp_df.loc[tmp_df[tmp_df['Drug Class'].isin(to_unite)].index, 'Drug Class'] = 'Other Antibiotics'

    # Removing positive proteins without known value to plot
    tmp_df = tmp_df[(pd.isnull(tmp_df['query_name'])) | (tmp_df[col_to_plot] != ' Newly Predicted')]

    groups_col_clean = groups_col.replace(f'_level_{tax_level}', 'onomy group').title()
    total_counts = tmp_df.groupby(groups_col)[col_to_plot].count()
    a = tmp_df.groupby(groups_col)[col_to_plot].value_counts(normalize=True).rename('prec').reset_index(level=1)
    b = a.pivot_table(values='prec', index=a.index, columns=col_to_plot, aggfunc='first').fillna(0.0)
    b = b.rename(columns={col: col.strip().title().replace(" To ", " to ") for col in b.columns})
    ax = b.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='Spectral', fontsize=12)

    xs = sorted(list(set([p.get_x() for p in ax.patches])))
    height = max([p.get_y() + p.get_height() for p in ax.patches])
    width = ax.patches[0].get_width()
    for i in range(len(total_counts)):
        ax.annotate(f'{total_counts.iloc[i]}', (xs[i] + (width / 2), height+0.01), ha='center', fontsize=9.5)

    plt.title(f"{col_to_plot} Distribution Per {groups_col_clean}", fontsize=15)
    plt.xlabel(groups_col_clean, fontsize=13)
    plt.xticks(rotation=75)  # 0
    plt.legend(loc=(1.04, 0), fontsize='x-large')
    plt.tight_layout()
    plt.savefig(f"{groups_col}_{col_to_plot}_positives_bars.pdf", bbox_inches='tight')
    plt.close()


def create_pie_plots(sizes, col, file_path, min_percentage=0.005, colors='YlGn'):
    fig, ax = plt.subplots(figsize=(7, 7))
    # uniting small categories to "Other"
    other_cat = "Other Environments" if col == 'Environment' else 'Other Taxonomy Groups'
    other_cat_size = sum([v for v in sizes.values() if v < min_percentage])
    sizes = {k:v for k,v in sizes.items() if v >= min_percentage or k == other_cat}
    sizes[other_cat] = other_cat_size + sizes.get(other_cat, 0.0)
    sizes = pd.Series(sizes).sort_values(ascending=False)

    labels = [f"{k} - {round(v*100, 2)}%" for k, v in sizes.to_dict().items()]
    patches, texts = ax.pie(sizes, colors=sns.color_palette(colors, len(sizes))[:], shadow=False, startangle=90)  # labels=classes, autopct='%1.1f%%',
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.1, 1.), fontsize=10)
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


def create_pie_plots_for_neg(df, col, min_percentage=0.005, colors='YlGn'):
    neg = df[pd.isnull(df['query_name'])]
    sizes = neg[col].value_counts(normalize=True).to_dict()
    create_pie_plots(sizes, col, f"{col}_negative_pie.pdf", min_percentage, colors)


def plot_enrichment(df, col, colors='rocket'):
    new_col = 'Taxonomy Group' if 'tax_level_' in col else col
    df = df.rename(columns={col: col+'_prec'})
    ax = sns.barplot(data=df.reset_index(), x=col, y='enrichment', palette=colors) # 'index
    ax.axhline(0, lw=2, color='black')
    plt.xlabel(new_col)
    plt.ylabel('High Model Score Enrichment')
    plt.title(f'Enrichment of High Model Score Proteins Across {new_col}s')
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.savefig(f'{new_col}_enrichments.pdf', bbox_inches='tight')
    plt.close()


def calculate_enrichment(df, gdf, all_gdf, mgdf, all_mgdf, tax_level):
    # uniting negative stats
    united_g = pd.DataFrame(gdf[pd.isnull(df['query_name'])][f"tax_level_{tax_level}"].value_counts(normalize=True)).join(all_gdf, how='right').fillna(0.0)
    united_mg = pd.DataFrame(mgdf[pd.isnull(df['query_name'])]['Environment'].value_counts(normalize=True)).join(all_mgdf, how='right').fillna(0.0)

    united_g['enrichment'] = united_g.apply(lambda r: np.log2(r[f'tax_level_{tax_level}']/r['prec']), axis=1)
    united_mg['enrichment'] = united_mg.apply(lambda r: np.log2(r['Environment'] / r['prec']), axis=1)

    plot_enrichment(united_g, f"tax_level_{tax_level}")
    plot_enrichment(united_mg, 'Environment', 'viridis')


def get_domains_df(hmmer_path, neg, fasta, fasta_suffix, e_value, n_cpu):
    create_fasta_from_df(neg, fasta, output_location=TMP_FASTA, suffix=fasta_suffix)
    tblout_file = fasta.replace(".fasta", "_pfam_res.tblout")
    get_tblout_file(hmmer_path, HMM_PATH, fasta, is_domain=True, tblout_path=tblout_file, retry=3, cpu=n_cpu)
    dom_df = parse_domtblout(tblout_file)
    dom_df = filter_domains(dom_df, e_value).set_index('ID')
    dom_df = dom_df.join(neg)
    return dom_df


def plot_top_domains(dom_df, top_doms):
    # plot most common domains
    dom_df['domain_ID'].value_counts().iloc[:top_doms].plot(kind='bar', color='#9A7CBE')
    plt.xlabel('Domain Name')
    plt.ylabel('Count')
    plt.title(f'Top {top_doms} Most Common Domains')
    plt.savefig(f'top_{top_doms}_most_common_domains.pdf', bbox_inches='tight')
    plt.close()


def plot_domains_frequency(dom_df, neg, top_doms):
    total_prots = dom_df.reset_index()['ID'].nunique()
    (dom_df.reset_index().groupby('domain_ID')['ID'].nunique() / total_prots).sort_values(ascending=False).iloc[:top_doms].plot(kind='bar', color='#9A7CBE')
    plt.xlabel('Domain Name')
    plt.ylabel('Percentage of Proteins')
    plt.title(f'Top {top_doms} Most Common Domains')
    plt.savefig(f'top_{top_doms}_most_common_domains_percentage.pdf', bbox_inches='tight')
    plt.close()
    # Get percentage of genes with no known domains
    missing_domain = len(total_prots) / len(neg)
    print(f"Percentage of genes with proteins missing a know domain: {round(missing_domain, 2)}")


def analyze_candidates(candidates, all_prots, fasta, fasta_suffix, hmmer_path, e_value=1e-6, tax_level=5, top_doms=20, n_cpu=3):
    candidates = candidates if 'ID' in candidates.columns else candidates.reset_index()
    get_drug_and_mech(candidates)
    drugs_to_unite = get_drugs_to_unite(candidates)

    all_gdf, all_mgdf = split_gdf_mgdf(all_prots[pd.isnull(all_prots['query_name'])], tax_level)
    all_gdf, tax_to_unite, tax_to_other = prepare_genomic_df(all_prots, tax_level)
    all_mgdf = prepare_metagenomic_df(all_mgdf)
    gdf, mgdf = get_candidate_dfs(candidates, tax_level, tax_to_unite, tax_to_other)

    if fasta and hmmer_path:  # Getting domains for negative candidates
        neg = candidates[pd.isnull(candidates['query_name'])].set_index('ID')
        dom_df = get_domains_df(hmmer_path, neg, fasta, fasta_suffix, e_value, n_cpu)
        plot_top_domains(dom_df, top_doms)
        plot_domains_frequency(dom_df, neg, top_doms)
        dom_df = dom_df.sort_values('prob', ascending=False)
    else:
        dom_df = pd.DataFrame([])

    # create bar plots of distributions
    for col_to_plot in ['Drug Class', 'Resistance Mechanism']:
        create_bar_plots(gdf, col_to_plot, f"tax_level_{tax_level}", tax_level, drugs_to_unite)
        create_bar_plots(mgdf, col_to_plot, 'Environment', tax_level, drugs_to_unite)

    # create pie plots for negative candidates
    create_pie_plots_for_neg(gdf, f"tax_level_{tax_level}", colors='YlOrBr')
    create_pie_plots_for_neg(mgdf, 'Environment')
    # All negatives
    create_pie_plots(all_gdf['prec'].to_dict(), f"tax_level_{tax_level}", f"taxonomy_all_negative_prots_pie.pdf", min_percentage=0.005, colors='YlOrBr')
    create_pie_plots(all_mgdf['prec'].to_dict(), 'Environment', f"environment_all_negative_prots_pie.pdf", min_percentage=0.005)

    calculate_enrichment(candidates, gdf, all_gdf, mgdf, all_mgdf, tax_level)
    return dom_df, gdf, mgdf


def parse_bool_arg(arg):
    return arg in ["True", "true", "T", "t"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analysis of positive and negative candidates according to taxonomy, environment and domains.')
    parser.add_argument("-cd", "--candidates", type=str, help="Path to  pickle file with candidates, note: positives should have query_name value")
    parser.add_argument("-ap", "--all_prots", type=str, help="Path to pickle file with all proteins (both high scoring and low scoring)")
    parser.add_argument("-o", "--output_path", type=str, help="Path to output pickle with the domains information.")
    parser.add_argument("-f", "--fasta", type=str, help="Path to fasta file or directory of fastas with the candidate proteins. If not supplied, no domain search is done. default: '", default='')
    parser.add_argument("-s", "--fasta_suffix", type=str, help="Suffix to the relevant protein fasta files. Only used if --fasta is a directory. default: '.proteins.faa'", default='.proteins.faa')
    parser.add_argument('--hmmer_path', type=str, help="full path to the HMMER's hmmsearch program. default: '' (no domain search is done)", default='')
    parser.add_argument("-e", "--e_value", type=float, default=1e-6, help="e-value threshold for domain HMM search. default: 1e-6")
    parser.add_argument("-tl", "--tax_level", type=int, default=5, help="Taxonomy hierarchy level to use. default: 5")
    parser.add_argument("-td", "--top_doms", type=int, default=20, help="how many top domains to show in figure. default: 20")
    parser.add_argument("-cpu", "--n_cpus", type=int, default=3, help="how many cpus to use for domain hmm search against Pfam. default: 3")
    args = parser.parse_args()
    args.drop_duplicates = parse_bool_arg(args.drop_duplicates)

    cdf = pd.read_pickle(args.candidates)
    ap_df = pd.read_pickle(args.all_prots)
    out_df, gdf, mgdf = analyze_candidates(cdf, ap_df, hmmer_path=args.hmmer_path, e_value=args.e_value, fasta=args.fasta, fasta_suffix=args.fasta_suffix, tax_level=args.tax_level, top_doms=args.top_doms, n_cpu=args.n_cpus)
    if len(out_df) > 0:
        out_df.to_pickle(args.output_path)
    gdf.to_pickle(args.output_path.replace('.pkl', '_genomes.pkl'))
    mgdf.to_pickle(args.output_path.replace('.pkl', '_metagenomes.pkl'))


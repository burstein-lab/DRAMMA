import pandas as pd
import numpy as np
import json
import os
import argparse
import subprocess
from sklearn.utils import shuffle
from resfam_lists import PUMPS_FAM_HIGH_SEC_LIST, ALL_PUMPS_FAM_LIST, HIGH_SEC_NO_PUMPS_LIST
from utilities import getIDs, combine_all_pkls, create_fasta_from_df


NEGATIVE_RATIO = 10
COLUMNS = ['query_name', 'query_accession', 'best_score', 'passed threshold', 'best_eval_exponent', 'Contig', 'GC_diff', 'contigGC', 'contigsize', 'geneGC', 'genesize', 'SmartGC_diff_from_AVG', 'molecular_weight_pp', 'aromaticity_pp', 'gravy_pp', 'instability_index_pp', 'isoelectric_point_pp', 'Helix_fraction_pp', 'Turn_fraction_pp', 'Sheet_fraction_pp', 'molar_extinction_coefficient_reduced_pp', 'molar_extinction_coefficient_cystines_pp', 'BBBBBBBB', 'BBBBBBBL', 'BBBBBBLB', 'BBBBBBLL', 'BBBBBLBB', 'BBBBBLBL', 'BBBBBLLB', 'BBBBBLLL', 'BBBBLBBB', 'BBBBLBBL', 'BBBBLBLB', 'BBBBLBLL', 'BBBBLLBB', 'BBBBLLBL', 'BBBBLLLB', 'BBBBLLLL', 'BBBLBBBB', 'BBBLBBBL', 'BBBLBBLB', 'BBBLBBLL', 'BBBLBLBB', 'BBBLBLBL', 'BBBLBLLB', 'BBBLBLLL', 'BBBLLBBB', 'BBBLLBBL', 'BBBLLBLB', 'BBBLLBLL', 'BBBLLLBB', 'BBBLLLBL', 'BBBLLLLB', 'BBBLLLLL', 'BBLBBBBB', 'BBLBBBBL', 'BBLBBBLB', 'BBLBBBLL', 'BBLBBLBB', 'BBLBBLBL', 'BBLBBLLB', 'BBLBBLLL', 'BBLBLBBB', 'BBLBLBBL', 'BBLBLBLB', 'BBLBLBLL', 'BBLBLLBB', 'BBLBLLBL', 'BBLBLLLB', 'BBLBLLLL', 'BBLLBBBB', 'BBLLBBBL', 'BBLLBBLB', 'BBLLBBLL', 'BBLLBLBB', 'BBLLBLBL', 'BBLLBLLB', 'BBLLBLLL', 'BBLLLBBB', 'BBLLLBBL', 'BBLLLBLB', 'BBLLLBLL', 'BBLLLLBB', 'BBLLLLBL', 'BBLLLLLB', 'BBLLLLLL', 'BLBBBBBB', 'BLBBBBBL', 'BLBBBBLB', 'BLBBBBLL', 'BLBBBLBB', 'BLBBBLBL', 'BLBBBLLB', 'BLBBBLLL', 'BLBBLBBB', 'BLBBLBBL', 'BLBBLBLB', 'BLBBLBLL', 'BLBBLLBB', 'BLBBLLBL', 'BLBBLLLB', 'BLBBLLLL', 'BLBLBBBB', 'BLBLBBBL', 'BLBLBBLB', 'BLBLBBLL', 'BLBLBLBB', 'BLBLBLBL', 'BLBLBLLB', 'BLBLBLLL', 'BLBLLBBB', 'BLBLLBBL', 'BLBLLBLB', 'BLBLLBLL', 'BLBLLLBB', 'BLBLLLBL', 'BLBLLLLB', 'BLBLLLLL', 'BLLBBBBB', 'BLLBBBBL', 'BLLBBBLB', 'BLLBBBLL', 'BLLBBLBB', 'BLLBBLBL', 'BLLBBLLB', 'BLLBBLLL', 'BLLBLBBB', 'BLLBLBBL', 'BLLBLBLB', 'BLLBLBLL', 'BLLBLLBB', 'BLLBLLBL', 'BLLBLLLB', 'BLLBLLLL', 'BLLLBBBB', 'BLLLBBBL', 'BLLLBBLB', 'BLLLBBLL', 'BLLLBLBB', 'BLLLBLBL', 'BLLLBLLB', 'BLLLBLLL', 'BLLLLBBB', 'BLLLLBBL', 'BLLLLBLB', 'BLLLLBLL', 'BLLLLLBB', 'BLLLLLBL', 'BLLLLLLB', 'BLLLLLLL', 'LBBBBBBB', 'LBBBBBBL', 'LBBBBBLB', 'LBBBBBLL', 'LBBBBLBB', 'LBBBBLBL', 'LBBBBLLB', 'LBBBBLLL', 'LBBBLBBB', 'LBBBLBBL', 'LBBBLBLB', 'LBBBLBLL', 'LBBBLLBB', 'LBBBLLBL', 'LBBBLLLB', 'LBBBLLLL', 'LBBLBBBB', 'LBBLBBBL', 'LBBLBBLB', 'LBBLBBLL', 'LBBLBLBB', 'LBBLBLBL', 'LBBLBLLB', 'LBBLBLLL', 'LBBLLBBB', 'LBBLLBBL', 'LBBLLBLB', 'LBBLLBLL', 'LBBLLLBB', 'LBBLLLBL', 'LBBLLLLB', 'LBBLLLLL', 'LBLBBBBB', 'LBLBBBBL', 'LBLBBBLB', 'LBLBBBLL', 'LBLBBLBB', 'LBLBBLBL', 'LBLBBLLB', 'LBLBBLLL', 'LBLBLBBB', 'LBLBLBBL', 'LBLBLBLB', 'LBLBLBLL', 'LBLBLLBB', 'LBLBLLBL', 'LBLBLLLB', 'LBLBLLLL', 'LBLLBBBB', 'LBLLBBBL', 'LBLLBBLB', 'LBLLBBLL', 'LBLLBLBB', 'LBLLBLBL', 'LBLLBLLB', 'LBLLBLLL', 'LBLLLBBB', 'LBLLLBBL', 'LBLLLBLB', 'LBLLLBLL', 'LBLLLLBB', 'LBLLLLBL', 'LBLLLLLB', 'LBLLLLLL', 'LLBBBBBB', 'LLBBBBBL', 'LLBBBBLB', 'LLBBBBLL', 'LLBBBLBB', 'LLBBBLBL', 'LLBBBLLB', 'LLBBBLLL', 'LLBBLBBB', 'LLBBLBBL', 'LLBBLBLB', 'LLBBLBLL', 'LLBBLLBB', 'LLBBLLBL', 'LLBBLLLB', 'LLBBLLLL', 'LLBLBBBB', 'LLBLBBBL', 'LLBLBBLB', 'LLBLBBLL', 'LLBLBLBB', 'LLBLBLBL', 'LLBLBLLB', 'LLBLBLLL', 'LLBLLBBB', 'LLBLLBBL', 'LLBLLBLB', 'LLBLLBLL', 'LLBLLLBB', 'LLBLLLBL', 'LLBLLLLB', 'LLBLLLLL', 'LLLBBBBB', 'LLLBBBBL', 'LLLBBBLB', 'LLLBBBLL', 'LLLBBLBB', 'LLLBBLBL', 'LLLBBLLB', 'LLLBBLLL', 'LLLBLBBB', 'LLLBLBBL', 'LLLBLBLB', 'LLLBLBLL', 'LLLBLLBB', 'LLLBLLBL', 'LLLBLLLB', 'LLLBLLLL', 'LLLLBBBB', 'LLLLBBBL', 'LLLLBBLB', 'LLLLBBLL', 'LLLLBLBB', 'LLLLBLBL', 'LLLLBLLB', 'LLLLBLLL', 'LLLLLBBB', 'LLLLLBBL', 'LLLLLBLB', 'LLLLLBLL', 'LLLLLLBB', 'LLLLLLBL', 'LLLLLLLB', 'LLLLLLLL', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'ANDN920101', 'ARGP820101', 'ARGP820102', 'BEGF750103', 'BIGC670101', 'BIOV880101', 'BULH740102', 'BUNA790101', 'BURA740101', 'CHAM810101', 'CHOC760102', 'CHOC760103', 'CHOP780202', 'CHOP780205', 'CHOP780213', 'CHOP780214', 'DAYM780101', 'FAUJ880107', 'FAUJ880111', 'FAUJ880112', 'FINA910102', 'GUYH850101', 'ISOY800102', 'ISOY800108', 'KLEP840101', 'LEVM760106', 'NAKH900102', 'OOBM770104', 'PRAM820102', 'PTIO830101', 'QIAN880106', 'RICJ880101', 'VELV850101', 'YUTK870101', 'YUTK870103', 'MUNV940101', 'MUNV940104', 'KUMS000103', 'GEOR030101', 'Aliphatic[AGILV]', 'Amide[NQ]', 'Aromatic[FWY]', 'Hydroxyl-Containing[ST]', 'Sulfur-Containing[CM]', 'Negative[DE]', 'Nonpolar[AGILVFWPCM]', 'Positive[KR]', 'Neutral[HAGILVFWPCMNQYST]', 'Polar[NQYST]', 'genes_in_orf_window_T6SS.immunity.proteins_1e-8_orf_window=5', 'genes_in_nucleotide_window_T6SS.immunity.proteins_1e-8_nucleotide_window=5000', 'nucleotide_distance_T6SS.immunity.proteins_1e-8_nucleotide_window=5000', 'gene_distance_T6SS.immunity.proteins_1e-8_gene_window=5', 'genes_in_orf_window_VirSorter_Phage_Clusters_current.hallmark.annot_1e-8_orf_window=5', 'genes_in_nucleotide_window_VirSorter_Phage_Clusters_current.hallmark.annot_1e-8_nucleotide_window=5000', 'nucleotide_distance_VirSorter_Phage_Clusters_current.hallmark.annot_1e-8_nucleotide_window=5000', 'gene_distance_VirSorter_Phage_Clusters_current.hallmark.annot_1e-8_gene_window=5', 'genes_in_orf_window_pfam_phage_plasmid_proteins_1e-8_orf_window=5', 'genes_in_nucleotide_window_pfam_phage_plasmid_proteins_1e-8_nucleotide_window=5000', 'nucleotide_distance_pfam_phage_plasmid_proteins_1e-8_nucleotide_window=5000', 'gene_distance_pfam_phage_plasmid_proteins_1e-8_gene_window=5', 'genes_in_orf_window_all_ARGs_filtered_1e-8_orf_window=5', 'genes_in_nucleotide_window_all_ARGs_filtered_1e-8_nucleotide_window=5000', 'nucleotide_distance_all_ARGs_filtered_1e-8_nucleotide_window=5000', 'gene_distance_all_ARGs_filtered_1e-8_gene_window=5', 'genes_in_orf_window_Acrs_HMM_1e-8_orf_window=5', 'genes_in_nucleotide_window_Acrs_HMM_1e-8_nucleotide_window=5000', 'nucleotide_distance_Acrs_HMM_1e-8_nucleotide_window=5000', 'gene_distance_Acrs_HMM_1e-8_gene_window=5', 'genes_in_orf_window_Antirestriction_HMM_merged_version4_1e-8_orf_window=5', 'genes_in_nucleotide_window_Antirestriction_HMM_merged_version4_1e-8_nucleotide_window=5000', 'nucleotide_distance_Antirestriction_HMM_merged_version4_1e-8_nucleotide_window=5000', 'gene_distance_Antirestriction_HMM_merged_version4_1e-8_gene_window=5', 'genes_in_orf_window_TnpPred.uniq_1e-8_orf_window=5', 'genes_in_nucleotide_window_TnpPred.uniq_1e-8_nucleotide_window=5000', 'nucleotide_distance_TnpPred.uniq_1e-8_nucleotide_window=5000', 'gene_distance_TnpPred.uniq_1e-8_gene_window=5', 'genes_in_orf_window_T6SS.immunity.proteins_1e-8_orf_window=10', 'genes_in_nucleotide_window_T6SS.immunity.proteins_1e-8_nucleotide_window=10000', 'nucleotide_distance_T6SS.immunity.proteins_1e-8_nucleotide_window=10000', 'gene_distance_T6SS.immunity.proteins_1e-8_gene_window=10', 'genes_in_orf_window_VirSorter_Phage_Clusters_current.hallmark.annot_1e-8_orf_window=10', 'genes_in_nucleotide_window_VirSorter_Phage_Clusters_current.hallmark.annot_1e-8_nucleotide_window=10000', 'nucleotide_distance_VirSorter_Phage_Clusters_current.hallmark.annot_1e-8_nucleotide_window=10000', 'gene_distance_VirSorter_Phage_Clusters_current.hallmark.annot_1e-8_gene_window=10', 'genes_in_orf_window_pfam_phage_plasmid_proteins_1e-8_orf_window=10', 'genes_in_nucleotide_window_pfam_phage_plasmid_proteins_1e-8_nucleotide_window=10000', 'nucleotide_distance_pfam_phage_plasmid_proteins_1e-8_nucleotide_window=10000', 'gene_distance_pfam_phage_plasmid_proteins_1e-8_gene_window=10', 'genes_in_orf_window_all_ARGs_filtered_1e-8_orf_window=10', 'genes_in_nucleotide_window_all_ARGs_filtered_1e-8_nucleotide_window=10000', 'nucleotide_distance_all_ARGs_filtered_1e-8_nucleotide_window=10000', 'gene_distance_all_ARGs_filtered_1e-8_gene_window=10', 'genes_in_orf_window_Acrs_HMM_1e-8_orf_window=10', 'genes_in_nucleotide_window_Acrs_HMM_1e-8_nucleotide_window=10000', 'nucleotide_distance_Acrs_HMM_1e-8_nucleotide_window=10000', 'gene_distance_Acrs_HMM_1e-8_gene_window=10', 'genes_in_orf_window_Antirestriction_HMM_merged_version4_1e-8_orf_window=10', 'genes_in_nucleotide_window_Antirestriction_HMM_merged_version4_1e-8_nucleotide_window=10000', 'nucleotide_distance_Antirestriction_HMM_merged_version4_1e-8_nucleotide_window=10000', 'gene_distance_Antirestriction_HMM_merged_version4_1e-8_gene_window=10', 'genes_in_orf_window_TnpPred.uniq_1e-8_orf_window=10', 'genes_in_nucleotide_window_TnpPred.uniq_1e-8_nucleotide_window=10000', 'nucleotide_distance_TnpPred.uniq_1e-8_nucleotide_window=10000', 'gene_distance_TnpPred.uniq_1e-8_gene_window=10', 'HTH_score_full_seq', 'HTH_rep', 'correlation_distance_2mers_nuc', 'cosine_distance_2mers_nuc', 'euclidean_distance_2mers_nuc', 'correlation_distance_3mers_nuc', 'cosine_distance_3mers_nuc', 'euclidean_distance_3mers_nuc', 'correlation_distance_4mers_nuc', 'cosine_distance_4mers_nuc', 'euclidean_distance_4mers_nuc', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Coprothermobacterota_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Coprothermobacterota_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Dictyoglomi_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Dictyoglomi_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Chrysiogenetes_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Chrysiogenetes_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Calditrichaeota_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Calditrichaeota_DB_reducted', '1e-06_e-value_percentage_Archaea_Small_Sub_Taxes_of_DPANN_group_DB_reducted', 'max_exponent_Archaea_Small_Sub_Taxes_of_DPANN_group_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Elusimicrobia_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Elusimicrobia_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Thermodesulfobacteria_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Thermodesulfobacteria_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Nitrospinae_Tectomicrobia_group_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Nitrospinae_Tectomicrobia_group_DB_reducted', '1e-06_e-value_percentage_Archaea_unclassified_Euryarchaeota_DB_reducted', 'max_exponent_Archaea_unclassified_Euryarchaeota_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Deferribacteres_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Deferribacteres_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Small_Sub_Taxes_of_Rhizaria_DB_reducted', 'max_exponent_Eukaryota_Small_Sub_Taxes_of_Rhizaria_DB_reducted', '1e-06_e-value_percentage_Bacteria_Aquificae_DB_reducted', 'max_exponent_Bacteria_Aquificae_DB_reducted', '1e-06_e-value_percentage_Bacteria_Synergistia_DB_reducted', 'max_exponent_Bacteria_Synergistia_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Small_Sub_Taxes_of_Heterolobosea_DB_reducted', 'max_exponent_Eukaryota_Small_Sub_Taxes_of_Heterolobosea_DB_reducted', '1e-06_e-value_percentage_Archaea_Small_Sub_Taxes_of_TACK_group_DB_reducted', 'max_exponent_Archaea_Small_Sub_Taxes_of_TACK_group_DB_reducted', '1e-06_e-value_percentage_Bacteria_Thermotogae_DB_reducted', 'max_exponent_Bacteria_Thermotogae_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Small_Sub_Taxes_of_Cryptophyta_DB_reducted', 'max_exponent_Eukaryota_Small_Sub_Taxes_of_Cryptophyta_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Small_Sub_Taxes_of_Haptista_DB_reducted', 'max_exponent_Eukaryota_Small_Sub_Taxes_of_Haptista_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Nitrospirae_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Nitrospirae_DB_reducted',  '1e-06_e-value_percentage_Bacteria_Fusobacteriia_DB_reducted', 'max_exponent_Bacteria_Fusobacteriia_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Proteobacteria_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Proteobacteria_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_FCB_group_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_FCB_group_DB_reducted', '1e-06_e-value_percentage_Archaea_Methanomada_group_DB_reducted', 'max_exponent_Archaea_Methanomada_group_DB_reducted', '1e-06_e-value_percentage_Bacteria_Tenericutes_DB_reducted', 'max_exponent_Bacteria_Tenericutes_DB_reducted', '1e-06_e-value_percentage_Archaea_Small_Sub_Taxes_of_Euryarchaeota_DB_reducted', 'max_exponent_Archaea_Small_Sub_Taxes_of_Euryarchaeota_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Small_Sub_Taxes_of_Metamonada_DB_reducted', 'max_exponent_Eukaryota_Small_Sub_Taxes_of_Metamonada_DB_reducted', '1e-06_e-value_percentage_Archaea_Crenarchaeota_DB_reducted', 'max_exponent_Archaea_Crenarchaeota_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Small_Sub_Taxes_of_Euglenozoa_DB_reducted', 'max_exponent_Eukaryota_Small_Sub_Taxes_of_Euglenozoa_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Acidobacteria_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Acidobacteria_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_PVC_group_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_PVC_group_DB_reducted', '1e-06_e-value_percentage_Bacteria_Small_Sub_Taxes_of_Terrabacteria_group_DB_reducted', 'max_exponent_Bacteria_Small_Sub_Taxes_of_Terrabacteria_group_DB_reducted', '1e-06_e-value_percentage_Bacteria_Chloroflexi_DB_reducted', 'max_exponent_Bacteria_Chloroflexi_DB_reducted', '1e-06_e-value_percentage_Bacteria_Verrucomicrobia_DB_reducted', 'max_exponent_Bacteria_Verrucomicrobia_DB_reducted', '1e-06_e-value_percentage_Bacteria_Spirochaetia_DB_reducted', 'max_exponent_Bacteria_Spirochaetia_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Evosea_DB_reducted', 'max_exponent_Eukaryota_Evosea_DB_reducted', '1e-06_e-value_percentage_Viruses_DB_reducted', 'max_exponent_Viruses_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Chlorophyta_DB_reducted', 'max_exponent_Eukaryota_Chlorophyta_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Small_Sub_Taxes_of_Alveolata_DB_reducted', 'max_exponent_Eukaryota_Small_Sub_Taxes_of_Alveolata_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Small_Sub_Taxes_of_Stramenopiles_DB_reducted', 'max_exponent_Eukaryota_Small_Sub_Taxes_of_Stramenopiles_DB_reducted', '1e-06_e-value_percentage_Archaea_Stenosarchaea_group_DB_reducted', 'max_exponent_Archaea_Stenosarchaea_group_DB_reducted', '1e-06_e-value_percentage_Bacteria_Cyanobacteria_Melainabacteria_group_DB_reducted', 'max_exponent_Bacteria_Cyanobacteria_Melainabacteria_group_DB_reducted', '1e-06_e-value_percentage_Bacteria_delta_epsilon_subdivisions_DB_reducted', 'max_exponent_Bacteria_delta_epsilon_subdivisions_DB_reducted', '1e-06_e-value_percentage_Bacteria_Betaproteobacteria_DB_reducted', 'max_exponent_Bacteria_Betaproteobacteria_DB_reducted', '1e-06_e-value_percentage_Bacteria_Gammaproteobacteria_DB_reducted', 'max_exponent_Bacteria_Gammaproteobacteria_DB_reducted', '1e-06_e-value_percentage_Bacteria_Bacteroidetes_Chlorobi_group_DB_reducted', 'max_exponent_Bacteria_Bacteroidetes_Chlorobi_group_DB_reducted', '1e-06_e-value_percentage_Bacteria_Alphaproteobacteria_DB_reducted', 'max_exponent_Bacteria_Alphaproteobacteria_DB_reducted', '1e-06_e-value_percentage_Bacteria_Firmicutes_DB_reducted', 'max_exponent_Bacteria_Firmicutes_DB_reducted', '1e-06_e-value_percentage_Bacteria_Actinobacteria_DB_reducted', 'max_exponent_Bacteria_Actinobacteria_DB_reducted', '1e-06_e-value_percentage_Eukaryota_Fungi_DB_reducted', 'max_exponent_Eukaryota_Fungi_DB_reducted', 'level_1_1e-06_percentage', 'level_2_1e-06_percentage', 'level_3_1e-06_percentage', '0.5_quantile_exponent', '0.75_quantile_exponent', '0.9_quantile_exponent']


def run_cdhit(df, protein_fasta, output_fasta='final_genes.fasta'):
    no_duplicates_fasta = create_fasta_from_df(df, protein_fasta)
    print('starting cd-hit')
    process_count = subprocess.run(
        [f'cd-hit -T 6 -M 0 -g 1 -s 0.8 -c 0.9 -d 1000 -i {no_duplicates_fasta} -o {output_fasta}'], capture_output=True,
        text=True, shell=True)
    print("finished cd-hit")
    id_list = list(getIDs(output_fasta)['ID'])
    print("finished getIDs")
    return df.loc[id_list]


def manage_bias_data(df, wanted_ratio, PUMPS):
    """
    Takes from the dataframe only wanted_ratio*(number of positive examples) of negative examples and returns it
    combines with the positive examples
    :param df: dataframe with True/false at the 'passed_threshold-AMR-1e-20' column
    :param wanted_ratio: how many negative do we want per 1 positive
    :return: a new dataframe with the positive and a portion of the negative set
    """
    only_positive = df[df.loc[:, 'passed threshold'] == True]
    only_negative = df[df.loc[:, 'passed threshold'] != True]
    resfam_column_name = 'query_name'
    if PUMPS:
        only_positive = only_positive[only_positive[resfam_column_name].isin(PUMPS_FAM_HIGH_SEC_LIST)]
    else:
        extra_negative = only_positive[only_positive[resfam_column_name].isin(ALL_PUMPS_FAM_LIST)]
        only_positive = only_positive[only_positive[resfam_column_name].isin(HIGH_SEC_NO_PUMPS_LIST)]
    size_of_positive = len(only_positive)
    print(f'Total len: {len(df)}')
    print('the len of the positive:' + str(size_of_positive))
    if not PUMPS and len(extra_negative) > 0:
        extra_negative = extra_negative.sample(n=min(size_of_positive, len(extra_negative)))  # sample size of extra_negative if it's smaller than size of positives
        extra_negative.loc[:, 'passed threshold'] = False
        extra_negative.loc[:, 'query_accession'] = None
        extra_negative.loc[:, resfam_column_name] = None
        only_positive = shuffle(pd.concat([extra_negative, only_positive]))
    sample_size = size_of_positive * wanted_ratio
    print('the sample size: ' + str(sample_size))
    try:
        unbiased_df = only_negative.sample(n=sample_size)
    except:
        unbiased_df = only_negative
    return shuffle(pd.concat([unbiased_df, only_positive]))  # not sure if the shuffle is necessary


def fix_labels(df, PUMPS):
    """
    Changes labels to fit Pump/ no pump according to PUMP value
    :param df: dataframe with True/false at the 'passed_threshold-AMR-1e-20' column
    :return: a new dataframe with the positive and a portion of the negative set
    """
    only_positive = df[df['passed threshold'] == True]
    resfam_column_name = 'query_name'
    if PUMPS:
        rows_to_change = only_positive[~only_positive[resfam_column_name].isin(PUMPS_FAM_HIGH_SEC_LIST)].index
    else:
        rows_to_change = only_positive[~only_positive[resfam_column_name].isin(HIGH_SEC_NO_PUMPS_LIST)].index

    df.loc[rows_to_change, 'passed threshold'] = False
    df.loc[rows_to_change, 'query_accession'] = None
    df.loc[rows_to_change, resfam_column_name] = None

    size_of_positive = len(only_positive)
    print('the len of the positive:' + str(size_of_positive))


def go_through_df(directory_path, whitelist):
    """
    runs through all the pickels in a directory and yields them one by one
    :param directory_path:
    :return: yields one dataframe at a time
    """
    with os.scandir(directory_path) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False) and whitelist in entry.name:
                yield combine_all_pkls(entry, pd.DataFrame([]).rename_axis('ID'))


def run_creator(dir, whitelist, PUMPS, columns, all_data=False, is_pickle=False, batch_size=0):
    """
    goes over the pickels one by one create the train dtataset with only a 10 fold negative/positive
    it is useful so we will not create the huge table with all the negative bias that we dont need right now
    :param dir: where all the pickles are
    :return: The new data frame.
    """
    is_first = True
    df_lst = []
    file_num = 1
    finale_df = pd.DataFrame()
    for df in go_through_df(dir, whitelist):
        # filtering out only non-empty Dataframes and those who did not fail in feature selection
        if len(df) > 0 and 'passed threshold' in df.columns and 'molecular_weight_pp' in df.columns:
            if all_data:
                fix_labels(df, PUMPS)
                missing_cols = [col for col in columns if col not in df.columns]
                df = df.reindex(columns=df.columns.tolist()+missing_cols, fill_value=np.nan)
                # appending dataframe to result file
                if batch_size == 0: # appending all results to one file
                    df[columns].to_csv(rf'{os.getcwd()}/ml_feature_table_complete_dataset.tsv.gz', sep='\t', mode='a', header=is_first, compression="gzip")
                    is_first = False
                else:
                    finale_df = df[columns] if is_first else pd.concat([finale_df, df[columns]])
                    is_first = False
                    if len(finale_df) >= batch_size: # time to write df to file
                        if is_pickle:
                            finale_df[columns].to_pickle(rf'{os.getcwd()}/ml_feature_table_complete_dataset_batch_{file_num}.pkl')
                        else:
                            finale_df[columns].to_csv(rf'{os.getcwd()}/ml_feature_table_complete_dataset_batch_{file_num}.tsv.gz', sep='\t', compression="gzip")
                        file_num += 1
                        is_first = True # next df will be the first one in batch
                        finale_df = pd.DataFrame()
            else:
                df = manage_bias_data(df, NEGATIVE_RATIO, PUMPS)
                missing_cols = [col for col in columns if col not in df.columns]
                df = df.reindex(columns=df.columns.tolist()+missing_cols, fill_value=np.nan)
                df_lst.append(df)

            # clean_df = prepare_for_learning(unbiased_df, throshold, hmmdb_name)
    if not all_data:
        finale_df = pd.concat(df_lst)
        return finale_df
    elif batch_size > 0:  # saving last batch
        if is_pickle:
            finale_df[columns].to_pickle(rf'{os.getcwd()}/ml_feature_table_complete_dataset_batch_{file_num}.pkl')
        else:
            finale_df[columns].to_csv(rf'{os.getcwd()}/ml_feature_table_complete_dataset_batch_{file_num}.tsv.gz', sep='\t', compression="gzip")


def main(args, columns):
    train_df = run_creator(args.directory, args.whitelist, args.pumps, columns, args.all_data, args.pickle, args.batch_size)
    print("##### finished run creator #####")
    if not args.all_data:
        if args.pickle:
            train_df.to_pickle(rf'{os.getcwd()}/pre_ml_feature_table.pkl')
        else:
            train_df.to_csv(rf'{os.getcwd()}/pre_ml_feature_table.tsv', sep='\t', float_format='%.3f')
        train_df = run_cdhit(train_df, args.fasta)
        filename = rf'{os.getcwd()}/ml_feature_table.pkl' if args.pickle else rf'{os.getcwd()}/ml_feature_table.tsv'
        if args.pickle:
            train_df.to_pickle(filename)
        else:
            train_df.to_csv(filename, sep='\t', float_format='%.3f')


def parse_boolean_arg(p):
    if p in ["True", "true", "T", "t"]:
        return True
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset out of featuretable, either all the proteins or a balanced subset')
    parser.add_argument("-wl", "--whitelist", type=str, default='',
                        help="how to filter which folders to check, default = '' (check all)")
    parser.add_argument("-d", "--directory", type=str, help="Directory to (only) all the pkls")
    parser.add_argument("-f", "--fasta", type=str, help="path to relevant protein fasta file to use for de-duplication. Only used when all_data is False")
    parser.add_argument("-p", "--pumps", type=str, default="False",
                        help="should we create the pump train set? keep empty for the no pumps option")
    parser.add_argument("-ad", "--all_data", type=str, default="False",
                        help="Should we create a balanced dataset (-ad False, default) or on the entire data")
    parser.add_argument("-pkl", "--pickle", type=str, default="True",
                        help="Should we save data to pickle (-pkl True, default) or save as tsv")
    parser.add_argument("-b", "--batch_size", type=int, default=0, help="batch size for saving dataset for all_data=True."
                                                                        " default: 0 (everything will be saved in one file)")
    parser.add_argument("-c", "--columns", help="JSON file with the columns we want in our dataset. if empty, "
                                                "all columns are used. default: ''", type=str, default='')
    args = parser.parse_args()
    args.pumps, args.all_data, args.pickle = parse_boolean_arg(args.pumps), parse_boolean_arg(args.all_data), parse_boolean_arg(args.pickle)
    if args.columns: # not empty
        with open(args.columns, "r") as j_in:
            columns = json.load(j_in)
            cols_to_add = [col for col in ['passed threshold', 'Contig', 'query_accession', 'query_name', 'best_eval_exponent'] if col not in columns]
            columns += cols_to_add
    else:
        columns = COLUMNS
    main(args, columns)

import time
import os
from utilities import getIDs, getData
from mmseqs_taxonomy_distribution import MMseqsTaxonomyFeatures
from labeling import Labeling
from multi_proximity_genes import MultiProximityFeatures
from HTH_domain import HTHDomainFeatures
from kmers_funcs import KMersFeatures
from amino_acid_features import AAFeatures
from GC_content import GCContentFeatures
from protparam_features import ProtParamsFeatures
from smartGC import SmartGCFeatures
from proteins_quality_kmers import SmartAAKmersFeatures
from cross_membrane import CrossMembraneFeatures

INDEX_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'AminoAcidIndexChoosenOnes')
SMART_GC_DICT = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'SmartGC_dict_simple')
ALL_FEATURES = ['Mmseqs', 'labeling', 'multi_proximity', 'default_multi_proximity', "HTH_domains", 'DNA_KMers', "AA_features", "GC_Content", 'Prot_param', 'SmartGC', 'Smart_AA_Kmers', 'Cross_Membrane']
DATA_PATH = os.path.join("..", 'data', 'feature_extraction')

class FeatureList:
    # IMPORTANT: keep labeling and multi_proximity at the beginning
    def __init__(self, hmmer_path, mmseqs_path, tmhmm_path, dna_kmer_size, by_eval, label_threshold, threshold_list, gene_window, nucleotide_window, features=()):
        tax_data_path = os.path.join(DATA_PATH, 'data_for_tax_features')
        hth_hmm_domains = os.path.join(DATA_PATH, 'hmms_for_proximity_features')
        hmm_dir = os.path.join(DATA_PATH, 'Pfam_HTH_domains.hmm')
        hmm_db = os.path.join(hmm_dir, 'DRAMMA_ARG_DB.hmm')
        feature_dict = {'labeling': Labeling(hmmer_path, hmm_db, label_threshold, by_eval),
                        'multi_proximity': MultiProximityFeatures(hmmer_path, hmm_dir, threshold_list, by_eval, gene_window, nucleotide_window),
                        'default_multi_proximity': MultiProximityFeatures(hmmer_path, hmm_dir, threshold_list, by_eval, delete_files=True),  # 5, 5000
                        "HTH_domains": HTHDomainFeatures(hmmer_path, hth_hmm_domains), 'DNA_KMers': KMersFeatures(dna_kmer_size),
                        "GC_Content": GCContentFeatures(), 'Prot_param': ProtParamsFeatures(),
                        'SmartGC': SmartGCFeatures(SMART_GC_DICT), 'Smart_AA_Kmers': SmartAAKmersFeatures(8, 8),
                        "AA_features": AAFeatures(INDEX_FILE), 'Mmseqs': MMseqsTaxonomyFeatures(mmseqs_path, tax_data_path, ncpus=64),
                        'Cross_Membrane': CrossMembraneFeatures(tmhmm_path)}
        self.features = list(feature_dict.values()) if not features else [value for key, value in feature_dict.items() if key in features]

    def run_features(self, protein_fasta, gff, ffn, fa, out_dir='features', do_print=True):
        ids = getIDs(protein_fasta)
        data = getData(fa, ffn)  # returns a DF with contig_id, contig_seq, gene_list (their sequences)
        for feature in self.features:
            before = time.time()
            feature.run_feature_to_file(protein_fasta, gff, fa, ids, data, out_dir)
            after = time.time()
            if do_print:
                print(f'{type(feature)} took: {(after-before)/60} minutes')

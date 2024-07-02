from create_tblout_file import get_tblout_file
from utilities import feature_to_file, fill_all_empty_orfs_with_zeros, MLFeature
from analyze_tblout_result import process_tblout_file


class HTHDomainFeatures(MLFeature):
    def __init__(self, hmmer_path, hmm_dir, threshold=50):
        """
        :param hmm_db: the database to run the hmm against - hmm of HTH
        """
        self.hmmer_path = hmmer_path
        self.hmm_dir = hmm_dir
        self.threshold = threshold

    def get_features(self, fasta_file, ids):
        '''
            creates a dataframe with the hth probability results and zeros in all the ids that did not pass the threshold
            :param fasta_file: a file with the fasta to predict on
            :param ids: a datarame with the list of the names of all the genes in the fasta file
            :return: dataframe
            '''
        hmm_tblout = get_tblout_file(self.hmmer_path, self.hmm_dir, fasta_file)
        hmm_df = process_tblout_file(hmm_tblout, self.threshold, by_e_value=False, only_higher_than_threshold=False)
        # score_full_seq - for all genes not just the ones that passed the threshold
        relavent_hmm_df = hmm_df[['ID', 'best_score', 'rep']].rename(columns={'best_score': 'HTH_score_full_seq', 'rep': 'HTH_rep'})
        relavent_hmm_df.HTH_rep[relavent_hmm_df['HTH_score_full_seq'] < self.threshold] = 0

        return fill_all_empty_orfs_with_zeros(relavent_hmm_df, ids['ID']).astype({"HTH_score_full_seq": "float16", 'HTH_rep': "uint32"})

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data):
        """
        This saves HTH_domain features to files. Parameters are given from the user and from class instance.
        :param protein_fasta: str, *.fasta[.gz], an absoulte path to the input file
        :param gff, fa, data: not used by this func, only accepted because this is an abstract method
        :param ids: a dataframe containing all of the fasta's IDs
        """
        feature_to_file('HTH_domains')(self.get_features)(protein_fasta, ids)

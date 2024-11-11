from .create_tblout_file import get_tblout_file
from utilities import feature_to_file, MLFeature, get_exponent
from .analyze_tblout_result import process_tblout_file


class Labeling(MLFeature):
    def __init__(self, hmmer_path, hmm_db, threshold, by_eval, n_cpus=3, res_name='labeling'):
        self.hmmer_path = hmmer_path
        self.hmm_db = hmm_db
        self.threshold = threshold
        self.by_eval = by_eval
        self.n_cpus = n_cpus
        self.res_name = res_name

    def get_features(self, fasta_file, IDs):
        print(f'fasta file is: {fasta_file}')
        hmm_tblout = get_tblout_file(self.hmmer_path, self.hmm_db, fasta_file, cpu=self.n_cpus)  # returns HMM search results
        print(f'tblout file is: {hmm_tblout}')
        hmm_df = process_tblout_file(hmm_tblout, self.threshold, by_e_value=self.by_eval)  # turns it to DF
        hmm_df['passed threshold'] = True
        hmm_df['best_eval_exponent'] = hmm_df['best_eval'].apply(get_exponent)  # taking exponent to reduce memory
        hmm_df.drop(['rep', 'hit_threshold_count', 'best_eval'], axis=1, inplace=True)
        hmm_df = hmm_df.astype({'best_eval_exponent': 'float16', 'best_score': 'float16'})
        prefix = self.res_name.replace("labeling", "")
        hmm_df = hmm_df.set_index("ID").rename(columns={col: prefix+col for col in hmm_df.columns})
        return IDs.set_index("ID").join(hmm_df)

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data, out_dir='features'):
        """
        This saves labling features to file. Parameters are given from the user and from class instance.
        :param protein_fasta: str, *.fasta[.gz], an absoulte path to the input file
        :param gff, fa, data: not used by this func, only accepted because this is an abstract method
        :param ids: a dataframe containing all of the fasta's IDs
        :param out_dir: path to output directory
        """
        feature_to_file(self.res_name, dir_path=out_dir)(self.get_features)(protein_fasta, ids)


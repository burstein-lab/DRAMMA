import pandas as pd
import subprocess
import os
from utilities import feature_to_file, MLFeature


class CrossMembraneFeatures(MLFeature):
    def __init__(self, tmhmm_path):
        """
        :param tmhmm_path: path to tmhmm program
        """
        self.tmhmm_path = tmhmm_path

    @staticmethod
    def __delete_the_output_dir(sample):
        cwd = os.getcwd()
        for entity in os.listdir(cwd):
            if entity.startswith('TMHMM_'):
                file_path = os.path.join(entity, sample.replace("|", "_")+'*')
                if file_path:
                    subprocess.call(['rm -R ' + entity], shell=True)
                    print('deleted ' + entity)
                    return

    def get_features(self, protein_fasta) -> pd.DataFrame:
        tmhmm_results = subprocess.run([f'{self.tmhmm_path} {protein_fasta}'], capture_output=True, text=True, shell=True)
        if tmhmm_results.stderr:
            print('error in cross membrane feature: ', tmhmm_results.stderr)

        gene_dict = {}
        for line in tmhmm_results.stdout.split('\n'):
            if 'Number of predicted TMHs:' in line:
                genename, count = line.split(' Number of predicted TMHs: ')
                gene_dict[genename[2:]] = count

        df = pd.DataFrame.from_dict(gene_dict, orient='index', columns=['cross_membrane_count']).rename_axis('ID')
        CrossMembraneFeatures.__delete_the_output_dir(df.index[0])
        return df.astype({"cross_membrane_count": "uint16"})

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data):
        """
        This saves cross_membrane features to files. Parameters are given from the user and from class instance.
        :param protein_fasta: str, *.fasta[.gz], an absoulte path to the input file
        :param gff, fa, ids, data: not used by this func, only accepted because this is an abstract method
        """
        feature_to_file('cross_membrane')(self.get_features)(protein_fasta)

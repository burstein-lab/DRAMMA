from Bio.SeqUtils import GC
import pandas as pd
import numpy as np
from utilities import feature_to_file, CONTIG_GENE_RATIO, MISSING_VALUE_SYMBOL, MLFeature


class GCContentFeatures(MLFeature):
    def __geneVScontigGC(self, gene, contigsize, contigGC, contigname):
        '''
        checks for the change in the GC content in the gene vs the rest of the contig, and adds more features.
        if contig size (without the gene size) < CONTIG_GENE_RATIO size it will fill it with MISSING_VALUE_SYMBOL
        :param gene: the sequence of the gene
        :param contig: the sequence of the contig
        :return: dictionary with: size of gene and contig, GC content of them both, and the ratio difference
        (contigraio-generaio) of the GC content between the gene and the rest of the contig.
        '''
        sequence = gene.seq
        final_dict = {'ID': gene.id, 'genesize': len(sequence), 'contigsize': contigsize, 'geneGC': GC(sequence),
                      'contigGC': contigGC, 'Contig': contigname}
        nogenesize = final_dict['contigsize'] - final_dict['genesize']
        numberOfGCgene = final_dict['geneGC'] * final_dict['genesize']
        if nogenesize / final_dict['genesize'] >= CONTIG_GENE_RATIO:
            nomberOfContigGC = final_dict['contigGC'] * final_dict['contigsize']
            nogeneGC = nomberOfContigGC - numberOfGCgene
            finalGC = nogeneGC / nogenesize
            difference = np.abs(finalGC - final_dict['geneGC'])
            final_dict['GC_diff'] = difference
        else:
            final_dict['GC_diff'] = MISSING_VALUE_SYMBOL
        return final_dict

    def __addGCinfo(self, contig, genes, contigname):
        '''
        adds the GC info for each gene separately and returns a dataframe with the genes and the features.
        :param genes: Dataframe of the genes so far
        :param contigs: the fasta of the contigs
        :return: a dataframe with the GCContent features - size of gene and contig, GC content of them both, and the
        ratio of the GC content between the gene and the rest of the contig - for all the genes.
        '''
        contigGC = GC(contig)
        contigsize = len(contig)
        geneGCdictList = []
        for gene in genes:  # maybe - .seq:
            geneGCdictList.append(self.__geneVScontigGC(gene, contigsize, contigGC, contigname))

        geneGCdataframe = pd.DataFrame.from_dict(geneGCdictList, orient='columns')
        return geneGCdataframe

    def get_features(self, fa, data):
        '''
        returns GC_content features
        :param data: df, contig name, seq, genes list
        :param fa: unused by this func, is only sent because of run_feature_to_file
        :return: a dataframe with the features described above.
        '''
        if len(data) == 0:
            print('Data is empty, cannot calculate GC content features!')
            print(f'fa file is: {fa}')
            return pd.DataFrame([]).rename_axis('ID')
        gc = data.apply(lambda r: self.__addGCinfo(r['contig_seq'], r['genes_list'], r['contig_name']), axis=1).to_list()
        df = pd.concat(gc, ignore_index=True).set_index('ID')
        return df.astype({'contigGC': "float16", "geneGC": "float16", "GC_diff": "float16"})

    def run_feature_to_file(self, protein_fasta, gff, fa, ids, data, out_dir='features'):
        """
        This saves GC_content features to files. Parameters are given from the user and from class instance.
        :param protein_fasta, gff, ids: not used by this func, only accepted because this is an abstract method
        :param fa: only sent to get_features so feature to file will work, as a filename is needed.
        :param data: a dataframe containing contig name, seq, genes list
        :param out_dir: path to output directory
        """
        feature_to_file('GC_Content', dir_path=out_dir)(self.get_features)(fa, data)

import pickle
import pandas as pd
import numpy as np


class LDmatrix:
    def __init__(self, ld_prefix):
        """
        Loading an existing LD matrix

        Parameters:
        ------------
        ld_prefix: prefix of LD matrix file 

        """
        self.ld_info = pd.read_csv(f"{ld_prefix}_ld_info.txt", delim_whitespace=True, header=None, 
                                   names=['CHR', 'SNP', 'CM', 'POS', 'A0', 'A1', 'N', 'block', 'block_idx'])
        if not (np.diff(pd.unique(self.ld_info['CHR'])) > 0).all():
            raise ValueError(f'The chrs in the LD matrix are not sorted')
        # if not self.ld_info.groupby('chr')['pos'].apply(lambda x: x.is_monotonic_increasing).all():
        #     raise ValueError(f'The SNPs in the LD matrix {ld_prefix} are not sorted')

        self.data = pickle.load(open(f"{ld_prefix}_ld.dat", 'rb')) 
        self.is_truncated = False
        self.block_sizes, self.block_ranges = self._get_block_info(self.ld_info)
        

    @classmethod
    def _get_block_info(cls, ld_info):
        block_sizes = list(pd.value_counts(ld_info['block']).sort_index())
        block_ranges = []
        begin, end = 0, 0
        for block_size in block_sizes:
            begin = end
            end += block_size
            block_ranges.append((begin, end))
        return block_sizes, block_ranges


    def _get_2d_matrix(self, dim, tril):
        """
        Recovering the lower triangle w/o the diag of a matrix

        """
        if len(tril.shape) == 2:
            raise ValueError('The block has wrong dimension (data may have been truncated)')
        matrix = np.zeros((dim, dim))
        matrix[np.tril_indices(matrix.shape[0], k = -1)] = tril
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, 1)

        return matrix


    def merge(self, ld_files):
        """
        Merging multiple LD matrices with the current one
        
        Parameters:
        ------------
        ld_files: a list of prefix of ld file
        
        """
        if self.is_truncated:
            raise ValueError('The LD matrix has been truncated')
        if len(ld_files) == 0:
            raise ValueError('There is nothing in the ld list')

        for ld_prefix in ld_files:
            ld_i = LDmatrix(ld_prefix)
            if ld_i.ld_info.loc[ld_i.ld_info.index[0], 'CHR'] >= self.ld_info.loc[self.ld_info.index[-1], 'CHR']:
                raise ValueError('Can only merge LD matrices in order (chr1, chr2, ...)')
            self.data.extend(ld_i.data)
            ld_i.ld_info['block'] += self.ld_info[self.ld_info.index[-1], 'block'] + 1
            self.ld_info = pd.concat([self.ld_info, ld_i.ld_info], axis=0)
            
            block_sizes_i, block_ranges_i = self._get_block_info(ld_i.ld_info)
            self.block_sizes.append(block_sizes_i)
            self.block_ranges.append(block_ranges_i)


    def extract(self, snps):
        """
        Extracting SNPs from the LD matrix
    
        Parameters:
        ------------
        snps: a list/set of rdID 

        Returns:
        ---------
        Updated LD matrix and LD info

        """

        if self.is_truncated:
            raise ValueError('The LD matrix has been truncated')
        self.ld_info = self.ld_info.loc[self.ld_info['SNP'].isin(snps)]
        block_dict = {k: g["block_idx"].tolist() for k,g in self.ld_info.groupby("block")}
        keep_list = []
        for i in range(len(self.data)):
            if i not in block_dict:
                continue
            else:
                self.data[i] = self._extract_by_idxs(self.data[i], self.block_sizes[i], block_dict[i])
                keep_list.append(i)
        self.data = [self.data[i] for i in keep_list]
        self.ld_info['block_idx'] = [i for _,v in self.ld_info.groupby('block') for i in range(len(v['block']))]
        self.block_sizes, self.block_ranges = self._get_block_info(self.ld_info)
        
    
    def _extract_by_idxs(self, data, size, idxs):
        """
        Extracting SNPs from an LD block using indices

        Parameters:
        ------------
        data: lower triangle w/o diag of an LD block
        size: original size of the LD block
        idxs: cols and rows to keep in the original LD block

        Returns:
        ---------
        The extracted lower triangle w/o diag of the LD block

        """
        cum_sum = np.cumsum(range(size - 1))
        idx = [cum_sum[idxs[i] - 1] + idxs[:i] for i in range(1, len(idxs))]
        if len(idx) > 0:
            idx = np.concatenate(idx)
            return data[idx] 
        else:
            return np.array([])
    
    
    def truncate(self, prop, inv):
        """
        Truncating SNPs based on eigenvalues
    
        Parameters:
        ------------
        prop: the proportion of variance to keep
        inv: if inverse the LD matrix  

        Returns:
        ---------
        A block diagonal matrix

        """

        if self.is_truncated:
            raise ValueError('The LD matrix has been truncated')
        
        ld_bd = []
        if prop == 1:
            for i in range(len(self.data)):
                block = self._get_2d_matrix(self.block_sizes[i], self.data[i])
                ld_bd.append(block)
        else: 
            for i in range(len(self.data)):
                block = self._get_2d_matrix(self.block_sizes[i], self.data[i])
                values, bases = np.linalg.eigh(block)
                values = np.flip(values)
                bases = np.flip(bases, axis=1)
                prop_var = np.cumsum(values) / np.sum(values)
                idxs = (prop_var <= prop) & (values != 0)
                values = values[idxs]
                bases = bases[:, idxs]
                if inv:
                    ld_bd.append(np.dot(bases * values ** -1, bases.T))
                else:
                    ld_bd.append(np.dot(bases * values, bases.T))
        self.data = ld_bd
        self.is_truncated = True
        

    def save(self, out):
        pickle.dump(self.data, open(f"{out}_ld.dat", 'wb'))
        self.ld_info.to_csv(f"{out}_ld_info.txt", sep='\t', index=None, header=None)
        

    
    def estimate_ldscore(self):
        """
        Estimating LD score from the LD matrix
        The Pearson correlation is adjusted by r2 - (1 - r2) / (N - 2)

        """
        if not self.is_truncated:
            raise ValueError('Truncate the LD matrix first then estimate LD scores')
        n_samples = np.array(self.ld_info['N'])
        # if n_samples < 3:
        #     raise ValueError('The number of samples to estimate LD matrix is wrong')
        ldscore = np.zeros(self.ld_info.shape[0])
        for i, (begin, end) in enumerate(self.block_ranges):
            block = self.data[i]
            raw_ld = np.sum(block ** 2, axis=0)
            adj_ld = raw_ld - (1 - raw_ld) / (n_samples[i] - 2) 
            ldscore[begin: end] = adj_ld

        merged_blocks = self._merge_blocks(self.block_sizes)

        return ldscore, merged_blocks
    
    
    def _merge_blocks(self, block_sizes):
        """
        Merge small blocks such that we have ~200 blocks with similar size

        Parameters:
        ------------
        block_ranges: a dictionary of (begin, end) of each block

        Returns:
        ---------
        merged_blocks: a list of merged blocks

        """
        n_blocks = len(block_sizes)
        mean_size = sum(block_sizes) / 200
        merged_blocks = []
        cur_size = 0
        cur_group = []
        for i, block_size in enumerate(block_sizes):
            if i < n_blocks - 1:
                if cur_size + block_size <= mean_size or cur_size + block_size // 2 <= mean_size:
                    cur_group.append(i)
                    cur_size += block_size
                else:
                    merged_blocks.append(tuple(cur_group))
                    cur_group = [i]
                    cur_size = block_size
            else:
                if cur_size + block_size <= mean_size or cur_size + block_size // 2 <= mean_size:
                    cur_group.append(i)
                    merged_blocks.append(tuple(cur_group))
                else:
                    merged_blocks.append(tuple([i]))
                    
        return merged_blocks



class LDmatrixBED(LDmatrix):
    def __init__(self, num_snps_part, ld_info, snp_getter):
        """
        Making an LD matrix
        TODO: add a check for MAF

        """
        self.data = []
        for num in num_snps_part:
            block = snp_getter(num)
            block = self._fill_na(block)
            corr = np.atleast_2d(np.corrcoef(block.T))
            tril_corr = self._get_lower_triangle(corr)
            self.data.append(tril_corr)
        self.ld_info = ld_info
        
    
    def _fill_na(self, block):
        """
        Filling missing genotypes with the mean

        """
        block_avg = np.nanmean(block, axis = 0)
        nanidx = np.where(np.isnan(block))
        block[nanidx] = block_avg[nanidx[1]]
        
        return block
    
    
    def _get_lower_triangle(self, matrix):
        """
        Extracting only the lower triangle w/o the diag of a matrix

        """
        return matrix[np.tril_indices(matrix.shape[0], k = -1)]


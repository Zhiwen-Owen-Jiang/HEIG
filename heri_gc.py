import os, time, argparse, traceback
import re
import pickle
import gzip, bz2
import numpy as np
import pandas as pd
from numpy.linalg import eigh
from scipy.stats import norm, chi2
from utils import GetLogger, sec_to_str
from ldmatrix import LDmatrix
from ldsc import LDSC


MASTHEAD = "***********************************************************************************\n"
MASTHEAD += "* Heritablity and genetic correlation for imaging genetic data analysis\n"
MASTHEAD += "***********************************************************************************"

COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'} 



class GWAS:
    required_cols = ['CHR', 'SNP', 'BP', 'N', 'Z', 'A1', 'A2']
    
    def __init__(self, gwas_files):
        """
        gwas_files: a list of gwas files
        
        """
        r = len(gwas_files)
        
        for i, gwas_file in enumerate(gwas_files):
            openfunc, compression = self._check_compression(gwas_file)
            self._check_header(openfunc, compression, gwas_file)
            gwas_data = pd.read_csv(gwas_file, delim_whitespace=True, compression=compression, 
                                    usecols=self.required_cols, na_values=[-9, 'NONE'])
            gwas_data.sort_values(by=['CHR', 'BP', 'SNP'], inplace=True) # in case the order is random
            if i == 0:
                orig_snps_list = gwas_data[['SNP', 'A1', 'A2', 'N']]
                z_mat = np.zeros((gwas_data.shape[0], r))
                valid_snp_idxs = np.zeros((gwas_data.shape[0], r))
            else:
                if not gwas_data['SNP'].equals(orig_snps_list['SNP']):
                    raise ValueError('There are different SNPs in the input gwas files')
            z_mat[:, i] = np.array(gwas_data['Z'])
            log.info(f'Pruning SNPs for {gwas_file} ...')
            gwas_data = self._prune_snps(gwas_data, log)
            final_snps_list = gwas_data['SNP']
            valid_snp_idxs[:, i] = orig_snps_list['SNP'].isin(final_snps_list)

        is_common_snp = (valid_snp_idxs == 1).all(axis=1)
        z_mat = z_mat[is_common_snp]
        common_snp_info = orig_snps_list.loc[is_common_snp]

        self.z_mat = z_mat
        self.SNP = common_snp_info['SNP']
        self.A1 = common_snp_info['A1']
        self.A2 = common_snp_info['A2']
        self.N = common_snp_info['N']


    def _check_compression(self, dir):
        """
        Checking which compression should use

        Parameters:
        ------------
        dir: diretory to the dataset

        Returns:
        ---------
        openfunc: function to open the file
        compression: type of compression
        
        """
        if dir.endswith('gz'):
            compression = 'gzip'
            openfunc = gzip.open
        elif dir.endswith('bz2'):
            compression = 'bz2'
            openfunc = bz2.BZ2File
        else:
            openfunc = open
            compression = None

        return openfunc, compression
    

    def _check_header(self, openfunc, compression, dir):
        header = openfunc(dir).readline().split()
        if compression is not None:
            header[0] = str(header[0], 'UTF-8')
            header[1] = str(header[1], 'UTF-8')
        for col in self.required_cols:
            if col not in header:
                raise ValueError(f'{col} (case sensitive) cannot be found in {dir}')
            
            
    def _prune_snps(self, gwas, log):
        """
        Prune SNPs with 
        1) any missing values in required columns
        2) infinity in Z scores, less than 0 sample size
        3) any duplicates in rsID
        4) stand ambiguous
        5) an effective sample size less than 0.67 times the 90th percentage of sample size

        Parameters:
        ------------
        gwas: a pd.DataFrame of summary statistics with required columns
        log: a logger

        Returns:
        ---------
        A pd.DataFrame of pruned summary statistics

        """
        n_snps = gwas.shape[0]
        log.info(f"{n_snps} SNPs in the raw data")

        gwas.drop_duplicates(subset=['SNP'], keep=False, inplace=True)
        log.info(f"Removed {n_snps - gwas.shape[0]} duplicated SNPs")
        n_snps = gwas.shape[0]

        # gwas = gwas.loc[~gwas.isna().any(axis=1)]
        gwas = gwas.loc[~gwas.isin([np.inf, -np.inf, np.nan]).any(axis=1)]
        log.info(f"Removed {n_snps - gwas.shape[0]} SNPs with any missing or infinate values")
        n_snps = gwas.shape[0]

        not_strand_ambiguous = [True if len(a2_) == 1 and len(a1_) == 1 and 
                                a2_ in COMPLEMENT and a1_ in COMPLEMENT and 
                                COMPLEMENT[a2_] != a1_ else False 
                                for a2_, a1_ in zip(gwas['A2'], gwas['A1'])]
        gwas = gwas.loc[not_strand_ambiguous]
        log.info(f"Removed {n_snps - gwas.shape[0]} non SNPs and strand-ambiguous SNPs")
        n_snps = gwas.shape[0]

        n_thresh = int(gwas['N'].quantile(0.9) / 1.5)
        gwas = gwas.loc[gwas['N'] >= n_thresh]
        log.info(f"Removed {n_snps - gwas.shape[0]} SNPs with N < {n_thresh}")
        
        n_snps = gwas.shape[0]
        log.info(f"{n_snps} SNPs remaining after pruning\n")

        return gwas
    
    
    def extract_snps(self, keep_snps):
        idxs = self.SNP.isin(keep_snps)
        self.z_mat = self.z_mat[idxs]
        self.SNP = self.SNP[idxs]
        self.A1 = self.A1[idxs]
        self.A2 = self.A2[idxs]
        self.N = self.N[idxs]



def align_alleles(gwas1, gwas2):
    """
    Aligning the gwas2 with the current gwas such that 
    the Z scores are measured on the same allele.
    This function assumes that current gwas and gwas2 have
    identical SNPs, which happens after calling self._prune_snps()
    
    Parameters:
    ------------
    gwas1: a GWAS instance
    gwas2: a GWAS instance

    Returns:
    ---------
    gwas2: a GWAS instance with align alleles

    """
    if not isinstance(gwas1, GWAS) or not isinstance(gwas2, GWAS):
        raise ValueError('Input should be an instance of GWAS')
    
    if len(gwas1.SNP) != len(gwas2.SNP):
        raise ValueError("The gwas2 has different number of SNPs than the current gwas")
    if (gwas1.SNP.reset_index(drop=True) != gwas2.SNP.reset_index(drop=True)).any():
        raise ValueError("Two gwas files have some different SNPs")
    
    aligned_z = [-gwas2.z_mat[i] if a11 != a12  else gwas2.z_mat[i]
                for i, (a11, a12) in enumerate(zip(gwas1.A1, gwas2.A1))]

    gwas2.z_mat = np.array(aligned_z)
    gwas2.A1 = gwas1.A1
            
    return gwas2



def parse_gwas_input(arg):
    """
    Parsing the gwas files for images. 

    Parameters:
    ------------
    arg: the prefix and file indices, seperated by a comma, e.g., `results/hipp_left_f1_score.{0:19}.glm.linear`

    Returns:
    ---------
    A list of parsed gwas files 

    """

    # prefix, indices = arg.split(',')
    # min_index, max_index = [int(s) for s in indices.split('-')]
    # gwas_files = [f"{prefix}.{i}.glm.linear" for i in range(min_index, max_index + 1)]
    p1 = r'{(.*?)}'
    p2 = r'({.*})'
    match = re.search(p1, arg)
    if match:
        file_range = match.group(1)
    else:
        raise ValueError('--ldr-gwas should be specified using `{}`, \
                         e.g. `prefix.{stard:end}.suffix`')
    start, end = [int(x) for x in file_range.split(":")]
    gwas_files = [re.sub(p2, str(i), arg) for i in range(start, end + 1)]

    return gwas_files



def check_input(args):
    """
    Checking that all inputs are correct
    replacing some args with the processed ones
    
    """
    if not args.ldr_gwas:
        raise ValueError('--ldr-gwas is required')
    if not args.bases:
        raise ValueError('--bases is required')
    if not args.inner_ldr:
        raise ValueError('--inner-ldr is required')
    if not args.ld_inv:
        raise ValueError('--ld-inv is required')
    if not args.ld:
        raise ValueError('--ld is required')
    if not args.out:
        raise ValueError('--out is required')
    args.out = args.out.split(',')
    for out in args.out:
        if not os.path.exists(os.path.dirname(out)):
            raise ValueError(f'{os.path.dirname(out)} does not exist')
    
    ldr_gwas_files = parse_gwas_input(args.ldr_gwas)
    for file in ldr_gwas_files:
        if not os.path.exists(file):
            raise ValueError(f"{file} does not exist")
    args.ldr_gwas = ldr_gwas_files
    if not os.path.exists(args.bases):
        raise ValueError(f"{args.bases} does not exist")
    if not os.path.exists(args.inner_ldr):
        raise ValueError(f"{args.inner_ldr} does not exist")
    if not os.path.exists(f"{args.ld}_ld.dat"):
        raise ValueError(f"{args.ld}_ld.dat does not exist")
    if not os.path.exists(f"{args.ld}_ld_info.txt"):
        raise ValueError(f"{args.ld}_ld_info.txt does not exist")
    if not os.path.exists(f"{args.ld_inv}_ld_info.txt"):
        raise ValueError(f"{args.ld_inv}_ld_info.txt does not exist")
    if not os.path.exists(f"{args.ld_inv}_ld_info.txt"):
        raise ValueError(f"{args.ld_inv}_ld_info.txt does not exist")
    if args.extract and not os.path.exists(args.extract):
        raise ValueError(f"{args.extract} does not exist")
    try:
        ld_prop, ld_inv_prop = [float(prop) for prop in args.var_prop.split(',')]
    except:
        log.info('ERROR: --var-prop should be specified as, e.g, 1,0.9, \
                 where there is no space before and after the comma')
    if ld_prop <= 0 or ld_prop > 1 or ld_inv_prop <= 0 or ld_inv_prop > 1:
        raise ValueError('Proportion of variance specified in \
                         --var-prop cannot be > 0 and <= 1')
    args.ld_prop = ld_prop
    args.ld_inv_prop = ld_inv_prop
    # if args.y2 and not os.path.exists(args.y2):
    #     raise ValueError(f"{args.y2} does not exist")
    # elif args.y2 and args.prevalence and not args.case_prop:
    #     raise ValueError(f"--prevalence and --case-prop should be specified simultaneously")
    # elif args.y2 and args.case_prop and not args.prevalence:
    #     raise ValueError(f"--prevalence and --case-prop should be specified simultaneously")
    # if args.prevalence:
    #     if not args.y2:
    #         log.info('WARNING: ignore --prevalence as --y2 is not specified')
    #     elif args.prevalence <= 0 or args.prevalence >= 1:
    #         raise ValueError("--prevalence shoule be > 0 and < 1")
    # if args.case_prop:
    #     if not args.y2:
    #         log.info('WARNING: ignore --case-prop as --y2 is not specified')
    #     elif args.case_prop <= 0 or args.case_prop >= 1:
    #         raise ValueError("--case-prop should be > 0 and < 1")
    if args.overlap and not args.y2:
        log.info('WARNING: ignore --overlap as --y2 is not specified')
    if args.y2:
        args.y2 = args.y2.split(',')
        if len(args.y2) != len(args.out):
            raise ValueError('--out and --y2 have different length')
        else:
            for y2 in args.y2:
                if not os.path.exists(y2):
                    raise ValueError(f"{y2} does not exist")

    return args



def get_common_snps(*snp_list):
    """
    Extracting common snps from multiple snp lists

    Parameters:
    ------------
    snp_list: multiple snp lists

    Returns:
    ---------
    common_snps: a list of common snp list

    """
    n_snp_list = len(snp_list)
    if n_snp_list == 0:
        raise ValueError('No snp list provided')

    common_snps = None
    for i in range(len(snp_list)):
        if hasattr(snp_list[i], 'SNP'):
            snp = getattr(snp_list[i], 'SNP')
            if not common_snps:
                common_snps = set(snp)
            else:
                common_snps = common_snps.intersection(snp)
                
    if not common_snps:
        raise ValueError('All the input snp lists are None or do not have a SNP column')

    return common_snps


def read_process_data(args, log):
    """
    Reading and preprocessing gwas data

    """
    r = len(args.ldr_gwas)

    # read bases and inner_ldr
    log.info(f'Reading bases from {args.bases}')
    bases = np.load(args.bases)
    if bases.shape[1] < r:
        raise ValueError('The number of bases is less than the number of LDR')
    bases = bases[:, :r]

    log.info(f'Reading inner product of LDR from {args.inner_ldr}')
    inner_ldr = np.load(args.inner_ldr)
    if inner_ldr.shape[0] < r or inner_ldr.shape[1] < r:
        raise ValueError('The dimension of inner product of LDR is less than the number of LDR')
    inner_ldr = inner_ldr[:r, :r]
    log.info(f'Keep the top {r} components\n')


    # read and prune LDR gwas
    log.info(f'Reading gwas summary statistics for {r} LDR')
    ldr_gwas = GWAS(args.ldr_gwas)
    

     # extract SNPs
    if args.extract: ## TODO: test
        try:
            header = open(args.extract).readline().split()
        except: 
            log.info('ERROR: --extract should be an unzipped txt file')
        if not header[0].startswith('rs'):
            raise ValueError('--extract should not have a header and \
                             the first column should be the rsID of SNP')
        # only the first column will be used
        keep_snps = pd.read_csv(args.extract, delim_whitespace=True, 
                               header=None, usecols=0, names='SNP') 
        log.info(f"{len(keep_snps)} SNPs are read from {args.extract}")
    else:
        keep_snps = None

    # read LD matrices
    log.info(f"Reading LD matrix from {args.ld}")
    ld = LDmatrix(args.ld)
    log.info(f"Reading LD inverse matrix from {args.ld_inv}\n")
    ld_inv = LDmatrix(args.ld_inv)
    # TODO: add a function to check if two LD matrices have identical blocks
        
    # read and prune y2 gwas
    if args.y2:
        log.info(f'Reading gwas summary statistics for the second trait')
        y2_gwas_list = [GWAS([y2_gwas_file]) for y2_gwas_file in args.y2]
    else:
        y2_gwas_list = []

    # get common snps from gwas, LD matrices, and keep_snps
    common_snps = get_common_snps(ldr_gwas, ld.ld_info, ld_inv.ld_info, 
                                  *y2_gwas_list, keep_snps) # TODO: be cautious
    if len(common_snps) > 200000:
        log.info(f"{len(common_snps)} SNPs are common in these files")
    else:
        log.info(f"Only {len(common_snps)} SNPs are common in these files. The results will be unreliable")
    ldr_gwas.extract_snps(common_snps)
    ld.extract(common_snps) # TODO: speed up
    ld_inv.extract(common_snps)

    # align summary statistics
    if args.y2:
        log.info(f"Aligning genetic effects of summary statistics")
        for y2_gwas in y2_gwas_list:
            y2_gwas.extract_snps(common_snps) 
        y2_gwas_list = [align_alleles(ldr_gwas, y2_gwas) for y2_gwas in y2_gwas_list]

    return ldr_gwas, y2_gwas_list, bases, inner_ldr, ld, ld_inv






class Estimation:
    def __init__(self, z_mat, n, ld, ld_inv, bases, inner_ldr):
        """
        ld and ld_inv have been truncated
        z_mat and y2_z have been normalized 
        """
        self.z_mat = z_mat
        self.n = np.array(n)
        self.nbar = np.mean(n)
        self.ld = ld
        self.ld_inv = ld_inv
        self.r = z_mat.shape[1]
        self.bases = bases
        self.inner_ldr = inner_ldr
        self.block_ranges = ld.block_ranges
        
        self.sigmaX_cov = np.dot(np.dot(bases, inner_ldr),bases.T) / self.nbar
        self.sigmaX_var = np.diag(self.sigmaX_cov)
        self.eff_num = self._get_eff_num(self.inner_ldr)
        self.ld_block_rank = np.array([np.sum(ld.data[i] * ld_inv.data[i].T) for i in range(len(ld.data))])
        self.ld_rank = np.sum(self.ld_block_rank)
        self.ldr_block_gene_cov = self._get_ldr_block_gene_cov()
        self.ldr_gene_cov = np.sum(self.ldr_block_gene_cov, axis=0)


    def _get_eff_num(self, matrix):
        values, _ = eigh(matrix)
        eff_num = np.sum(values) ** 2 / np.sum(values ** 2)

        return eff_num


    def _get_ldr_block_gene_cov(self): # TODO
        block_gene_cov = np.zeros((len(self.ld_block_rank), self.r, self.r)) 
        for i, (begin, end) in enumerate(self.block_ranges):
            block_gene_cov[i, :, :] = (np.dot(np.dot(self.z_mat[begin: end, :].T, 
                                                     self.ld_inv.data[i]), self.z_mat[begin: end, :]) 
            - self.ld_block_rank[i] * self.inner_ldr / self.nbar ** 2)
        
        return block_gene_cov
    

    def _get_heri_se(self, heri, d, n):
        """
        Estimating standard error of heritability estimates
        
        Parameters:
        ------------
        heri: an N by 1 vector of heritability estimates
        d: number of SNPs, or Tr(R\Omega)  
        n: sample size

        Returns:
        ---------
        An N by 1 vector of standard error estimates

        """
        part1 = 2 * d / n ** 2
        part2 = 2 * heri / n
        part3 = 2 * heri * (1 - heri) / n
        res = part1 + part2 + part3
        res[np.abs(res) < 10 ** -10] = 0

        return np.sqrt(res)
    


class OneSample(Estimation):
    def __init__(self, z_mat, n, ld, ld_inv, bases, inner_ldr):
        super().__init__(z_mat, n, ld, ld_inv, bases, inner_ldr)

        self.gene_cov = np.dot(np.dot(self.bases, self.ldr_gene_cov), self.bases.T)
        self.heri = np.diag(self.gene_cov) / self.sigmaX_var
        sigmaEta_cov = self.sigmaX_cov - self.gene_cov
        
        self.heri_se = self._get_heri_se(self.heri, self.ld_rank, self.nbar)
        self.gene_cor = self._get_gene_cor()
        self.gene_cor_se = self._get_gene_cor_se(self.heri, self.gene_cor, self.gene_cov, 
                                                 sigmaEta_cov, self.ld_rank, self.nbar)


    def _get_gene_cor(self):    
        gene_var = np.diag(self.gene_cov)
        gene_cor = self.gene_cov / np.sqrt(np.outer(gene_var, gene_var))
        
        return gene_cor


    def _get_gene_cor_se(self, heri, gene_cor, gene_cov, sigmaEta_cov, d, n):
        """
        Estimating standard error of one-sample genetic correlation estimates 
        
        Parameters:
        ------------
        heri: an N by 1 vector of heritability estimates of images
        gene_cor: an N by N matrix of genetic correlation estimates of images
        gene_cov: an N by N matrix of genetic covariance estimates of images
        sigmaEta_cov: an N by N matrix of sigmaEta estimates (the residual variance of an image)
        d: number of SNPs, or Tr(R\Omega)  
        n: sample size

        Returns:
        ---------
        An N by N matrix of standard error estimates

        """
        heri[heri <= 0] = np.nan
        gene_cov[gene_cov == 0] = 0.01 # just to avoid division by zero error

        N = len(heri)
        inv_heri = np.repeat(1 / heri - 1, N).reshape(N, N)
        gene_cor2 = gene_cor ** 2
        part1 = 1 - gene_cor2 
        part2 = sigmaEta_cov / gene_cov
        part3 = inv_heri + inv_heri.T

        res = ((4 / n + d / n**2) * part1**2 + 
            (1 / n + d / n**2) * part1 * (part3 - 2 * gene_cor2 * part2) +
            d / n**2 / 2 * gene_cor2 * (inv_heri - part2)**2 + 
            d / n**2 / 2 * gene_cor2 * (inv_heri.T - part2)**2 +
            d / n**2 * (gene_cor2**2 * part2**2 + inv_heri * inv_heri.T) -
            d / n**2 * gene_cor2 * part2 * part3)
        res[np.abs(res) < 10 ** -10] = 0

        return np.sqrt(res)
    


class TwoSample(Estimation):
    def __init__(self, z_mat, n, ld, ld_inv, bases, inner_ldr, 
                 y2_z, n2, overlap=False):
        super().__init__(z_mat, n, ld, ld_inv, bases, inner_ldr)
        self.y2_z = y2_z
        self.n2 = np.array(n2)
        self.n2bar = np.mean(n2)

        y2_block_gene_cov = self._get_y2_block_gene_cov()
        self.y2_heri = np.atleast_1d(np.sum(y2_block_gene_cov)) # since the sum stats have been normalized

        ldr_y2_block_gene_cov_part1 = self._get_ldr_y2_block_gene_cov_part1()
        ldr_y2_gene_cov_part1 = np.sum(ldr_y2_block_gene_cov_part1, axis=0)

        self.heri = np.sum(np.dot(self.bases, self.ldr_gene_cov) * self.bases, axis=1) / self.sigmaX_var
        self.heri_se = self._get_heri_se(self.heri, self.ld_rank, self.nbar)

        
        if not overlap:
            self.gene_cor_y2 = np.squeeze(self._get_gene_cor_y2(ldr_y2_gene_cov_part1, self.heri, 
                                                     self.y2_heri))
            self.gene_cor_y2_se = self._get_gene_cor_se(self.heri, self.y2_heri, self.gene_cor_y2, 
                                                        self.ld_rank, self.nbar, self.n2bar)
        else:
            ldscore, merged_blocks = ld.estimate_ldscore() # TODO: how truncation affects ldscore estimation?
            n_merged_blocks = len(merged_blocks)

            # compute left-one-block-out heritability
            ldr_lobo_gene_cov = self._lobo_estimate(self.ldr_gene_cov, self.ldr_block_gene_cov,
                                                     merged_blocks)
            y2_lobo_heri = self._lobo_estimate(self.y2_heri, y2_block_gene_cov, merged_blocks)
            temp = np.matmul(np.swapaxes(ldr_lobo_gene_cov, 1, 2), bases.T).swapaxes(1, 2)
            temp = np.sum(temp * np.expand_dims(bases, 0), axis=2)
            image_lobo_heri = temp / self.sigmaX_var

            # compute left-one-block-out cross-trait LDSC intercept
            self.ldr_heri = np.diag(self.ldr_gene_cov) / np.diag(self.inner_ldr) * self.nbar
            z_mat_raw = self.z_mat / np.sqrt(np.diagonal(inner_ldr)) * self.n.reshape(-1, 1)
            y2_z_raw = self.y2_z * np.sqrt(self.n2).reshape(-1, 1)
            ldsc_intercept = LDSC(z_mat_raw, y2_z_raw, ldscore, self.ldr_heri, self.y2_heri,
                                  self.n, self.n2, self.ld_rank, self.block_ranges, 
                                  merged_blocks)
            # ldsc_intercept.lobo_ldsc *= 0
            # ldsc_intercept.total_ldsc *= 0
            
            # compute left-one-block-out genetic correlation
            ldr_y2_lobo_gene_cov_part1 = self._lobo_estimate(ldr_y2_gene_cov_part1, 
                                                             ldr_y2_block_gene_cov_part1, 
                                                             merged_blocks)
            ld_rank_lobo = self._lobo_estimate(self.ld_rank, self.ld_block_rank, merged_blocks)    
            lobo_gene_cor = self._get_gene_cor_ldsc(ldr_y2_lobo_gene_cov_part1, ld_rank_lobo, 
                                                             ldsc_intercept.lobo_ldsc, 
                                                             image_lobo_heri, y2_lobo_heri) # n_blocks * N
            
            # compute genetic correlation using all blocks
            image_y2_gene_cor = self._get_gene_cor_ldsc(ldr_y2_gene_cov_part1, self.ld_rank, 
                                                        ldsc_intercept.total_ldsc, 
                                                        self.heri, self.y2_heri) # 1 * N

            # compute jackknite estimate of genetic correlation and se
            self.gene_cor_y2, self.gene_cor_y2_se = self._jackknife(image_y2_gene_cor, 
                                                                    lobo_gene_cor, 
                                                                    n_merged_blocks)

        # if prevalence and case_prop:
        #     self.y2_heri_lia = self.y2_heri * self._obs_to_liability(prevalence, case_prop)
        #     self.y2_heri_lia_se = self._get_heri_se(self.y2_heri_lia, self.ld_rank, self.n2bar)
        self.y2_heri_se = self._get_heri_se(self.y2_heri, self.ld_rank, self.n2bar)


    def _get_gene_cor_ldsc(self, part1, ld_rank, ldsc, heri1, heri2):
        # ldr_gene_cov = part1 - ld_rank.reshape(-1, 1) * ldsc / np.mean(np.sqrt(self.n * self.n2))
        ldr_gene_cov = part1 - ld_rank.reshape(-1, 1) * ldsc * np.sqrt(np.diagonal(self.inner_ldr)) / self.nbar / np.sqrt(self.n2bar)
        gene_cor = self._get_gene_cor_y2(ldr_gene_cov.T, heri1, heri2)

        return gene_cor
    
        
    def _get_gene_cor_y2(self, inner_part, heri1, heri2):
        bases_inner_part = np.dot(self.bases, inner_part).reshape(self.bases.shape[0], -1)
        gene_cov_y2 =  bases_inner_part / np.sqrt(self.sigmaX_var).reshape(-1, 1)
        gene_cor_y2 = gene_cov_y2.T / np.sqrt(heri1 * heri2)

        return gene_cor_y2


    def _jackknife(self, total, lobo, n_blocks):
        """
        Jackknite estimator

        Parameters:
        ------------
        total: the estimate using all blocks
        lobo: an n_blocks by N matrix of lobo estimates
        n_blocks: the number of blocks

        Returns:
        ---------
        estimate: an N by 1 vector of jackknife estimates
        se: an N by 1 vector of jackknife se estimates

        """
        mean_lobo = np.mean(lobo, axis=0)
        estimate = n_blocks * total - (n_blocks - 1) * mean_lobo
        se = np.sqrt((n_blocks - 1) / n_blocks * np.sum((lobo - mean_lobo) ** 2, axis=0))
        estimate = np.squeeze(estimate)
        se = np.squeeze(se)

        return estimate, se


    def _lobo_estimate(self, total_est, block_est, merged_blocks):
        lobo_est = []

        for blocks in merged_blocks:
            lobo_est_i = total_est.copy()
            for ii in blocks:
                lobo_est_i -= block_est[ii]
            lobo_est.append(lobo_est_i)
        
        return np.array(lobo_est)


    def _get_gene_cor_se(self, heri, heri_y2, gene_cor_y2, d, n1, n2):
        """
        Estimating standard error of two-sample genetic correlation estimates
        without sample overlap
        
        Parameters:
        ------------
        heri: an N by 1 vector of heritability estimates of images
        heri_y2: an 1 by 1 number of heritability estimate of a single trait
        gene_cor_y2: an N by 1 vector of genetic correlation estimates 
                    between images and a single trait
        d: number of SNPs, or Tr(R\Omega)  
        n1: sample size of images
        n2: sample size of the single trait

        Returns:
        ---------
        An N by N matrix of standard error estimates
        This estimator assumes no sample overlap
        
        """
        gene_cor_y2sq = gene_cor_y2 ** 2
        gene_cor_y2sq1 = 1 - gene_cor_y2sq
        part1 = gene_cor_y2sq / (2 * heri ** 2) * d / n1 ** 2
        part2 = gene_cor_y2sq / (2 * heri_y2 ** 2) * d / n2 ** 2
        part3 = 1 / (heri * heri_y2) * d / (n1 * n2)
        part4 = gene_cor_y2sq1 / (heri * n1)
        part5 = gene_cor_y2sq1 / (heri_y2 * n2)
        var = part1 + part2 + part3 + part4 + part5

        return np.sqrt(var)



    def _get_y2_block_gene_cov(self):
        block_gene_cov = np.zeros(len(self.block_ranges)) 
        for i, (begin, end) in enumerate(self.block_ranges):
            block_gene_cov[i] = (np.dot(np.dot(self.y2_z[begin: end].T, self.ld_inv.data[i]), 
                                        self.y2_z[begin: end]) 
            - self.ld_block_rank[i] / self.n2bar)
            
        return block_gene_cov
    

    def _get_ldr_y2_block_gene_cov_part1(self):
        gene_cov = np.zeros((len(self.block_ranges), self.r))
        for i, (begin, end) in enumerate(self.block_ranges):
            gene_cov[i, :] = np.squeeze(np.dot(np.dot(self.z_mat[begin: end].T, self.ld_inv.data[i]), 
                                    self.y2_z[begin: end]))
            
        return gene_cov
    

    # def _obs_to_liability(self, prevalence, case_prop):
    #     num = prevalence ** 2 * (1 - prevalence) ** 2 
    #     denom = case_prop * (1 - case_prop) * norm.pdf(norm.ppf(prevalence)) ** 2
        
    #     return num / denom
        


def format_heri(heri, heri_se):
    invalid_heri = (heri > 1) | (heri < 0) | (heri_se > 1) | (heri_se < 0)
    heri[invalid_heri] = np.nan
    heri_se[invalid_heri] = np.nan
    log.info('Removed out-of-bound results (if any)')

    chisq = (heri / heri_se) ** 2
    pv = chi2.sf(chisq, 1)
    data = {'index': range(1, len(heri) + 1), 'heri': heri, 'se': heri_se, 
                           'chisq': chisq, 'pv': pv}
    output = pd.DataFrame(data)
    return output



def format_gene_cor_y2(heri, heri_se, gene_cor, gene_cor_se):
    invalid_heri = (heri > 1) | (heri < 0) | (heri_se > 1) | (heri_se < 0)
    heri[invalid_heri] = np.nan
    heri_se[invalid_heri] = np.nan
    invalid_gene_cor = (gene_cor > 1) | (gene_cor < -1) | (gene_cor_se > 1) | (gene_cor_se < 0)
    gene_cor[invalid_gene_cor] = np.nan
    gene_cor_se[invalid_gene_cor] = np.nan
    log.info('Removed out-of-bound results (if any)')

    heri_chisq = (heri / heri_se) ** 2
    heri_pv = chi2.sf(heri_chisq, 1) 
    gene_cor_chisq = (gene_cor / gene_cor_se) ** 2
    gene_cor_pv = chi2.sf(gene_cor_chisq, 1)
    data = {'index': range(1, len(gene_cor) + 1), 
            'image_heri': heri, 'image_heri_se': heri_se, 
            'image_heri_chisq': heri_chisq, 'image_heri_pv': heri_pv, 
            'image_y2_gc': gene_cor, 'image_y2_gc_se': gene_cor_se, 
            'image_y2_gc_chisq': gene_cor_chisq, 'image_y2_gc_pv': gene_cor_pv}
    output = pd.DataFrame(data)
    return output


def print_results_two(heri_gc, output, i):
    if i == 0:
        n_image_heri_sig = np.sum(output['image_heri_pv'] < 0.05 / heri_gc.eff_num)
        msg = 'Heritability of the image\n'
        msg += '-------------------------\n'
        msg += f"Mean h^2: {round(np.nanmean(output['image_heri']), 4)} ({round(np.nanmean(output['image_heri_se']), 4)})\n"
        msg += f"Max h^2: {round(np.nanmax(output['image_heri']), 4)}\n"
        msg += f"Min h^2: {round(np.nanmin(output['image_heri']), 4)}\n"
        msg += f'{n_image_heri_sig} grid points have significant heritability (threshold: {np.round(0.05 / heri_gc.eff_num, 4)})\n'
        msg += '\n'
    else:
        msg = ''

    chisq_y2_heri = (heri_gc.y2_heri[0] / heri_gc.y2_heri_se[0]) ** 2
    pv_y2_heri = chi2.sf(chisq_y2_heri, 1) 
    msg += 'Heritability of the second trait\n'
    msg += '--------------------------------\n'
    msg += f"Total observed scale h^2: {round(heri_gc.y2_heri[0], 4)} ({round(heri_gc.y2_heri_se[0], 4)})\n"
    msg += f"Chisq: {round(chisq_y2_heri, 4)}\n"
    msg += f"P: {round(pv_y2_heri, 4)}\n"
    # if hasattr(heri_gc, "y2_heri_lia") and hasattr(heri_gc, "y2_heri_lia_se"):
    #     msg += f"Total liability scale h^2: {round(heri_gc.y2_heri_lia[0], 4)} ({round(heri_gc.y2_heri_lia_se[0], 4)})\n"
    #     chisq_y2_heri_lia = (heri_gc.y2_heri_lia[0] / heri_gc.y2_heri_lia_se[0]) ** 2
    #     pv_y2_heri_lia = chi2.sf(chisq_y2_heri_lia, 1)
    #     msg += f"Chisq: {round(chisq_y2_heri_lia, 4)}\n"
    #     msg += f"P: {round(pv_y2_heri_lia, 4)}\n"         
    msg += '\n'
    
    n_image_y2_gc_sig = np.nansum(output['image_y2_gc_pv'] < 0.05 / heri_gc.eff_num)
    if args.overlap:
        msg += 'Genetic correlation (with sample overlap)\n'
    else:
        msg += 'Genetic correlation (without sample overlap)\n'
    msg += '--------------------------------------------\n'
    msg += f"Mean genetic correlation: {round(np.nanmean(output['image_y2_gc']), 4)} ({round(np.nanmean(output['image_y2_gc_se']), 4)})\n"
    msg += f"Max genetic correlation: {round(np.nanmax(output['image_y2_gc']), 4)}\n"
    msg += f"Min genetic correlation: {round(np.nanmin(output['image_y2_gc']), 4)}\n"
    msg += f'{n_image_y2_gc_sig} grid points have significant genetic correlation (threshold: {np.round(0.05 / heri_gc.eff_num, 4)})\n'
    
    return msg


def print_results_one(heri_output, gene_cor, gene_cor_se, eff_num):
    n_image_heri_sig = np.nansum(heri_output['pv'] < 0.05 / eff_num)
    msg = 'Heritability of the image\n'
    msg += '-------------------------\n'
    msg += f"Mean h^2: {round(np.nanmean(heri_output['heri']), 4)} ({round(np.nanmean(heri_output['se']), 4)})\n"
    msg += f"Max h^2: {round(np.nanmax(heri_output['heri']), 4)}\n"
    msg += f"Min h^2: {round(np.nanmin(heri_output['heri']), 4)}\n"
    msg += f'{n_image_heri_sig} grid points have significant heritability (threshold: {np.round(0.05 / eff_num, 4)})\n'
    msg += '\n'

    msg += 'Genetic correlation of the image\n'
    msg += '--------------------------------\n'
    msg += f"Mean genetic correlation: {round(np.nanmean(gene_cor), 4)} ({round(np.nanmean(gene_cor_se), 4)})\n"

    return msg


    
def main(args, log):
    args = check_input(args)
    ldr_gwas, y2_gwas_list, bases, inner_ldr, ld, ld_inv = read_process_data(args, log)

    # normalize summary statistics of LDR
    z_mat = ldr_gwas.z_mat * np.sqrt(np.diagonal(inner_ldr)) / np.array(ldr_gwas.N).reshape(-1, 1)

    # truncate LD matrices
    log.info(f"Keep {args.ld_prop * 100}% variance of the LD matrix")
    ld.truncate(prop=args.ld_prop, inv=False)
    log.info(f"Keep {args.ld_inv_prop * 100}% variance of the LD inverse matrix\n")
    ld_inv.truncate(prop=args.ld_inv_prop, inv=True)
    
    # # for debug
    # ldr_gwas = pickle.load(open('ldr_gwas_top27.dat', 'rb'))
    # y2_gwas = pickle.load(open('insomnia_gwas_top27.dat', 'rb'))
    # ld = pickle.load(open('ld_top27.dat', 'rb'))
    # ld_inv = pickle.load(open('ld_inv_top27.dat', 'rb'))
    # bases = np.load('/work/users/o/w/owenjf/image_genetics/methods/real_data_analysis/res/ACR.L/FA_ACR.L_phase123_white_bases_top27.npy')
    # inner_ldr = np.load('/work/users/o/w/owenjf/image_genetics/methods/real_data_analysis/res/ACR.L/FA_ACR.L_phase123_white_proj_innerldr_top27.npy')
    # n = ldr_gwas.N.mean()
    # z_mat = ldr_gwas.z_mat * np.sqrt(np.diagonal(inner_ldr)) / n

    if not args.y2:
        heri_gc = OneSample(z_mat, ldr_gwas.N, ld, ld_inv, bases, inner_ldr)
        heri_output = format_heri(heri_gc.heri, heri_gc.heri_se)
        msg = print_results_one(heri_output, heri_gc.gene_cor, heri_gc.gene_cor_se, heri_gc.eff_num)
        log.info(f'{msg}')
        heri_output.to_csv(f"{args.out[0]}_heri.txt", sep='\t', index=None, float_format='%.5e', na_rep='NA')
        gene_cor_tril = heri_gc.gene_cor[np.tril_indices(heri_gc.gene_cor.shape[0], k = -1)]
        gene_cor_se_tril = heri_gc.gene_cor_se[np.tril_indices(heri_gc.gene_cor_se.shape[0], k = -1)]
        invalid_gene_cor = (gene_cor_tril > 1) | (gene_cor_tril < -1) | (gene_cor_se_tril > 1) | (gene_cor_se_tril < 0)
        gene_cor_tril[invalid_gene_cor] = np.nan
        gene_cor_se_tril[invalid_gene_cor] = np.nan
        np.savez_compressed(f'{args.out[0]}_gene_cor', gc=gene_cor_tril, se=gene_cor_se_tril)
        log.info(f'Save the heritability results to {args.out[0]}_heri.txt')
        log.info(f'Save the genetic correlation results to {args.out[0]}_gene_cor.npz')
    else:
        for i, y2_gwas in enumerate(y2_gwas_list):
            y2_z = y2_gwas.z_mat / np.sqrt(np.array(y2_gwas.N)).reshape(-1, 1)
            heri_gc = TwoSample(z_mat, ldr_gwas.N, ld, ld_inv, bases, inner_ldr,
                                y2_z, y2_gwas.N, args.overlap)
            gene_cor_y2_output = format_gene_cor_y2(heri_gc.heri, heri_gc.heri_se,
                                                    heri_gc.gene_cor_y2, heri_gc.gene_cor_y2_se)
            msg = print_results_two(heri_gc, gene_cor_y2_output, i)
            log.info(f'{msg}')
            gene_cor_y2_output.to_csv(f"{args.out[i]}_gene_cor_y2.txt", sep='\t', index=None, float_format='%.5e', na_rep='NA')
            log.info(f'Save the genetic correlation results to {args.out[i]}_gene_cor_y2.txt\n')
    # log.info(f"A suggested effective number for multiple hypothesis adjustment is {np.round(heri_gc.eff_num, 4)}\n")
        
        


parser = argparse.ArgumentParser()
parser.add_argument('--ldr-gwas', help='directory to LDR gwas files (prefix)')
parser.add_argument('--bases', help='directory to bases')
parser.add_argument('--inner-ldr', help='directory to inner product of LDR')
parser.add_argument('--ld-inv', help='directory to the inverse LD matrix')
parser.add_argument('--ld', help='directory to LD matrix')
parser.add_argument('--out', help='directory for output')
parser.add_argument('--var-prop', help='prop of variance to keep for LD and LD inverse, respectively, such as 1,0.9')
parser.add_argument('--extract', help='a text file of SNPs to extract')
parser.add_argument('--y2', help='gwas results of another traits')
parser.add_argument('--overlap', action='store_true', help='if there are sample overlap')
# parser.add_argument('--prevalence', type=float, help='prevalence of a binary trait')
# parser.add_argument('--case-prop', type=float, help='proportion of cases in the summary statistics')



if __name__ == '__main__':
    args = parser.parse_args()

    logpath = os.path.join(f"{args.out.split(',')[0]}_heri_gc.log")
    log = GetLogger(logpath)

    log.info(MASTHEAD)
    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(''))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "Parsed arguments\n"
        options = ['--'+x.replace('_','-')+' '+str(opts[x]) for x in non_defaults]
        header += '\n'.join(options).replace('True','').replace('False','')
        header = header+'\n'
        log.info(header)
        main(args, log)
    except Exception:
        log.info(traceback.format_exc())
        raise
    finally:
        log.info(f"Analysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")
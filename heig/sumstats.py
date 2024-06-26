import os
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2
from heig import utils
import heig.input.dataset as ds

"""
TODO: 
add a parallel option for preprocessing LDR GWAS summary statistics

"""


def check_input(args, log):
    # required arguments
    if args.ldr_gwas is None and args.y2_gwas is None:
        raise ValueError('either --ldr-gwas or --y2-gwas should be provided')
    if args.snp_col is None:
        raise ValueError('--snp-col is required')
    if args.a1_col is None:
        raise ValueError('--a1-col is required')
    if args.a2_col is None:
        raise ValueError('--a2-col is required')

    # optional arguments
    if args.n_col is None and args.n is None:
        raise ValueError('either --n-col or --n is required')
    if args.ldr_gwas is not None and args.y2_gwas is not None:
        raise ValueError('can only specify --ldr-gwas or --y2-gwas')
    elif args.ldr_gwas is not None:
        if args.effect_col is None:
            raise ValueError(
                '--effect-col is required for LDR summary statistics')
        if args.se_col is None:
            raise ValueError(
                '--se-col is required for LDR summary statistics')
        if args.chr_col is None:
            raise ValueError(
                '--chr-col is required for LDR summary statistics')
        if args.pos_col is None:
            raise ValueError(
                '--pos-col is required for LDR summary statistics')
    elif args.y2_gwas is not None:
        if not (args.z_col is not None or args.effect_col is not None and args.se_col is not None
                or args.effect_col is not None and args.p_col is not None):
            raise ValueError(('specify --z-col or --effect-col + --se-col or '
                              '--effect-col + --p-col for --y2-gwas'))

    if args.maf_col is not None and args.maf_min is not None:
        if args.maf_min <= 0 or args.maf_min >= 0.5:
            raise ValueError(
                '--maf-min must be greater than 0 and less than 0.5')
    elif args.maf_col is None and args.maf_min is not None:
        log.info('WARNING: No --maf-col is provided. Ignore --maf-min')
        args.maf_min = None
    elif args.maf_col and args.maf_min is None:
        log.info('Set minimum MAF as 0.9 by default.')
        args.maf_min = 0.01

    if args.info_col is not None and args.info_min is not None:
        if args.info_min <= 0 or args.info_min >= 1:
            raise ValueError('--info-min should be between 0 and 1')
    elif args.info_col is None and args.info_min:
        log.info('WARNING: No --info-col column is provided. Ignore --info-min')
        args.info_min = None
    elif args.info_col and args.info_min is None:
        log.info('Set minimum INFO as 0.9 by default.')
        args.info_min = 0.9

    if args.n is not None and args.n <= 0:
        raise ValueError('--n should be greater than 0')

    # processing some arguments
    if args.ldr_gwas is not None:
        ldr_gwas_files = ds.parse_input(args.ldr_gwas)
        for file in ldr_gwas_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} does not exist")
        args.ldr_gwas = ldr_gwas_files
    elif args.y2_gwas is not None and not os.path.exists(args.y2_gwas):
        raise FileNotFoundError(f"{args.y2_gwas} does not exist")

    if args.effect_col is not None:
        try:
            args.effect, args.null_value = args.effect_col.split(',')
            args.null_value = int(args.null_value)
        except:
            raise ValueError(
                '--effect-col should be specified as `BETA,0` or `OR,1`')
        if args.null_value not in (0, 1):
            raise ValueError(
                'The null value should be 0 for BETA (log OR) or 1 for OR')
    else:
        args.effect, args.null_value = None, None

    return args


def map_cols(args):
    """
    Creating two dicts for mapping provided colnames and standard colnames

    Parameters:
    ------------
    args: instance of arguments

    Returns:
    ---------
    cols_map: keys are standard colnames, values are provided colnames
    cols_map2: keys are provided colnames, values are standard colnames

    """
    cols_map = dict()
    cols_map['N'] = args.n_col
    cols_map['n'] = args.n
    cols_map['CHR'] = args.chr_col
    cols_map['POS'] = args.pos_col
    cols_map['SNP'] = args.snp_col
    cols_map['EFFECT'] = args.effect
    cols_map['null_value'] = args.null_value
    cols_map['SE'] = args.se_col
    cols_map['A1'] = args.a1_col
    cols_map['A2'] = args.a2_col
    cols_map['Z'] = args.z_col
    cols_map['P'] = args.p_col
    cols_map['MAF'] = args.maf_col
    cols_map['maf_min'] = args.maf_min
    cols_map['INFO'] = args.info_col
    cols_map['info_min'] = args.info_min

    cols_map2 = dict()
    for k, v in cols_map.items():
        if v is not None and k not in ('n', 'maf_min', 'info_min', 'null_value'):
            cols_map2[v] = k

    return cols_map, cols_map2


def read_sumstats(prefix):
    """
    Reading preprocessed summary statistics and creating a GWAS instance.

    Parameters:
    ------------
    prefix: the prefix of summary statistics file

    Returns:
    ---------
    a GWAS instance

    """
    snpinfo_dir = f'{prefix}.snpinfo'
    sumstats_dir = f'{prefix}.sumstats'

    if not os.path.exists(snpinfo_dir) or not os.path.exists(sumstats_dir):
        raise FileNotFoundError(
            f"either .sumstats or .snpinfo file does not exist")
    
    with open(sumstats_dir, 'rb') as file:
        sumstats = pickle.load(file)
    snpinfo = pd.read_csv(snpinfo_dir, sep='\s+')

    if sumstats['beta'] is not None:
        n_snps = sumstats['beta'].shape[0]
    else:
        n_snps = sumstats['z'].shape[0]
    if snpinfo.shape[0] != n_snps:
        raise ValueError(("summary statistics and the meta data contain different number of SNPs, "
                          "which means the files have been modified"))

    return GWAS(sumstats['beta'], sumstats['se'], sumstats['z'], snpinfo)


class GWAS:
    required_cols_ldr = ['CHR', 'POS', 'SNP', 'A1', 'A2']
    required_cols_y2 = ['SNP', 'A1', 'A2']
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    def __init__(self, beta, se, z, snpinfo):
        self.beta = beta
        self.se = se
        self.z = z
        self.snpinfo = snpinfo

    @classmethod
    def from_rawdata_ldr(cls, gwas_files, cols_map, cols_map2,
                         maf_min=None,
                         info_min=None,
                         fast_sumstats=False):
        """
        Preprocessing LDR GWAS summary statistics. BETA and SE are required columns. 

        Parameters:
        ------------
        gwas_files: a list of gwas files
        cols_map: a dict mapping standard colnames to provided colnames
        cols_map2: a dict mapping provided colnames to standard colnames 
        maf_min: the minumum of MAF
        info_min: the minumum of INFO
        fast_sumstats: if only do pruning for the first LDR gwas file

        Returns:
        ---------
        a GWAS instance

        """
        cls.logger = logging.getLogger(__name__)
        r = len(gwas_files)
        if fast_sumstats:
            cls.logger.info('Using fast mode that only the first GWAS file will be QCed.')
        cls.logger.info(
            f'Reading and processing {r} LDR GWAS summary statistics files ...\n')
        
        for i, gwas_file in enumerate(gwas_files):
            cls.logger.info(f'GWAS file {i+1}')
            openfunc, compression = utils.check_compression(gwas_file)
            cls._check_header(openfunc, compression,
                              gwas_file, cols_map, cols_map2, True)
            gwas_data = pd.read_csv(gwas_file, sep='\s+', compression=compression,
                                    usecols=list(cols_map2.keys()), na_values=[-9, 'NONE', '.'],
                                    dtype={'A1': 'category', 'A2': 'category'})
            gwas_data = gwas_data.rename(cols_map2, axis=1)
            gwas_data['A1'] = gwas_data['A1'].str.upper().astype('category')
            gwas_data['A2'] = gwas_data['A2'].str.upper().astype('category')

            if i == 0:
                if cols_map['N'] is None:
                    gwas_data['N'] = cols_map['n']
                cls._check_median(gwas_data['EFFECT'],
                                  'EFFECT', cols_map['null_value'])
                orig_snps_list = gwas_data[[
                    'CHR', 'POS', 'SNP', 'A1', 'A2', 'N']]
                beta_mat = np.zeros((gwas_data.shape[0], r))
                se_mat = np.zeros((gwas_data.shape[0], r))
                valid_snp_idxs = np.ones(gwas_data.shape[0], dtype=bool)

            if i > 0 and not fast_sumstats:
                cls._check_median(gwas_data['EFFECT'],
                                  'EFFECT', cols_map['null_value'])
                if not gwas_data['SNP'].equals(orig_snps_list['SNP']):
                    raise ValueError(
                        'different SNPs in the input LDR GWAS files')

            if cols_map['null_value'] == 1:
                gwas_data['EFFECT'] = np.log(gwas_data['EFFECT'])
            beta_mat[:, i] = np.array(gwas_data['EFFECT'])
            se_mat[:, i] = np.array(gwas_data['SE'])
            
            if i == 0 or not fast_sumstats:
                cls.logger.info(f'Pruning SNPs for {gwas_file} ...')
                gwas_data = cls._prune_snps(gwas_data, maf_min, info_min)
                final_snps_list = gwas_data['SNP']
                valid_snp_idxs = valid_snp_idxs & orig_snps_list['SNP'].isin(
                    final_snps_list).values

        is_common_snp = valid_snp_idxs == 1
        beta_mat = beta_mat[is_common_snp]
        se_mat = se_mat[is_common_snp]
        common_snp_info = orig_snps_list.loc[is_common_snp]

        z = None
        snpinfo = common_snp_info.reset_index(drop=True)

        return cls(beta_mat, se_mat, z, snpinfo)

    @classmethod
    def from_rawdata_y2(cls, gwas_file, cols_map, cols_map2, maf_min=None, info_min=None):
        """
        Preprocessing non-imaging GWAS summary statistics.

        Parameters:
        ------------
        gwas_files: a list of gwas files
        cols_map: a dict mapping standard colnames to provided colnames
        cols_map2: a dict mapping provided colnames to standard colnames 
        maf_min: the minumum of MAF
        info_min: the minumum of INFO

        Returns:
        ---------
        a GWAS instance

        """
        cls.logger = logging.getLogger(__name__)
        cls.logger.info(
            f'Reading and processing the non-imaging GWAS summary statistics file ...\n')

        openfunc, compression = utils.check_compression(gwas_file)
        cls._check_header(openfunc, compression, gwas_file,
                          cols_map, cols_map2, False)
        gwas_data = pd.read_csv(gwas_file, sep='\s+', compression=compression,
                                usecols=list(cols_map2.keys()), na_values=[-9, 'NONE', '.'],
                                dtype={'A1': 'category', 'A2': 'category'})  # TODO: read by block
        gwas_data = gwas_data.rename(cols_map2, axis=1)
        gwas_data['A1'] = gwas_data['A1'].str.upper().astype(
            'category')  # increase memory usage from 1k to 2k
        gwas_data['A2'] = gwas_data['A2'].str.upper().astype('category')

        if cols_map['N'] is None:
            gwas_data['N'] = cols_map['n']

        if cols_map['EFFECT'] is not None and cols_map['SE'] is not None:
            cls._check_median(gwas_data['EFFECT'],
                              'EFFECT', cols_map['null_value'])
            if cols_map['null_value'] == 1:
                gwas_data['EFFECT'] = np.log(gwas_data['EFFECT'])
            gwas_data['Z'] = gwas_data['EFFECT'] / gwas_data['SE']
        elif cols_map['null_value'] is not None and cols_map['P'] is not None:
            abs_z_score = np.sqrt(chi2.ppf(1 - gwas_data['P'], 1))
            if cols_map['null_value'] == 0:
                gwas_data['Z'] = ((gwas_data['EFFECT'] > 0)
                                  * 2 - 1) * abs_z_score
            else:
                gwas_data['Z'] = ((gwas_data['EFFECT'] > 1)
                                  * 2 - 1) * abs_z_score
        else:
            cls._check_median(gwas_data['Z'], 'Z', 0)

        cls.logger.info(f'Pruning SNPs for {gwas_file} ...')
        gwas_data = cls._prune_snps(gwas_data, maf_min, info_min)
        beta = None
        se = None
        # z = gwas_data['Z'].reset_index(drop=True)
        z = gwas_data['Z'].to_numpy().reshape(-1, 1)
        snpinfo = gwas_data[['SNP', 'A1', 'A2', 'N']].reset_index(drop=True)

        return cls(beta, se, z, snpinfo)

    @classmethod
    def _prune_snps(cls, gwas, maf_min, info_min):
        """
        Pruning SNPs according to
        1) any missing values in required columns
        2) infinity in Z scores, less than 0 sample size
        3) any duplicates in rsID (indels)
        4) strand ambiguous
        5) an effective sample size less than 0.67 times the 90th percentage of sample size
        6) small MAF or small INFO score (optional)

        Parameters:
        ------------
        gwas: a pd.DataFrame of summary statistics with required columns
        maf_min: the minimum MAF
        info_min: the minimum INFO

        Returns:
        ---------
        A pd.DataFrame of pruned summary statistics

        """
        n_snps = cls._check_remaining_snps(gwas)
        cls.logger.info(f"{n_snps} SNPs in the raw data.")

        gwas.drop_duplicates(subset=['SNP'], keep=False, inplace=True)
        cls.logger.info(f"Removed {n_snps - gwas.shape[0]} duplicated SNPs.")
        n_snps = cls._check_remaining_snps(gwas)

        # increased a little memory
        gwas = gwas.loc[~gwas.isin([np.inf, -np.inf, np.nan]).any(axis=1)]
        cls.logger.info(
            f"Removed {n_snps - gwas.shape[0]} SNPs with any missing or infinite values.")
        n_snps = cls._check_remaining_snps(gwas)

        not_strand_ambiguous = [True if len(a2_) == 1 and len(a1_) == 1 and
                                a2_ in cls.complement and a1_ in cls.complement and
                                cls.complement[a2_] != a1_ else False
                                for a2_, a1_ in zip(gwas['A2'], gwas['A1'])]
        gwas = gwas.loc[not_strand_ambiguous]
        cls.logger.info(
            f"Removed {n_snps - gwas.shape[0]} non SNPs and strand-ambiguous SNPs.")
        n_snps = cls._check_remaining_snps(gwas)

        n_thresh = int(gwas['N'].quantile(0.9) / 1.5)
        gwas = gwas.loc[gwas['N'] >= n_thresh]
        cls.logger.info(
            f"Removed {n_snps - gwas.shape[0]} SNPs with N < {n_thresh}.")
        n_snps = cls._check_remaining_snps(gwas)

        if maf_min is not None:
            gwas = gwas.loc[gwas['MAF'] >= maf_min]
            cls.logger.info(
                f"Removed {n_snps - gwas.shape[0]} SNPs with MAF < {maf_min}.")
            n_snps = cls._check_remaining_snps(gwas)

        if info_min is not None:
            gwas = gwas.loc[gwas['INFO'] >= info_min]
            cls.logger.info(
                f"Removed {n_snps - gwas.shape[0]} SNPs with INFO < {info_min}.")
            n_snps = cls._check_remaining_snps(gwas)

        cls.logger.info(f"{n_snps} SNPs remaining after pruning.\n")

        return gwas

    @staticmethod
    def _check_remaining_snps(gwas):
        """
        Checking #SNPs 

        """
        n_snps = gwas.shape[0]
        if n_snps == 0:
            raise ValueError(
                'no SNP remaining. Check if misspecified columns')
        return n_snps

    @classmethod
    def _check_header(cls, openfunc, compression, dir, cols_map, cols_map2, ldr=True):
        """
        Checking if all required columns exist; 
        checking if all provided columns exist.

        Parameters:
        ------------
        openfunc: function to open the file
        compression: compression mode
        dir: directory to gwas file
        cols_map: a dict mapping standard colnames to provided colnames
        cols_map2: a dict mapping provided colnames to standard colnames 
        ldr: if it is an LDR gwas file

        """
        with openfunc(dir, 'r') as file:
            header = file.readline().split()
        if compression is not None:
            header = [str(x, 'UTF-8') for x in header]
        if ldr:
            for col in cls.required_cols_ldr:
                if cols_map[col] not in header:
                    raise ValueError(
                        f'{cols_map[col]} (case sensitive) cannot be found in {dir}')
        else:
            for col in cls.required_cols_y2:
                if cols_map[col] not in header:
                    raise ValueError(
                        f'{cols_map[col]} (case sensitive) cannot be found in {dir}')
        for col, _ in cols_map2.items():
            if col not in header:
                raise ValueError(
                    f'{col} (case sensitive) cannot be found in {dir}')

    @classmethod
    def _check_median(cls, data, effect, null_value):
        """
        Checking if the median value of effects (beta, or) is reasonable

        Parameters:
        ------------
        data: a pd.Series of effects
        effect: BETA or OR
        null_value: 1 or 0

        """
        median_beta = np.nanmedian(data)
        if np.abs(median_beta - null_value > 0.1):
            raise ValueError((f"median value of {effect} is {round(median_beta, 4)} "
                              f"(should be close to {null_value}). "
                              "This column may be mislabeled"))
        else:
            cls.logger.info((f"Median value of {effect} is {round(median_beta, 4)}, "
                            "which is reasonable."))

    def get_zscore(self):
        """
        Computing z score from beta and se, and removing beta and se

        """
        self.z = self.beta / self.se
        self.beta = None
        self.se = None

    def extract_snps(self, keep_snps):
        """
        Extracting SNPs

        Parameters:
        ------------
        keep_snps: a pd.Series/DataFrame of SNPs

        """
        if isinstance(keep_snps, pd.Series):
            keep_snps = pd.DataFrame(keep_snps, columns=['SNP'])
        self.snpinfo['id'] = self.snpinfo.index  # keep the index in df
        self.snpinfo = keep_snps.merge(self.snpinfo, on='SNP')
        if self.z is None:
            self.beta = self.beta[self.snpinfo['id']]
            self.se = self.se[self.snpinfo['id']]
        else:
            self.z = self.z[self.snpinfo['id']]
        del self.snpinfo['id']

    def save(self, out):
        """
        Save the GWAS data

        Parameters:
        ------------
        out: prefix of output

        """
        pickle.dump({'beta': self.beta, 'se': self.se, 'z': self.z},
                    open(f'{out}.sumstats', 'wb'), protocol=4)
        self.snpinfo.to_csv(f'{out}.snpinfo', sep='\t',
                            index=None, na_rep='NA')

    def __eq__(self, other):
        if isinstance(other, GWAS):
            if not self.snpinfo.equals(other.snpinfo):
                return False
            if (not np.equal(self.z, other.z).all() or
                not np.equal(self.beta, other.beta).all() or
                    not np.equal(self.se, other.se).all()):
                return False
            return True
        return False


def run(args, log):
    args = check_input(args, log)
    cols_map, cols_map2 = map_cols(args)

    if args.ldr_gwas is not None:
        sumstats = GWAS.from_rawdata_ldr(args.ldr_gwas, cols_map, cols_map2,
                                         args.maf_min, args.info_min, args.fast_sumstats)
    elif args.y2_gwas is not None:
        sumstats = GWAS.from_rawdata_y2(args.y2_gwas, cols_map, cols_map2,
                                        args.maf_min, args.info_min)
    sumstats.save(args.out)

    log.info(
        f'Save the processed summary statistics to {args.out}.sumstats and {args.out}.snpinfo')

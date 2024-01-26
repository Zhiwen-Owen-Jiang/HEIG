import os
import argparse
import pickle
import time
import pandas as pd
import numpy as np
from parse import PlinkBIMFile, PlinkFAMFile, PlinkBEDFile
from utils import GetLogger, sec_to_str
from ldmatrix import LDmatrix, LDmatrixBED

MASTHEAD = "***************************************************************************\n"
MASTHEAD += "* Generate LD as a block diagonal matrix for imaging genetic data analysis\n"
MASTHEAD += "***************************************************************************"


def read_plink(dir):
    array_file, array_obj = f"{dir}.bed", PlinkBEDFile
    snp_file, snp_obj = f"{dir}.bim", PlinkBIMFile
    ind_file, ind_obj = f"{dir}.fam", PlinkFAMFile

    array_snps = snp_obj(snp_file)

    array_indivs = ind_obj(ind_file)
    n = len(array_indivs.IDList)

    geno_array = array_obj(array_file, n, array_snps)
    snp_getter = geno_array.nextSNPs
    array_snps.df['n'] = n

    return n, array_snps.df, snp_getter


def partition_genome(bim, part):
    num_snps_part = []
    end = -1
    bim['block'] = None
    bim['block_idx'] = None
    abs_begin = 0
    abs_end = 0
    for i in range(part.shape[0]):
        cand = list(bim.loc[bim['chr'] == part.iloc[i, 0], 'pos'])
        begin = end
        end = find_loc(cand, part.iloc[i, 2])
        if end < begin:
            begin = -1
        if end > begin:
            num_snps_part.append(end - begin)
            if not abs_begin and not abs_end:
                abs_begin = begin + 1
                abs_end = end + 1
            else:
                abs_begin = abs_end
                abs_end += end - begin
            bim.loc[bim.index[abs_begin: abs_end], 'block'] = i
            bim.loc[bim.index[abs_begin: abs_end], 'block_idx'] = range(end - begin)
        else:
            log.info(f"Block {i} has no SNP, skipped") 

    return num_snps_part, bim


def find_loc(num_list, target):
    l = 0
    r = len(num_list) - 1
    while l <= r:
        mid = (l + r) // 2
        if num_list[mid] == target:
            return mid
        elif num_list[mid] > target:
            r = mid - 1
        else:
            l = mid + 1
    return r


def check_partition(dir):
    try:
        header = open(dir).readline().split()
    except:
        log.info('ERROR: --partition should be an unzipped txt file')
    if header[0] == 'X' or header[0] == '23':
        raise ValueError('The X chromosome is not supported')
    if len(header) != 3:
        raise ValueError('The partition file should have three columns for CHR, START, and END (no headers)')
    for x in header:
        try:
            int(x)
        except:
            log.info('ERROR: the CHR, START, and END should be an integer')



def main(args, log):
    if args.ld_list:
        with open(dir, 'r') as file:
            ld_files = file.readlines().rstrip()
        if len(ld_files) <= 1:
            raise ValueError('At least two LD matrices should be provided to merge')

        log.info(f'Merging multiple LD matrix from {args.ld_list} ...')
        ld = LDmatrix(ld_files[0])
        ld.merge(ld_files[1:])

    elif args.bfile and args.partition:
        log.info(f"Reading genome partition info from {args.partition}")
        check_partition(args.partition)
        genome_part = pd.read_csv(args.partition, header=None, delim_whitespace=True)
        log.info(f"There are {genome_part.shape[0]} genome blocks to partition")

        log.info(f"Reading bfile from {args.bfile} ...")
        _, bim, snp_getter = read_plink(args.bfile)

        log.info(f"Doing genome partition ...")
        num_snps_part, ld_info = partition_genome(bim, genome_part)
        log.info(f"There are {sum(num_snps_part)} SNPs partitioned into {len(num_snps_part)} blocks")

        log.info('Making an LD matrix ...')
        ld = LDmatrixBED(num_snps_part, ld_info, snp_getter)
    else:
        raise ValueError('Either --ld-list or --bfile + --partition should be specified')
    
    ld.save(args.out)
    log.info(f"Save LD matrix to {args.out}_ld.dat")
    log.info(f"Save LD matrix info to {args.out}_ld_info.txt")



parser = argparse.ArgumentParser()
parser.add_argument('--partition', help='directory to genome partition file \
                    (a tab deliminated file without header)')
parser.add_argument('--bfile', help='directory to bfile (prefix)')
parser.add_argument('--ld-list', help='a txt file with one prefix of LD matirx file at one row')
parser.add_argument('--out', required=True, help='output prefix')


if __name__ == '__main__':
    args = parser.parse_args()

    logpath = os.path.join(f"{args.out}_ld_matrix.log")
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
    finally:
        log.info(f"Analysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")



    


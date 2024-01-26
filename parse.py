import pandas as pd
import numpy as np
import bitarray as ba


def get_compression(fh):
    if fh.endswith('gz'):
        compression = 'gzip'
    elif fh.endswith('bz2'):
        compression = 'bz2'
    else:
        compression = None

    return compression


def __ID_List_Factory__(colnames, keepcol, id_dtypes, fname_end, header=None, usecols=None):
    
    class IDContainer():
        def __init__(self, fname):
            self.__usecols__ = usecols
            self.__colnames__ = colnames
            self.__keepcol__ = keepcol
            self.__fname_end__ = fname_end
            self.__header__ = header
            self.__id_dtypes = id_dtypes
            self.__read__(fname)
            self.n = len(self.df)

        def __read__(self, fname):
            end = self.__fname_end__
            if end and not fname.endswith(end):
                raise ValueError(f"{fname} must end in {end}")

            comp = get_compression(fname)
            self.df = pd.read_csv(fname, header=self.__header__, usecols=self.__usecols__,
                                  delim_whitespace=True, compression=comp, dtype=self.__id_dtypes)

            if self.__colnames__:
                self.df.columns = self.__colnames__

            if self.__keepcol__ is not None:
                self.IDList = self.df.iloc[:, self.__keepcol__].astype('object')

    return IDContainer

PlinkBIMFile = __ID_List_Factory__(['chr', 'snp', 'cm', 'pos', 'a0', 'a1'], 1, {1: 'string'}, \
                                    '.bim', usecols=[0, 1, 2, 3, 4, 5])
PlinkFAMFile = __ID_List_Factory__(['fid', 'iid', 'sex'], [0, 1], {0: 'string', 1: 'string'}, \
                                    '.fam', usecols=[0, 1, 4])



class __GenotypeArrayInMemory__():
    '''
    Parent class for various classes containing inferences for files with genotype
    matrices, e.g., plink .bed files, etc
    '''
    def __init__(self, fname, n, snp_list):
        self.m = len(snp_list.IDList)
        self.n = n
        self._currentSNP = 0
        (self.nru, self.geno) = self.__read__(fname, self.m, n)
    
    def __read__(self, fname, m, n):
        raise NotImplementedError


class PlinkBEDFile(__GenotypeArrayInMemory__):
    '''
    Interface for Plink .bed format
    '''
    def __init__(self, fname, n, snp_list):
        self._bedcode = {
            2: ba.bitarray('11'),
            np.nan: ba.bitarray('10'),
            1: ba.bitarray('01'),
            0: ba.bitarray('00')
        }

        __GenotypeArrayInMemory__.__init__(self, fname, n, snp_list)

    def __read__(self, fname, m, n):
        if not fname.endswith('.bed'):
            raise ValueError('.bed filename must end in .bed')

        fh = open(fname, 'rb')
        magicNumber = ba.bitarray(endian='little')
        magicNumber.fromfile(fh, 2)
        bedMode = ba.bitarray(endian='little')
        bedMode.fromfile(fh, 1)
        e = (4 - n % 4) if n % 4 != 0 else 0
        nru = n + e
        self.nru = nru

        # check magic number
        if magicNumber != ba.bitarray('0011011011011000'):
            raise IOError('Magic number from PLINK .bed file not recognized')
        
        if bedMode != ba.bitarray('10000000'):
            raise IOError('Plink .bed file must be in default SNP-major mode')

        # check file length
        self.geno = ba.bitarray(endian='little')
        self.geno.fromfile(fh)
        self.__test_length__(self.geno, self.m, self.nru)
        return (self.nru, self.geno)

    def __test_length__(self, geno, m, nru):
        exp_len = 2*m*nru
        real_len = len(geno)
        if real_len != exp_len:
            raise IOError(f"Plink .bed file has {real_len} bits, expected {exp_len}")

    def nextSNPs(self, num):
        '''
        Unpacks the binary array of genotypes and returns an n x 1 matrix of 
        genotypes for the next SNP, where n := number of samples.
        '''

        snps = np.zeros((self.n, num))
        for i in range(num):
            if self._currentSNP + 1 > self.m:
                raise ValueError(f"No more SNPs remaining")

            slice = self.geno[2*self._currentSNP*self.nru : 2*(self._currentSNP+1)*self.nru]
            X = np.array(slice.decode(self._bedcode), dtype=float)
            X = X[0:self.n]
            snps[:, i] = X
            self._currentSNP += 1

        return snps

    def gen_SNPs(self):
        for c in range(self.m):
            slice = self.geno[2*c*self.nru : 2*(c+1)*self.nru]
            X = np.array(slice.decode(self._bedcode), dtype=float)
            X = X[0:self.n]

            yield c, X
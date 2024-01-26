import argparse, time, os
import pandas as pd
from utils import GetLogger, sec_to_str


MASTHEAD = "*********************************************************************************\n"
MASTHEAD += "* Generate VTK file for visualizing the results of imaging genetic data analysis\n"
MASTHEAD += "********************************************************************************"


def main(args, log):
    # res = np.load(args.res)
    res_files = [file for file in os.listdir(args.res) if file.endswith('gene_cor_y2.txt') or file.endswith('heri.txt')]
    for file in res_files:
        res_file = pd.read_csv(os.path.join(args.res, file), sep='\t')
        if hasattr(res_file, 'heri'):
            res = res_file['heri']
        elif hasattr(res_file, 'image_y2_gc'):
            res = res_file['image_y2_gc']

        with open(args.temp, 'r') as input:
            tempB = input.readlines()

        res_file_name = args.res.split('/')[-1]
        tempB.append(f"SCALARS {res_file_name} float\n")
        tempB.append(f"LOOKUP_TABLE {res_file_name}\n")
        
        for num in res:
            tempB.append(f"{str(num)}\n")
        
        out_file = os.path.join(args.out, file[:-4]) # remove the .txt suffix
        with open(f"{out_file}.vtk", 'w') as output:
            output.writelines(tempB)
             
        log.info(f"Write the vtk file for {file} to {out_file}.vtk")
    

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--res', help='directory to the results to generate vtk')
parser.add_argument('--temp', help='directory to template')
parser.add_argument('-o', '--out', help='directory to save the vtk file' )

if __name__ == '__main__':
    args = parser.parse_args()
    
    logpath = f"{args.out}_vtk.log"
    log = GetLogger(logpath)

    log.info(MASTHEAD)
    log.info("Parsed arguments")
    for arg in vars(args):
        log.info(f'--{arg} {getattr(args, arg)}')

    start_time = time.time()
    try:
        main(args, log,)
    finally:
        log.info(f"Analysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")






    
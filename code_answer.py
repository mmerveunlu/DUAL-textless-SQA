"""
This script adds two columns to the given dataframe.
The columns are: code_start and code_end. They are collected after feature extraction.
Args:
    input: str, the path of the input csv file
    code: str, the path of the input code folder
    output: str, the path of the output csv file
Usage:
> python code_answer.py --input train-answer-span.csv \
                        --code train_code/ \
                        --output train-final.csv
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

FORMAT = "%(asctime)s - %(name)s - %(levelname)s -[%(filename)s:%(lineno)s - %(funcName)20s() ]- %(message)s"
logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format=FORMAT,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)


def add_code_start_end(input_path, code_input_path, output):
    """
    Reads dataframe containing start/end time frames and add two columns
    for code start and end of the answer.
    Args:
        :param input_path, str, the path of the csv file
        :param code_input_path, str, the path of the folder that contains code and cnt files
        :param output, str, the path of the output file
    Returns
      None
    """
    logger.info("Reading input file %s " % input_path)
    df = pd.read_csv(input_path)

    code_start = []
    code_end = []
    for i, row in tqdm(df.iterrows()):
        fname = os.path.join(code_input_path, 'context-' + row['context_id'] + '.cnt')
        if os.path.exists(fname):
            context_cnt = np.loadtxt(fname)
            start_ind = row['new_start'] / 0.02
            end_ind = row['new_end'] / 0.02
            context_cnt_cum = np.cumsum(context_cnt)

            new_start_ind, new_end_ind = None, None
            prev = 0
            for idx, cum_idx in enumerate(context_cnt_cum):
                if cum_idx >= start_ind and new_start_ind is None:
                    if abs(start_ind - prev) <= abs(cum_idx - start_ind):
                        new_start_ind = idx - 1
                    else:
                        new_start_ind = idx
                if cum_idx >= end_ind and new_end_ind is None:
                    if abs(end_ind - prev) <= abs(cum_idx - end_ind):
                        new_end_ind = idx - 1
                    else:
                        new_end_ind = idx
                prev = cum_idx
            if new_start_ind is None:
                new_start_ind = idx
            if new_end_ind is None:
                new_end_ind = idx

            code_start.append(new_start_ind)
            code_end.append(new_end_ind)
            if i%100 == 0:
                logger.info("Processed %d numbers of example" % i)
        else:
            code_start.append(-1)
            code_end.append(-1)

    df['code_start'] = code_start
    df['code_end'] = code_end

    df.to_csv(output)
    logger.info("Resulting dataframe saved into %s " % output)


def parse_args():
    """ Argument parser"""
    parser = argparse.ArgumentParser(description='Generate meta-file for given tsv file')
    parser.add_argument('--input',
                        help='input folder that contains TextGrid files',
                        required=True)
    parser.add_argument('--code',
                        help='input json files that contains squad-format data',
                        required=True)
    parser.add_argument('--output',
                        help='Output file name to save resulting csv',
                        required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info("Starting the script using input file %s" % args.input)
    add_code_start_end(args.input, args.code, args.output)


if __name__ == "__main__":
    main()

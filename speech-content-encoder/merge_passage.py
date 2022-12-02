"""
This script merges the context codes/cnt files into one.
The codes and cnt files should be generated previously using S2U_train_dev.py of S2U_text.py scripts.
Args:
    segment: str, the path of the segment id dictionary
       Its keys are the context ids, and values are the part ids.
    data: str, the path of the data dir, that contains code/cnt files
    output: str, the path of the output file

Usage:
> python merge_passage.py --segment train_segment_id.json --data train_code --output train_code
"""
import argparse
import json
import logging
from os import path

import numpy as np
from tqdm import tqdm

SEPARATOR = "_"

FORMAT = "%(asctime)s - %(name)s - %(levelname)s -[%(filename)s:%(lineno)s - %(funcName)20s() ]- %(message)s"
logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format=FORMAT,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)


def merge_passage(segment_file, data_dir, output_dir):
    """
    This function merges the passage codes into one.
    :param segment_file, str
    :param data_dir, str
    :param output_dir, str
    Returns:

    """
    logging.info("Opening segment-id file")
    with open(segment_file, 'r') as f:
        segment_dict = json.load(f)
    ind = 0

    for passage, segment_list in tqdm(segment_dict.items()):
        for idx, idy in enumerate(segment_list):
            # opens the code and cnt files
            code = np.loadtxt(path.join(data_dir, "".join(['context-', str(passage), SEPARATOR, str(idy), '.code'])))
            cnt = np.loadtxt(path.join(data_dir, "".join(['context-', str(passage), SEPARATOR, str(idy), '.cnt'])))
            if idx == 0:
                # if id is 0, then it is the starting utterance
                merge_passage = code
                merge_cnt = cnt
            else:
                # if id is not 0, then they are in the middle
                try:
                    merge_passage = np.concatenate([merge_passage, code], axis=-1)
                    merge_cnt = np.concatenate([merge_cnt, cnt], axis=-1)
                except:
                    print(f'passage: {passage} len {merge_passage.shape[-1]}')
                    code = np.array([code])
                    cnt = np.array([cnt])
                    merge_passage = np.concatenate([merge_passage, code], axis=-1)
                    merge_cnt = np.concatenate([merge_cnt, cnt], axis=-1)

        output_code = path.join(output_dir, "".join(['context-', str(passage), '.code']))
        output_cnt = path.join(output_dir, "".join(['context-', str(passage), '.cnt']))
        np.savetxt(output_code, merge_passage, fmt='%i')
        np.savetxt(output_cnt, merge_cnt, fmt='%i')
        ind += 1
        if ind % 100 == 0:
            logging.info("%d numbers of examples are merged" % i)


def parse_args():
    """ Argument parser"""
    parser = argparse.ArgumentParser(description='Merging passage codes into one')
    parser.add_argument('--segment',
                        help='segment file path',
                        required=True)
    parser.add_argument('--data',
                        help='data dir path contains code/cnt files',
                        required=True)
    parser.add_argument('--output',
                        help='Output file name',
                        required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    merge_passage(args.segment, args.data, args.output)
    logging.info("Merging is done.")
    return


if __name__ == "__main__":
    main()

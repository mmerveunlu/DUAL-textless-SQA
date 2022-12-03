"""
This script finds the answer spans in text grid files and finds the new start/end point in the context.
Args:
    input: str, the input folder that contains textgrid files
    data: str, the path of json file, should be SQuAD-like format
    meta_data: str, the path of the meta data as csv format
    output: str, the path of the output file

Usage:
> python alignment.py --input textGrid/ \
                      --data train-v1.1.json \
                      --meta-data meta-train.csv
                      --output train_answer_span.csv
"""
import argparse
import json
import logging
import os

import pandas as pd
import textgrid
from tqdm import tqdm
from utils import text_preprocess

FORMAT = "%(asctime)s - %(name)s - %(levelname)s -[%(filename)s:%(lineno)s - %(funcName)20s() ]- %(message)s"
logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format=FORMAT,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)


def find_timespan(tg_file, answer):
    """
    :param tg_file, str
    :param answer, str
    Returns
       float, start time of the answer
       float, end time of the answer
    """
    # opens textgrid file
    try:
        tg = textgrid.TextGrid.fromFile(tg_file)
    except:
        return [0], [0]
    # gets the normalized words from the file
    words = [text_preprocess(tg[0][i].mark) for i in range(len(tg[0])) if tg[0][i].mark != '']
    # gets the normalized words from the file including empty characters
    words_SIL = [text_preprocess(tg[0][i].mark) for i in range(len(tg[0]))]

    pos_map = [i for i, word in enumerate(words_SIL) if word != '']
    # sentence contains the context
    sentence = ' '.join(words)

    # finds the starting point of the answer in the sentence
    match_idxs = [i for i in range(len(sentence)) if sentence.startswith(answer, i)]

    # if no match, then return 0,0
    if len(match_idxs) == 0:
        return [0], [0]
    else:
        # if match is found, then find start and end time spans
        start_times, end_times = [], []
        for match_idx in match_idxs:
            match_span = sentence[match_idx:match_idx + len(answer)]

            span = match_span.split()
            res = [words[idx:idx + len(span)] == span for idx in range(len(words))]
            try:
                index = res.index(True)
            except:
                return [0], [0]
            start_times.append(tg[0][pos_map[index]].minTime)
            end_times.append(tg[0][pos_map[index + len(span) - 1]].maxTime)

        return start_times, end_times


def create_timespan_file(input_folder, answers_file):
    """
    This function creates a dataframe that contains time spans of the answers.
    :param input_folder, str, the path of the input textgrid files
    :param answers_file, str, original dataset json file
    Returns
       pandas.DataFrame, with columns ['hash','text','utterance','start','end']
    """
    logger.info("Opening the original data from %s" % answers_file)
    # get the original dataset
    with open(answers_file) as fp:
        original_data = json.load(fp)['data']
    # get a dict where keys are question ids and values are answer text
    question2ans = []
    for i, art in enumerate(original_data):
        for j, p in enumerate(art['paragraphs']):
            for qa in p['qas']:
                question2ans.append({"id": qa['id'], "text": qa['answers'][0]['text'],
                                     "context": "context-" + str(i) + "_" + str(j) + "_"})
    logger.info("Total number of question is %d " % len(question2ans))
    # first create a dict from original json
    data_dict = []
    input_files = sorted(os.listdir(input_folder))
    for i, qa in tqdm(enumerate(question2ans)):
        textfiles = [f for f in input_files if f.startswith(qa['context'])]
        # search each file
        for f in textfiles:
            s, e = find_timespan(os.path.join(input_folder, f), text_preprocess(qa['text']))
            # if answer is found in the file, then add it to dict
            if s != [0] and e != [0]:
                data_dict.append({"hash": qa['id'],
                                  "text": qa['text'],
                                  "utterance": f.replace(".TextGrid", ""),
                                  "start": s[0],
                                  "end": e[0]})
        if i % 100 == 0:
            logger.info("Processed %d numbers of example" % i)

    # save the resulting file as csv
    df = pd.DataFrame.from_dict(data_dict)
    return df


def add_columns(answer_span_df, meta_data, output_file):
    """
    This function adds columns to the answer span dataframe.
      added columns: context_segment_id, context_id, new_start, new_end, new_utterance
    context_segment_id: the order of the utterance in the context
    context_id: the context that contains all the utterances
    new_start: the starts of the answer in the context, equals the sum of the previous parts and start in the utterance
    new_end: the ending of the answer in the context, equals the sum of the previous parts and end in the utterance
    Args:
        :param answer_span_df, DataFrame
        :param meta_data, str, the path of the metadata
        :param output_file, str, the path of the output file
    Returns:
       None
    """
    # opens metadata file
    meta_data = pd.read_csv(meta_data)

    logger.info("Context segment id added")
    # Adding context_segment_id
    answer_span_df["context_segment_id"] = answer_span_df.utterance.str.split("_", expand=True).iloc[:, 2]
    logger.info("Context segment id is added")
    # Adding context id
    cols = answer_span_df.utterance.str.split("-", expand=True).iloc[:, 1].str.split("_", expand=True)
    logger.info("Context id is added")
    answer_span_df["context_id"] = cols[0] + "_" + cols[1]
    # Adding new_start and new_end
    new_starts = []
    new_ends = []
    for index, row in answer_span_df.iterrows():
        # get the previous parts
        prev = ["context-" + row['context_id'] + "_" + str(f) for f in range(int(row['context_segment_id']))]
        # get the sum of the durations
        filter_in = meta_data['id'].isin(prev)
        prev_sum = sum(meta_data[filter_in].duration)
        new_starts.append(round(row['start'] + prev_sum, 3))
        new_ends.append(round(row['end'] + prev_sum, 3))
    # add them to df
    answer_span_df['new_start'] = new_starts
    answer_span_df['new_end'] = new_ends
    logger.info("New start/end is added")

    # Apparently new_utterance is the same as utterance
    answer_span_df['new_utterance'] = answer_span_df['utterance']

    answer_span_df.to_csv(output_file, sep=",", header=True, index=False)
    logger.info("Resulting data is saved to %s " % output_file)


def parse_args():
    """ Argument parser"""
    parser = argparse.ArgumentParser(description='Generate meta-file for given tsv file')
    parser.add_argument('--input',
                        help='input folder that contains TextGrid files',
                        required=True)
    parser.add_argument('--data',
                        help='input json files that contains squad-format data',
                        required=True)
    parser.add_argument('--meta',
                        help='meta data dataframe as csv format',
                        required=True)
    parser.add_argument('--output',
                        help='Output file name to save resulting csv',
                        required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info("Starting the script using input file %s" % args.input)
    df = create_timespan_file(args.input, args.data)
    logger.info("Answer spans are found in each utterance")
    add_columns(df, args.meta, args.output)
    logger.info("New columns are added")


if __name__ == "__main__":
    main()

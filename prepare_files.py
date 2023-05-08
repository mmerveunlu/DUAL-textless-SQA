import argparse
import json
import logging
import os
import pandas as pd

FORMAT = "%(asctime)s - %(name)s - %(levelname)s -[%(filename)s:%(lineno)s - %(funcName)20s() ]- %(message)s"
logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format=FORMAT,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)


def save_meta_csv(dt, out_file):
    """ saves the given dt into a file
    Args:
        :param dt, pandas DataFrame
        :param out_file, str, the path of the output csv file
    """
    dt.to_csv(out_file, sep=",", header=True, index=False)
    logger.info("Resulting file saved into %s " % out_file)


def create_meta_files(tsv_data, meta_all_train, main_hash2question, output_folder):
    """
    Returns

    """
    context_paths = tsv_data.context_path.unique()
    # clear mp3 and add _
    context_paths = [c.replace(".mp3", "_") for c in context_paths]
    # add questions
    question_paths = tsv_data.question_path.unique()
    # clear mp3
    question_paths = [q.replace(".mp3", "") for q in question_paths]
    cols = meta_all_train.id.str.split("_", expand=True)
    meta_all_train['context_path'] = cols[0] + "_" + cols[1] + "_"
    meta_train = meta_all_train[
        (meta_all_train.context_path.isin(context_paths)) | (meta_all_train.id.isin(question_paths))]

    # drop context_path folder
    meta_train = meta_train.drop(['context_path'], axis=1)
    # re-index
    meta_train = meta_train.reset_index()
    # save meta_train
    save_meta_csv(meta_train, os.path.join(output_folder, "meta_train.csv"))
    logger.info("Meta data is saved: %s" % output_folder)

    # create segment json file
    contexts = meta_train[meta_train.id.str.startswith("context-")]
    cols = contexts.id.str.split("_", expand=True)
    cols[0] = cols[0].str.split("-", expand=True)[1]
    cols[0] = cols[0] + "_" + cols[1]
    cols = cols.drop([1], axis=1)
    grouped_cols = cols.groupby(0).agg(lambda x: ', '.join(tuple(x.tolist())))
    group_dict = grouped_cols.to_dict()
    segments = {k: v.split(",") for k, v in group_dict[2].items()}

    with open(os.path.join(output_folder, "data_segment_id.json"), "w+") as fp:
        json.dump(segments, fp)
    logger.info("Segments data is saved: %s" % output_folder)

    # create hash2question file
    with open(main_hash2question) as fp:
        original_hash = json.load(fp)
    hash2question_part = {k: v for k, v in original_hash.items() if v in question_paths}
    with open(os.path.join(output_folder, "hash2question.json"), "w+") as fp:
        json.dump(hash2question_part, fp)
    logger.info("Hash2Question data is saved: %s" % output_folder)

    # create lab files for each audio
    for i, row in meta_train.iterrows():
        with open(os.path.join(output_folder, "lab_files", row['id'] + ".lab"), "w+") as fp:
            fp.writelines(row['text'])
    logger.info("Lab-text files are saved: %s" % output_folder)


def create_answer_code_file(answer_code_all, tsv_file, output_file):
    """
    It creates a csv file that contains a subset from all data using tsv file.
    This file is used directly in train function.
    Args:
    :param answer_code_all, str, the path of the file for all data
    :param tsv_file, str, the path of the file that contains subset data
    :param output_file, str, the path of the output file
    Returns:
      None
    """
    answer_all = pd.read_csv(answer_code_all)
    data_tsv = pd.read_csv(tsv_file, sep="\t")

    # get the part that contains only the question ids
    train_code_answer_part = answer_all[answer_all.hash.isin(data_tsv.questionid.unique())]

    tsv_dict = data_tsv[['questionid', 'answer_text', 'answer_start', 'answer_end']].to_dict(orient="records")
    # tsv dict contains question ids and answer text
    tsv_dict = {k['questionid']: [k['answer_text'], k['answer_start'], k['answer_end']] for k in tsv_dict}

    # now choose the samples with the question id and text same as the tsv_dict
    # let s think a for loop
    indx = []
    for i, row in train_code_answer_part.iterrows():
        if row['hash'] in tsv_dict.keys():
            d = tsv_dict[row['hash']]
            if row['text'] == d[0] and round(row['new_start'], 3) == round(d[1], 3) \
                    and round(row['new_end'], 3) == round(d[2], 3):
                indx.append(row[0])
    sub_train_final = train_code_answer_part[train_code_answer_part.index.isin(indx)]
    sub_train_final = sub_train_final.drop(['Unnamed: 0'], axis=1)
    sub_train_final = sub_train_final.reset_index()
    sub_train_final.drop(['index'], axis=1, inplace=True)
    # save the resulting file
    sub_train_final.to_csv(output_file)
    return


def parse_args():
    """ Argument parser"""
    parser = argparse.ArgumentParser(description='Generate meta-file for given tsv file')
    parser.add_argument('--input',
                        help='input tsv/json file name')
    parser.add_argument('--meta_file',
                        help='meta file contains all meta info')
    parser.add_argument('--answer_file',
                        help='answer file contains final answer info')
    parser.add_argument('--hash_file',
                        help='hash file contains all hash-question info')
    parser.add_argument('--tsv_file',
                        help='tsv file contains subset data')
    parser.add_argument('--output',
                        help='str, output folder')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.meta_file:
        logger.info("Starting the script using input file %s" % args.input)
        data_tsv = pd.read_csv(args.input, sep="\t")
        main_meta_all = pd.read_csv(args.meta_file)
        create_meta_files(data_tsv, main_meta_all, args.hash_file, args.output)
    if args.answer_file:
        logger.info("Starting to create sub-answer file")
        create_answer_code_file(args.answer_file, args.tsv_file, args.output)
    return


if __name__ == "__main__":
    main()

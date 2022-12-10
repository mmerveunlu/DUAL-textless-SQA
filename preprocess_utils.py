"""
This script generates necessary files for preprocessing and training the dataset.
It takes a json file (formatted as SQuAD-train) and audio folder.
Outputs are three files and text files for each audios
  * meta-data.csv: contains id, speaker, duration, context, normalized context
  * data_segment_id.json: contains paragraphs indices and utterance indices as "0_0":{"0","1","2"}
  * hash2question.json: contains question ids from original data and ids from generated files as "hash1":"question-0_0_0"
  * [*].lab: contains text files related to each audio

Inputs:
  input: str, the path of the original json/tsv file
  audio: str, the path of the audio files
  audio_format: str, the extension of audio files as mp3, wav
  format: str, the extension of the original file as tsv of json
  output: str, the path of the output folder
  debug: bool, set True if the generated files are checked.
Usage:
> python preprocess_utils.py --input original_train.json
                             --audio audios/ \
                             --audio_format wav \
                             --format json
                             --output train/ \
"""

import argparse
import json
import logging
from os import path, listdir
import re

import librosa
from mutagen.mp3 import MP3
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm


from utils import text_preprocess

FORMAT = "%(asctime)s - %(name)s - %(levelname)s -[%(filename)s:%(lineno)s - %(funcName)20s() ]- %(message)s"
logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format=FORMAT,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)


def get_length(audio):
    """returns the length of the audio in seconds"""
    sound = AudioSegment.from_file(audio)
    return round((sound.duration_seconds % 60), 3)


def create_text_files(data, output):
    """ The function creates text files related to each audio file
    Args:
        :param data, pandas DataFrame, contains id, text, normalized text and speaker
        :param output, str, the path of the output folder
    """
    for i, row in data.iterrows():
        with open(path.join(output, row['id'] + ".lab"), "w+") as fp:
            fp.writelines(row['text'])


def generate_meta_segment_hash_file(original_data, audio_dir, output_folder, format="wav"):
    """
    This function creates
       * meta file that contains id, speaker, duration, text and normalized text
       * segment file that contains the index of sentences and the context.
       * hash-to-question file that contains question ids to question index
    It uses json file and audio files,
        tokenizes the paragraphs into sentences (utterances)
        gets the duration of related audio file.
        saves three files to the given output folder.
    Args
      :param original_data, dict of data, SQuAD formatted
      :param audio_dir, str, the path of the audio files, files naming as context-X_X_X.wav
      :param output_folder, str, the path of the output folder
      :param format, str, the audio format as wav, mp3
    """
    # meta dict is an array of dict.
    # each element is a dict with keys: "id","speaker","duration","text","normalized_text"
    meta_dict = []
    meta_dict_err = []
    segments = {}
    # get the original dict
    logger.info("Opening the original data file from %s " % original_data)
    with open(original_data) as fp:
        original_data = json.load(fp)['data']

    audio_files = sorted([f for f in sorted(listdir(audio_dir)) if f.endswith("."+format)])
    logger.info("Number of audio files in %s : %d "% (audio_dir, len(audio_files)))
    # getting the files anmes only grouped with passages
    context_names = list(set(["_".join(f.split("_")[0:2]) for f in audio_files]))
    logger.info("Number of audio files by passages %d "% len(audio_files))

    for i, ctx in enumerate(tqdm(context_names)):
        # for each context first find all audio files
        ctx_audios = [f for f in audio_files if f.startswith((ctx))]
        indices = re.findall(r'\d+', ctx)
        context = original_data[int(indices[0])]['paragraphs'][int(indices[1])]['context']
        sentences = re.split("[.?!]( )+", context)
        sentences = [s for s in sentences if not (s == " ")]
        for j, file in enumerate(ctx_audios):
            # indices = file.replace("context-", "").split(".")[0].split("")
            # get the duration of the  audio file
            duration = get_length(path.join(audio_dir, file))
            seg_key = indices[0] + "_" + indices[1]
            if not (seg_key in segments):
                segments[seg_key] = []
            segments[seg_key].append(file.split("_")[-1])

            meta_dict.append({"id": file.replace("."+format, ""),
                              "speaker": "Google-TTS",
                              "text": sentences[j],
                              "normalized_text": text_preprocess(sentences[j]),
                              "duration": duration})
        if i % 100 == 0:
            logger.info("Processed %d numbers of file " % i)
    # saving meta-[part].csv
    meta_df_train = pd.DataFrame.from_dict(meta_dict)
    # save meta info
    save_meta_csv(meta_df_train, path.join(output_folder, "meta_data.csv"))
    logger.info("Meta data is saved: %s" % output_folder)

    # save segment info
    with open(path.join(output_folder, "data_segment_id.json"), "w+") as fp:
        json.dump(segments, fp)
    logger.info("Segments data is saved: %s" % output_folder)

    # save hash2question
    generate_hash2question(original_data, output_folder)
    logger.info("Hash2Question data is saved: %s" % output_folder)

    # create text files inn the audio folder
    create_text_files(meta_df_train, audio_dir)
    logger.info("Text files are saved into %s", audio_dir)


def generate_hash2question(original_data, output_folder):
    """
    This function takes a dict file and generates hash2question dictionary.
    The keys are the hash ids from dict file, and the values are the question indices.
    Ex:   "038v87hfbv": "question-0_0_0"
    Args
        :param original_data, dict object, SQuAD formatted
        :param output_folder, str, the path of the output folder
    Returns
       None
    """
    hash2question = {}
    for i, a in enumerate(original_data):
        for k, p in enumerate(a['paragraphs']):
            for j, qa in enumerate(p['qas']):
                hash2question[qa['id']] = "question" + "-" + "_".join([str(i), str(k), str(j)])
    # print saving hash2question
    with open(path.join(output_folder, "hash2question.json"), "w+") as fp:
        json.dump(hash2question, fp)


def check_files(original_data, audio_dir, output_folder, format):
    """
    It simply check the number of occurences in the generated files.
    Args:
        :param original_data, str, the path of the json file, SQuAD formatted
        :param audio_dir, str, the path of the audio files
        :param output_folder, str, the path of the output contains all generated files
        :param format, str, audio files extension as wav, mp3
    Returns:
        None
    """
    # Simple check for the files
    # get the original dict
    logger.info("Checking files")
    with open(original_data) as fp:
        original_data = json.load(fp)['data']
    # the number of rows in meta must be the same as the number of wav/mp3 files in audio
    audios = [f for f in listdir(audio_dir) if f.endswith(format)]
    meta_data = pd.read_csv(path.join(output_folder, "meta_data.csv"))
    assert (meta_data.shape[0] == len(audios))
    logger.info("Check on meta data rows: OK.")
    # Number of keys in segments must be the same as the number of paragraphs in the original data
    nbr_a = len(original_data)
    nbr_p = 0
    nbr_q = 0
    for a in original_data:
        nbr_p += len(a['paragraphs'])
        for p in a['paragraphs']:
            nbr_q += len(p['qas'])
    with open(path.join(output_folder, "data_segment_id.json")) as fp:
        segments = json.load(fp)
    assert(nbr_p == len(segments))
    logger.info("Check on segments keys: OK.")

    # Number of values in segments must be the same as the number of rows of the meta = (nbr of audio)
    segments_value = sum([len(s) for k, s in segments.items()])
    assert (segments_value == meta_data.shape[0])
    logger.info("Check on segments values: OK.")

    # Number of elements in hash2question must be the same as the number of question in the original data
    with open(path.join(output_folder, "hash2question.json")) as fp:
        hash2questions = json.load(fp)
    assert(len(hash2questions) == nbr_q)
    logger.info("Check on hash2question: OK.")


def preprocess_data_from_tsv(text_file, audio_dir):
    """
    takes the data and generates rows
    The function assumes that input paragraphs are already split into sentence.
    :param text_file, str, should be in tsv format
       each line starts with the path followed by the sentence
    :param audio_dir, str, should contain mp3/wav files
    """
    # read tsv file that contains sentences
    dt = pd.read_csv(text_file, sep="\t")
    # Normalize sentences
    dt['normalized_text'] = dt['sentence'].apply(lambda x: text_preprocess(x))

    # get the duration of each audio file
    audios = dt['path'].to_list()
    durations = []
    for faudio in tqdm(audios):
        if faudio.endswith(".mp3"):
            audio = MP3(path.join(audio_dir, faudio))
            # find the duration of the audio
            duration = audio.info.length
        elif faudio.endswith(".wav"):
            duration = librosa.get_duration(filename=path.join(audio_dir, faudio))
        durations.append(duration)
    dt['duration'] = durations
    dt['speaker'] = ['Google-TTS'] * len(durations)
    dt['id'] = dt['path'].apply(lambda x: "context-" + x.replace('.wav', ""))
    if 'Unnamed: 0' in dt:
        dt.drop(columns=['Unnamed: 0'], inplace=True)
    if 'path' in dt:
        dt.drop(columns=['path'], inplace=True)
    dt = dt.sort_values(by=['id'])
    return dt


def save_meta_csv(dt, out_file):
    """ saves the given dt into a file
    Args:
        :param dt, pandas DataFrame
        :param out_file, str, the path of the output csv file
    """
    dt.to_csv(out_file, sep=",", header=True, index=False)
    logger.info("Resulting file saved into %s " % out_file)


def parse_args():
    """ Argument parser"""
    parser = argparse.ArgumentParser(description='Generate meta-file for given tsv file')
    parser.add_argument('--input',
                        help='input tsv/json file name',
                        required=True)
    parser.add_argument('--audio',
                        help='input audio folder',
                        required=True)
    parser.add_argument("--audio_format",
                        help="audio files format ex: mp3, wav",
                        default="wav")
    parser.add_argument('--format',
                        help='the format of the input ex: json, tsv',
                        default="json")
    parser.add_argument('--output',
                        help='Output file/folder name',
                        required=True)
    parser.add_argument('--debug',
                        help='Set True if simple check is applied',
                        action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info("Starting the script using input file %s" % args.input)
    if args.format == "tsv":
        logger.info("Running with tsv file")
        dt = preprocess_data_from_tsv(args.input, args.audio)
        logger.info("Saving the meta file into %s " % args.output)
        save_meta_csv(dt, args.output)
    elif args.format == "json":
        logger.info("Running with json file")
        generate_meta_segment_hash_file(args.input, args.audio, args.output, args.audio_format)
    else:
        print("The format of the input file must be given as json or tsv")
    if args.debug:
        logger.info("Checking the generated files")
        check_files(args.input, args.audio, args.output, args.audio_format)
    return


if __name__ == "__main__":
    main()

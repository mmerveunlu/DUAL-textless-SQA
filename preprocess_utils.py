"""
This script reads tsv file and generates meta file in csv format.
The output meta file contains following columns:
   - id: the name of the sentence, unique for each row has template as [context|question]-[0-9]+_[0-9]+_[0-9]+
   - speaker: the name of the speaker
   - duration: the duration of the audio file
   - text: string, the content as sentence
   - normalized_text: string, normalized text

Usage:
>> python preprocess_utils.py --input train.tsv --audio clips --output meta-train.tsv
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
    Args:
        :param original_data, dict of data, SQuAD formatted
        :param audio_dir, str, the path of the audio files, files naming as context-X_X_X.wav
        :param output_folder, str, the path of the output folder
        :param format, str, the audio format as wav, mp3
    Returns:
      None
    """
    # meta dict is an array of dict.
    # each element is a dict with keys: "id","speaker","duration","text","normalized_text"
    meta_dict = []
    meta_dict_err = []
    segments = {}
    # get the original dict
    with open(original_data) as fp:
        original_data = json.load(fp)['data']

    audio_files = [f for f in sorted(listdir(audio_dir)) if f.endswith("."+format)]

    for file in tqdm(audio_files):
        # for each audio file
        # indices = file.replace("context-", "").split(".")[0].split("")
        indices = re.findall(r'\d', file)
        # get the duration of the sentence/utterance
        duration = get_length(path.join(audio_dir, file))
        seg_key = indices[0] + "_" + indices[1]
        if not (seg_key in segments):
            segments[seg_key] = []

        context = original_data[int(indices[0])]['paragraphs'][int(indices[1])]['context']
        try:
            # sentences = tokenizer.tokenize(context)
            sentences = re.split("[.?!]( )+", context)
            sentences = [s for s in sentences if not(s == " ")]
            meta_dict.append({"id": "context-" + "_".join(indices),
                              "speaker": "Google-TTS",
                              "text": sentences[int(indices[2])],
                              "normalized_text": text_preprocess(sentences[int(indices[2])]),
                              "duration": duration})
            segments[seg_key].append(indices[2])
        except:
            # very simple check
            meta_dict_err.append(file)
    # saving meta-[part].csv
    meta_df_train = pd.DataFrame.from_dict(meta_dict)
    # save meta info
    meta_df_train.to_csv(path.join(output_folder, "meta_data.csv"))
    # save segment info
    with open(path.join(output_folder, "data_segment_id.json"),"w+") as fp:
        json.dump(segments, fp)
    # save hash2question
    generate_hash2question(original_data, output_folder)


def generate_hash2question(original_data, output_folder):
    """
    This function takes a dict file and generates hash2question dictionary.
    The keys are the hash ids from dict file, and the values are the question indices.
    Ex:   "038v87hfbv": "question-0_0_0"
    Args:
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


def check_files(original_data, audio_dir, output_folder):
    """
    It simply check the number of occurences in the generated files.
    Args:
        original_data: str, the path of the json file, SQuAD formatted
        audio_dir: str, the path of the audio files
        output_folder: str, the path of the output contains all generated files
    Returns:
        None
    """
    # Simple check for the files
    # get the original dict
    with open(original_data) as fp:
        original_data = json.load(fp)['data']
    # the number of rows in meta must be the same as the number of wav/mp3 files in audio
    audios = [f for f in listdir(audio_dir) if f.endswith(".wav")]
    meta_data = pd.read_csv(path.join(output_folder, "meta_data.csv"))
    assert (meta_data.shape[0] == len(audios))
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
    # Number of values in segments must be the same as the number of rows of the meta = (nbr of audio)
    segments_value = sum([len(s) for k, s in segments.items()])
    assert (segments_value == meta_data.shape[0])

    # Number of elements in has2question must be the same as the number of question in the original data
    with open(path.join(output_folder, "hash2question.json")) as fp:
        hash2questions = json.load(fp)
    assert(len(hash2questions) == nbr_q)


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
    """
    saves the given dt into a file
    """
    dt.to_csv(out_file, sep=",", header=True, index=False)


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
    parser.add_argument('--language',
                        help='Language abbreviation for speech',
                        default="eng")
    parser.add_argument('--debug',
                        help='Set True if simple check is applied',
                        action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info("Starting the script using input file %s" % args.input)
    if args.format == "tsv":
        dt = preprocess_data_from_tsv(args.input, args.audio)
        logger.info("Saving the meta file into %s " % args.output)
        save_meta_csv(dt, args.output)
    elif args.format == "json":
        generate_meta_segment_hash_file(args.input, args.audio, args.output, args.audio_format)
    else:
        print("The format of the input file must be given as json or tsv")
    if args.debug:
        check_files(args.input, args.audio, args.output)
    return


if __name__ == "__main__":
    main()

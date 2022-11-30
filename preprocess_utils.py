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
import logging
from os import path

import librosa
from mutagen.mp3 import MP3
import pandas as pd
from tqdm import tqdm

from utils import text_preprocess

FORMAT = "%(asctime)s - %(name)s - %(levelname)s -[%(filename)s:%(lineno)s - %(funcName)20s() ]- %(message)s"
logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format=FORMAT,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)


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
                        help='input tsv file name',
                        required=True)
    parser.add_argument('--audio',
                        help='input audio folder',
                        required=True)
    parser.add_argument('--output',
                        help='Output file name',
                        required=True)
    parser.add_argument('--language',
                        help='Language abbreviation for speech',
                        default="eng")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info("Starting the script using input file %s" % args.input)
    dt = preprocess_data_from_tsv(args.input, args.audio)
    logger.info("Saving the meta file into %s " % args.output)
    save_meta_csv(dt, args.output)
    return


if __name__ == "__main__":
    main()

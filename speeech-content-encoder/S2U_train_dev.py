"""
This script runs the feature extraction using trained hubert and kmeans.
It takes meta file and audio files and generates features as .code and .cnt files for each audio.

Args:
    part: str, the name of the part ex: train, dev
    hubert: str, the path of the hubert model
    kmeans: str, the path of the kmeans
    meta: str, the path of the meta file
    audios: str, the path of the audio files
    output: str, the output folder
    audio_type: str, the extension of audio files as .mp3, .wav

Usage:
 >

"""
import argparse
import joblib
import logging
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

SAMPLE_RATE = 16000
CHUNK_LENGTH = 250000

FORMAT = "%(asctime)s - %(name)s - %(levelname)s -[%(filename)s:%(lineno)s - %(funcName)20s() ]- %(message)s"
logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format=FORMAT,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)


class ApplyKmeans(object):
    """
    This class contains necessary functions to apply trained kmeans.
    """

    def __init__(self, km_path, return_diff=False):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.return_diff = return_diff
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = torch.sqrt(
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            min_dist = dist.detach().min(dim=1)
            if self.return_diff:
                return min_dist.indices.cpu().numpy(), min_dist.values.cpu().numpy()
            else:
                return min_dist.indices.cpu().numpy()
        else:
            dist = np.sqrt(
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            if self.return_diff:
                return np.argmin(dist, axis=1), np.min(dist, axis=1)
            else:
                return np.argmin(dist, axis=1)


def reader(fname):
    """
    :param fname, str reader for audio files
    Returns:
    """
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)
    return wav.squeeze()


def run_train(train_meta,
              audio_file_dir,
              output_dir,
              extractor,
              kmeans,
              audio_type='.mp3'):
    """
    running kmeans for train set
    """
    # reading the meta csv file
    logger.info("Reading file %s", train_meta)
    df = pd.read_csv(train_meta)

    if torch.cuda.is_available():
        # load the extractor
        extractor = extractor.cuda()
    ind = 0

    for file in tqdm(df['id'].values, desc='transforming passage to discrete code'):
        # for each audio file
        audio_file = os.path.join(audio_file_dir, file + audio_type)
        wavs = reader(audio_file)

        if len(wavs) > 20 * SAMPLE_RATE:
            continue

        wavs = wavs.cuda()
        # extract the features for each audio file
        feature = extractor([wavs])

        code = kmeans(feature['hidden_state_22'].squeeze().cuda())
        code = torch.tensor(code)

        merged_code, counts = torch.unique_consecutive(code, return_counts=True)
        np.savetxt(os.path.join(output_dir, file + '.code'), merged_code.long(), fmt='%i')
        np.savetxt(os.path.join(output_dir, file + '.cnt'), counts.long(), fmt='%i')
        ind += 1
        if ind % 100 == 0:
            logger.info("Feature extracted for %d examples ", ind)
    logger.info("Feature extraction is done for training, file %s", train_meta)


def run_dev(dev_meta,
            audio_file_dir,
            output_dir,
            extractor,
            kmeans,
            audio_ftype='.mp3'):
    """
     running kmeans for train set
    """
    logger.info("Reading file %s", dev_meta)
    df = pd.read_csv(dev_meta)
    ind = 0

    for file in tqdm(df['id'].values, desc='transforming passage to discrete code'):
        audio_file = os.path.join(audio_file_dir, file + audio_ftype)
        wavs = reader(audio_file)
        wavs = wavs.cuda()

        if len(wavs) > 20 * SAMPLE_RATE:
            print(f'{file} too long')
            chunks = torch.split(wavs, CHUNK_LENGTH)
            for i, chunk in enumerate(chunks):
                feat = extractor([chunk])
                feat = feat['hidden_state_22'].squeeze()

                if i == 0:
                    feature = feat
                else:
                    feature = torch.cat([feature, feat], dim=0)

            code = kmeans(feature.cuda())

        else:
            feature = extractor([wavs])

            code = kmeans(feature['hidden_state_22'].squeeze().cuda())

        code = torch.tensor(code)

        merged_code, counts = torch.unique_consecutive(code, return_counts=True)
        np.savetxt(os.path.join(output_dir, file + '.code'), merged_code.long(), fmt='%i')
        np.savetxt(os.path.join(output_dir, file + '.cnt'), counts.long(), fmt='%i')
        ind += 1
        if ind % 100 == 0:
            logger.info("Feature extracted for %d examples ", ind)
    logger.info("Feature extraction is done for dev set, file %s", dev_meta)


def parse_args():
    """ Argument parser"""
    parser = argparse.ArgumentParser(description='Generate meta-file for given tsv file')
    parser.add_argument('--part',
                        help='the name of the part as train or dev',
                        required=True)
    parser.add_argument('--hubert',
                        help='path of the trained hubert model',
                        required=True)
    parser.add_argument('--kmeans',
                        help='path of the trained kmeans',
                        required=True)
    parser.add_argument('--meta',
                        help='path of the meta file',
                        required=True)
    parser.add_argument('--audios',
                        help='path of the audios',
                        required=True)
    parser.add_argument('--output',
                        help='output folder',
                        required=True)
    parser.add_argument('--audio_type',
                        help='the extension of audios as mp3,wav',
                        required=True,
                        default='.mp3')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info("Starting the script for feature extraction")
    logger.info("Loading extractor from %s", args.hubert_path)
    extractor = torch.hub.load('s3prl/s3prl', args.hubert_path)
    extractor.eval()
    logger.info("Loading trained kmeans %s ", args.kmeans)
    kmeans = ApplyKmeans(args.kmeans)

    if args.part == "train":
        run_train(args.meta,
                  args.audios,
                  args.output,
                  extractor,
                  kmeans,
                  args.audio_type)
    elif args.path == "dev":
        run_dev(args.mete,
                args.audios,
                args.output,
                extractor,
                kmeans,
                args.audio_type)
    else:
        print("Part name is wrong, should be train or dev.")

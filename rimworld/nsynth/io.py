from glob import glob as real_glob
import random

import tensorflow_io as tfio


def glob(pattern):
    """
    :param pattern: example pattern: 'data/nsynth-test.jsonwav/nsynth-test/audio/bass_electronic_*.wav'
    :return:
    """
    files = [f.replace("\\", "/") for f in real_glob(pattern)]
    return files


def audio_dataset_from_pattern(pattern, shuffle=True):
    """
    Create a lazy? AudioIODataset from a file pattern.

    :param pattern: file pattern according to glob.glob standards
    :param shuffle: randomize order of files
    :return: AudioIODataset with tuples (data, label)
    """
    files = glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError("No files found matching given pattern")

    if shuffle:
        random.shuffle(files)

    audio_dataset = tfio.audio.AudioIODataset.from_tensor_slices((files, files))
    return audio_dataset

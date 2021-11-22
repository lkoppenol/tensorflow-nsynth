import abc

import tensorflow as tf

"""
Capture groups:
\1 = path
\2 = instrument_name
\3 = instrument_id
\4 = pitch
\5 = velocity
"""
FILE_PATTERN = r"^(.*)/(.*)_([0-9]*)-([0-9]*)-([0-9]*)\.wav$"


class LabelOperation(abc.ABC):
    def __init__(self):
        pass

    def label_op(self, label: tf.Tensor):
        raise NotImplementedError()

    def __call__(self, data: tf.Tensor, label: tf.Tensor):
        return data, self.label_op(data)


class ExtractFilename(LabelOperation):
    def label_op(self, label: tf.Tensor):
        """
        Extract the filename from a path by taking the part after the last "/"

        :param label: the full file path, dtype=string
        :return: the filename, dtype=string
        """
        filename = (
            tf.strings
            .split(label, sep="/")
            [-1]
        )
        return filename


class ExtractPitchSparse(LabelOperation):
    def label_op(self, label: tf.Tensor):
        """
        Example:
        data/nsynth-test.jsonwav/nsynth-test/audio/bass_electronic_018-028-100.wav --> extract 028 --> bucket 7
        (min pitch is 21, so mapping is using -21)

        :param label: full path label Tensor, dtype=string
        :return: Tensor with Sparse label: e.g 6, dtype=int32
        """
        # See https://magenta.tensorflow.org/datasets/nsynth -> Description
        min_pitch = 21  # max_pitch = 108

        pitch_string = tf.strings.regex_replace(
            label,
            pattern=FILE_PATTERN,
            rewrite=r"\4"
        )
        pitch = tf.strings.to_number(pitch_string, out_type=tf.int32)
        pitch_label = pitch - min_pitch
        return pitch_label


class OneHotPitch(LabelOperation):
    def label_op(self, label: tf.Tensor):
        """
        Only pitches correctly encoded using extract_pitch_sparse are allowed.

        :param label: a sparse label, e.g. 6
        :return: a dense label, e.g. [0, 0, 0, 1, ...]
        """
        min_pitch = 21
        max_pitch = 108
        depth = max_pitch - min_pitch
        one_hot = tf.one_hot(label, depth=depth)
        return one_hot


class AllOnes(LabelOperation):
    def label_op(self, label: tf.Tensor):
        """
        Set label to 1

        :param label: is ignored
        :return: one, dtype=tf.int32
        """
        return tf.constant(1, dtype=tf.int32)


class AllZeros(LabelOperation):
    def label_op(self, label: tf.Tensor):
        """
        Set label to 0

        :param label: is ignored
        :return: one, dtype=tf.int32
        """
        return tf.constant(0, dtype=tf.int32)

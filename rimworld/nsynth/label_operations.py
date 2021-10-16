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


def label_operation(function):
    """
    Decorator function for data operations.
    An operation takes two input arguments; a data tensor and a label tensor.
    A label operation requires only the label tensor and passed the rest on.
    :param function:
    :return:
    """
    def decorator(data: tf.Tensor, label: tf.Tensor):
        """
        :param data: data tensor used by the function
        :param label: Is ignored and passed on to next operator
        :return:
        """
        result = function(label)
        return data, result
    return decorator


@label_operation
def extract_filename(label: tf.Tensor):
    filename = (
        tf.strings
        .split(label, sep="/")
        [-1]
    )
    return filename


@label_operation
def extract_pitch_sparse(label: tf.Tensor):
    """
    Example:
    data/nsynth-test.jsonwav/nsynth-test/audio/bass_electronic_018-028-100.wav --> extract 018 --> bucket 4

    :param label: full path label Tensor, dtype=string
    :return: Tensor with Sparse label: e.g 6, dtype=int8,.
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

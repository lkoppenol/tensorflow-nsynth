import tensorflow as tf
import tensorflow_io as tfio
import librosa


def data_operation(function):
    """
    Decorator function for data operations.
    An operation takes two input arguments; a data tensor and a label tensor.
    A data operation requires only the data tensor and passed the rest on.
    :param function:
    :return:
    """
    def decorator(data: tf.Tensor, label: tf.Tensor):
        """
        :param data: data tensor used by the function
        :param label: Is ignored and passed on to next operator
        :return:
        """
        result = function(data)
        return result, label
    return decorator


@data_operation
def read_file(tensor: tf.Tensor):
    """
    `data operation` version of tf io read_file

    :param tensor: Tensor with filename, dtype=string
    :return: Tensor with file contents, dtype=string
    """
    file_content = tf.io.read_file(tensor)
    return file_content


@data_operation
def decode_wav(tensor: tf.Tensor):
    """
    Decode a wave file which was read by `tf.io.read_file`

    :param tensor: Tensor with dtype=string
    :return: Tensor with dtype=int16
    """
    decoded = tfio.audio.decode_wav(
        tensor,
        dtype=tf.int16
    )
    return decoded


@data_operation
def squeeze(tensor: tf.Tensor):
    """
    Drop the last dimension of a Tensor

    :param tensor: Tensor with dimension [x, 1]
    :return: Tensor with dimension [x, ]
    """
    squeezed = tf.squeeze(
        tensor,
        axis=[-1]
    )
    return squeezed


@data_operation
def cast_normalized_float(tensor: tf.Tensor):
    """
    Cast to float and normalize between -1 and 1 by dividing by the max of the absolute

    :param tensor: Tensor with any numerical type
    :return: Tensor with dtype=float32, ranged between -1 and 1
    """
    floated = tf.cast(
        tensor,
        tf.float32
    )
    absolute = tf.abs(floated)
    maximum = tf.reduce_max(absolute)
    normalized = floated / maximum
    return normalized


@data_operation
def trim(tensor: tf.Tensor, epsilon=0.1):
    """
    Drop noise at start and end of signal if below threshold
    :param tensor:
    :param epsilon:
    :return:
    """
    position = tfio.audio.trim(tensor, axis=0, epsilon=epsilon)
    start = position[0]
    stop = position[1]
    return tensor[start:stop]


@data_operation
def zero_pad(tensor: tf.Tensor, length=64_000):
    """
    Add zeros to a 1D tensor to match given length

    :param tensor: 1D numerical tensor of length below given length
    :param length:
    :return: 1D numerical tensor of given length
    """
    zeros = tf.zeros(length - tf.shape(tensor))
    padded = tf.concat([tensor, zeros], axis=0)
    return padded


@data_operation
def get_first_second(tensor: tf.Tensor, sample_rate=16_000):
    """
    Select only the first second of audio

    :param tensor: 1D numerical tensor
    :param sample_rate: default nsynth = 16_000 hz
    :return: 1D numerical tensor with same dtype, shape=(sample_rate, )
    """
    first_second = tensor[:sample_rate]
    return first_second


@data_operation
def create_spectrogram(tensor: tf.Tensor, nfft=2048, window=2048, stride=512):
    """
    Create a frequency-time grid with relative intensity from an audio signal. The complex component (phase) is lost.

    TODO: explain dimensions & axes
    :param tensor: 1D tensor with dtype=float
    :param nfft:
    :param window:
    :param stride:
    :return: 2D tensor with dtype=float32
    """
    spectrogram = tfio.audio.spectrogram(
        tensor,
        nfft=nfft,
        window=window,
        stride=stride
    )
    return spectrogram


@data_operation
def decode_spectrogram(tensor: tf.Tensor, window=2048, stride=512):
    """
    Use librosa's griffinlim to reconstruct a spectrogram without phase to an audio signal.

    :param tensor: 2D tensor with dtype=float32
    :param window: as used to construct the spectrum
    :param stride: as used to construct the spectrum
    :return: 1D tensor with dtype=float32
    """
    def _decode(_tensor: tf.Tensor):
        return librosa.griffinlim(
            _tensor.numpy().T,
            win_length=window,
            hop_length=stride
        )
    signal = tf.py_function(
        func=_decode,
        inp=[tensor],
        Tout=tf.float32
    )
    return signal


@data_operation
def decode_log_spectrogram(tensor: tf.Tensor, window=2048, stride=512):
    """
    Use librosa's griffinlim to reconstruct a spectrogram without phase to an audio signal. Do exp(), inverse of log,
    before decoding.

    :param tensor: 2D tensor with dtype=float32
    :param window: as used to construct the spectrum
    :param stride: as used to construct the spectrum
    :return: 1D tensor with dtype=float32
    """
    def _decode(_tensor: tf.Tensor):
        return librosa.griffinlim(
            _tensor.numpy().T,
            win_length=window,
            hop_length=stride
        )

    unlogged = tf.math.exp(tensor)
    signal = tf.py_function(
        func=_decode,
        inp=[unlogged],
        Tout=tf.float32
    )
    return signal


@data_operation
def create_log_spectrogram(tensor: tf.Tensor, nfft=2048, window=2048, stride=512):
    """
    Create a frequency-time grid with relative intensity from an audio signal. The complex component (phase) is lost.

    TODO: explain dimensions & axes
    :param tensor: 1D tensor with dtype=float
    :param nfft:
    :param window:
    :param stride:
    :return: 2D tensor with dtype=float32
    """
    spectrogram = tfio.audio.spectrogram(
        tensor,
        nfft=nfft,
        window=window,
        stride=stride
    )
    log_spectrogram = tf.math.log(spectrogram)
    return log_spectrogram

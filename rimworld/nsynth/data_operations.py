import abc
import tensorflow as tf
import tensorflow_io as tfio
import librosa


class DataOperation(abc.ABC):
    def __init__(self):
        pass

    def data_op(self, data: tf.Tensor):
        raise NotImplementedError()

    def __call__(self, data: tf.Tensor, label: tf.Tensor):
        return self.data_op(data), label


class ReadFile(DataOperation):
    def data_op(self, data: tf.Tensor):
        """
        `data operation` version of tf io read_file

        :param data: Tensor with filename, dtype=string
        :return: Tensor with file contents, dtype=string
        """
        file_content = tf.io.read_file(data)
        return file_content


class DecodeWav(DataOperation):
    def data_op(self, data: tf.Tensor):
        """
        Decode a wave file which was read by `tf.io.read_file`

        :param data: Tensor with dtype=string
        :return: Tensor with dtype=int16
        """
        decoded = tfio.audio.decode_wav(
            data,
            dtype=tf.int16
        )
        return decoded


class Squeeze(DataOperation):
    def data_op(self, data: tf.Tensor):
        """
        Drop the last dimension of a Tensor

        :param data: Tensor with dimension [x, 1]
        :return: Tensor with dimension [x, ]
        """
        squeezed = tf.squeeze(
            data,
            axis=[-1]
        )
        return squeezed


class CastNormalizedFloat(DataOperation):
    def data_op(self, data: tf.Tensor):
        """
        Cast to float and normalize between -1 and 1 by dividing by the max of the absolute

        :param data: Tensor with any numerical type
        :return: Tensor with dtype=float32, ranged between -1 and 1
        """
        floated = tf.cast(
            data,
            tf.float32
        )
        absolute = tf.abs(floated)
        maximum = tf.reduce_max(absolute)
        normalized = floated / maximum
        return normalized


class Trim(DataOperation):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        super().__init__()

    def data_op(self, data: tf.Tensor):
        """
        Drop noise at start and end of signal if below threshold
        :param data:
        :return:
        """
        position = tfio.audio.trim(data, axis=0, epsilon=self.epsilon)
        start = position[0]
        stop = position[1]
        return data[start:stop]


class ZeroPad(DataOperation):
    def __init__(self, length=64_000):
        self.length = length
        super().__init__()

    def data_op(self, data: tf.Tensor):
        """
        Add zeros to a 1D tensor to match given length

        :param data: 1D numerical tensor of length below given length
        :return: 1D numerical tensor of given length
        """
        zeros = tf.zeros(self.length - tf.shape(data))
        padded = tf.concat([data, zeros], axis=0)
        return padded


class GetFirstSecond(DataOperation):
    def __init__(self, sample_rate=16_000):
        """

        :param sample_rate: default nsynth = 16_000 hz
        """
        self.sample_rate = sample_rate
        super().__init__()

    def data_op(self, data: tf.Tensor):
        """
        Select only the first second of audio

        :param data: 1D numerical tensor
        :return: 1D numerical tensor with same dtype, shape=(sample_rate, )
        """
        first_second = data[:self.sample_rate]
        return first_second


class EncodeSpectrogram(DataOperation):
    def __init__(self, nfft: int, window: int, stride: int, log: bool):
        self.nfft = nfft
        self.window = window
        self.stride = stride
        self.log = log
        super().__init__()

    def data_op(self, data: tf.Tensor):
        """
        Create a frequency-time grid with relative intensity from an audio signal. The complex component (phase) is lost.

        TODO: explain dimensions & axes
        :param data: 1D tensor with dtype=float
        :return: 2D tensor with dtype=float32
        """
        spectrogram = tfio.audio.spectrogram(
            data,
            nfft=self.nfft,
            window=self.window,
            stride=self.stride
        )

        if self.log:
            spectrogram = tf.math.log(spectrogram)

        return spectrogram


class DecodeSpectrogram(DataOperation):
    def __init__(self, window: int, stride: int, log: bool):
        self.window = window
        self.stride = stride
        self.log = log
        super().__init__()

    def data_op(self, data: tf.Tensor):
        """
        Use librosa's griffinlim to reconstruct a spectrogram without phase to an audio signal.

        :param data: 2D tensor with dtype=float32
        :return: 1D tensor with dtype=float32
        """
        def _decode(_tensor: tf.Tensor):
            return librosa.griffinlim(
                _tensor.numpy().T,
                win_length=self.window,
                hop_length=self.stride
            )

        if self.log:
            data = tf.math.exp(data)

        signal = tf.py_function(
            func=_decode,
            inp=[data],
            Tout=tf.float32
        )
        return signal


class Spectrogram:
    def __init__(self, nfft: int, window: int, stride: int, log: bool = False):
        self.nfft = nfft
        self.window = window
        self.stride = stride
        self.log = log

    def get_encoder(self):
        return EncodeSpectrogram(
            nfft=self.nfft,
            window=self.window,
            stride=self.stride,
            log=self.log
        )

    def get_decoder(self):
        return DecodeSpectrogram(
            window=self.window,
            stride=self.stride,
            log=self.log
        )


class Spectrum:
    def __init__(self, nfft: int, window: int, stride: int, log: bool = False):
        self.nfft = nfft
        self.window = window
        self.stride = stride
        self.log = log

    def get_encoder(self):
        return EncodeSpectrum(
            nfft=self.nfft,
            window=self.window,
            stride=self.stride,
            log=self.log
        )

    def get_decoder(self):
        return DecodeSpectrum(
            window=self.window,
            stride=self.stride,
            log=self.log
        )


class EncodeSpectrum(DataOperation):
    def __init__(self, nfft: int, window: int, stride: int, log: bool):
        self.nfft = nfft
        self.window = window
        self.stride = stride
        self.log = log
        super().__init__()

    def data_op(self, data: tf.Tensor):
        spectrogram_encoder = EncodeSpectrogram(self.nfft, self.window, self.stride, self.log)
        spectrogram = spectrogram_encoder.data_op(data)
        spectrum = tf.reduce_sum(spectrogram, axis=0)
        power_of_two_spectrum = spectrum[:-1]
        return power_of_two_spectrum


class DecodeSpectrum(DataOperation):
    def __init__(self, window: int, stride: int, log: bool):
        self.window = window
        self.stride = stride
        self.log = log
        self.length = None
        super().__init__()

    def data_op(self, data: tf.Tensor):
        """
        Use librosa's griffinlim to reconstruct a spectrogram without phase to an audio signal.

        :param data: 2D tensor with dtype=float32
        :return: 1D tensor with dtype=float32
        """

        def _decode(_tensor: tf.Tensor):
            if self.length is None:
                self.length = _tensor.shape[0]
            corrected_length = tf.concat([_tensor, tf.zeros((1, ))], axis=0)
            corrected_shape = tf.reshape(corrected_length, (self.length + 1, 1))
            spectrogram = tf.concat([corrected_shape] * 32, axis=1)

            return librosa.griffinlim(
                spectrogram.numpy(),
                win_length=self.window,
                hop_length=self.stride
            )

        if self.log:
            data = tf.math.exp(data)

        signal = tf.py_function(
            func=_decode,
            inp=[data],
            Tout=tf.float32
        )
        return signal

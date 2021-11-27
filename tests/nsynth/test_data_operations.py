import pathlib

import pytest
import tensorflow as tf
import numpy as np

from rimworld.nsynth import data_operations as ops


@pytest.fixture()
def raw_wav_tensor():
    cd = pathlib.Path(__file__).parent.resolve()
    test_wav_path = str(cd / "test_file.wav")
    return tf.convert_to_tensor(test_wav_path, dtype=tf.string)


@pytest.fixture()
def parsed_wav_tensor():
    raw_values = np.array([[32767], [3276], [327], [-328], [-3277], [-32768]])
    return tf.convert_to_tensor(raw_values, dtype=tf.int16)


@pytest.fixture()
def squeezed_wav_tensor():
    raw_values = np.array([32767, 3276, 327, -328, -3277,  -32768])
    return tf.convert_to_tensor(raw_values, dtype=tf.int16)


def test_decode_wav(raw_wav_tensor, parsed_wav_tensor):
    data = ops.ReadFile().data_op(raw_wav_tensor)
    parsed = ops.DecodeWav().data_op(data)
    assert tf.reduce_all(tf.equal(parsed, parsed_wav_tensor))


def test_squeeze(parsed_wav_tensor, squeezed_wav_tensor):
    squeezed = ops.Squeeze().data_op(parsed_wav_tensor)
    assert tf.reduce_all(tf.equal(squeezed, squeezed_wav_tensor))


def test_cast_normalized_float():
    data = tf.convert_to_tensor(np.array([10, 1, 0, -1, -10]),dtype=tf.int16)
    normalized = ops.CastNormalizedFloat().data_op(data)
    to_be = tf.convert_to_tensor(np.array([1.0, 0.1, 0, -0.1, -1.0]), dtype=tf.float32)
    assert tf.reduce_all(tf.equal(normalized, to_be))

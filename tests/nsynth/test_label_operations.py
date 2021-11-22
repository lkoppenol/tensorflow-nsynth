from rimworld.nsynth import label_operations as ops
import tensorflow as tf
import numpy as np


def test_extract_filename():
    """
    Test if the filename can be correctly extracted from a longer path
    """
    tensor = tf.convert_to_tensor("data/nsynth-test.jsonwav/nsynth-test/audio/bass_electronic.wav", dtype=tf.string)
    extract_filename = ops.ExtractFilename()
    a = extract_filename.label_op(tensor)
    b = tf.convert_to_tensor("bass_electronic.wav", dtype=tf.string)
    assert tf.equal(a, b)


def test_extract_pitch_sparse():
    """
    Test if a subset of pitches is correctly extracted from the filenames
    """
    filenames = [
        f"data/nsynth-test.jsonwav/nsynth-test/audio/bass_electronic_001-{i:03}-001.wav"
        for i in range(21, 31)
    ]
    tensors = [
        tf.convert_to_tensor(filename, dtype=tf.string)
        for filename in filenames
    ]
    extract_pitch_sparse = ops.ExtractPitchSparse()
    pitches = [
        extract_pitch_sparse.label_op(tensor)
        for tensor in tensors
    ]
    for i, bucket in enumerate(pitches):
        assert tf.equal(bucket, tf.convert_to_tensor(i, dtype=tf.int32))


def test_one_hot_pitch():
    """
    Test if all pitches correctly one-hot encode
    """
    min_pitch = 21
    max_pitch = 108
    depth = max_pitch - min_pitch

    tensors = []
    for i in range(depth):
        row = np.zeros((depth, ))
        row[i] = 1
        tensors.append(tf.convert_to_tensor(row, dtype=tf.float32))

    one_hot_pitch = ops.OneHotPitch()

    for i, tensor in enumerate(tensors):
        a = tf.convert_to_tensor(i, dtype=tf.int32)
        one_hot = one_hot_pitch.label_op(a)
        assert tf.reduce_all(tf.equal(tensor, one_hot))


def test_all_ones():
    """
    Test if we get a one
    """
    tensor = tf.constant("2", dtype=tf.string)
    all_ones = ops.AllOnes()
    one = all_ones.label_op(tensor)
    assert tf.equal(one, tf.constant(1, dtype=tf.int32))


def test_all_zeros():
    """
    Test if we get a one
    """
    tensor = tf.constant("2", dtype=tf.string)
    all_zeros = ops.AllZeros()
    zero = all_zeros.label_op(tensor)
    assert tf.equal(zero, tf.constant(0, dtype=tf.int32))

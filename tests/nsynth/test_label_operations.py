from rimworld.nsynth import label_operations as ops
import tensorflow as tf


def test_decorator():
    def func(a):
        return a * 2
    decorated_function = ops.label_operation(func)
    data, label = decorated_function(2, 4)
    assert data == 2
    assert label == 8


def test_extract_filename():
    tensor = tf.convert_to_tensor("data/nsynth-test.jsonwav/nsynth-test/audio/bass_electronic.wav", dtype=tf.string)
    _, a = ops.extract_filename(None, tensor)
    b = tf.convert_to_tensor("bass_electronic.wav", dtype=tf.string)
    assert tf.equal(a, b)


def test_extract_pitch():
    filenames = [
        f"data/nsynth-test.jsonwav/nsynth-test/audio/bass_electronic_001-{i:03}-001.wav"
        for i in range(21, 31)
    ]
    tensors = [
        tf.convert_to_tensor(filename, dtype=tf.string)
        for filename in filenames
    ]
    pitches = [
        ops.extract_pitch_sparse(None, tensor)[1]
        for tensor in tensors
    ]
    for i, bucket in enumerate(pitches):
        assert tf.equal(bucket, tf.convert_to_tensor(i, dtype=tf.int32))

from rimworld.nsynth import data_operations as ops


def test_decorator():
    def func(a):
        return a * 2
    decorated_function = ops.data_operation(func)
    data, label = decorated_function(2, 4)
    assert data == 4
    assert label == 4

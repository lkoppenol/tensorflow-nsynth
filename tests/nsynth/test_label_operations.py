from rimworld.nsynth import label_operations as ops


def test_decorator():
    def func(a):
        return a * 2
    decorated_function = ops.label_operation(func)
    data, label = decorated_function(2, 4)
    assert data == 2
    assert label == 8

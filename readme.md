![test workflow](https://github.com/lkoppenol/tensorflow-nsynth/actions/workflows/build.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/lkoppenol/tensorflow-nsynth/badge.svg?branch=main)](https://coveralls.io/github/lkoppenol/tensorflow-nsynth?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
### Requirements
Tested with Python 3.8
```commandline
pip install -r requirements.txt
```

### Test
```commandline
python -m pytest
```

### Example usage
```python
from rimworld.nsynth import io as nsynth_io
from rimworld.nsynth import data_operations as data_ops
from rimworld.nsynth import label_operations as label_ops

THREADS = 8
FILE_PATTERN = "data/nsynth-test.jsonwav/nsynth-test/audio/bass_electronic_*.wav"

operations = [
    data_ops.read_file,
    data_ops.decode_wav,
    data_ops.squeeze,
    data_ops.cast_normalized_float,
    data_ops.create_log_spectrogram,
    data_ops.decode_log_spectrogram,
    label_ops.extract_instrument_bucket
]

audio_dataset = nsynth_io.audio_dataset_from_pattern(FILE_PATTERN)
for o in operations:
    audio_dataset = audio_dataset.map(o, num_parallel_calls=THREADS)
```
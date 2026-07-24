# Architecture

The toolkit separates dataset loading, preprocessing, encoders, evaluation,
study planning, registry storage, and presentation.

`EpochBatch` is the public EEG boundary. Its axes are `(trial, channel,
sample)`. MATLAB/HDF5 storage order remains inside the dataset loader.

Supported studies start from YAML. `StudyRunner` writes the requested and
resolved configuration, a run plan, and a progress journal before execution.

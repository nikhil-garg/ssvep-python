# Architecture

The processing engine is independent of its interfaces. `cli.py` parses user
input, `config` validates it, `data` owns MATLAB/HDF5 axis conversion, and
`preprocessing` performs transformations. A future GUI will construct the same
validated configuration and call the same runners.

Raw MATLAB arrays use the logical axis order `(condition, channel, sample,
frequency, block)`. MATLAB 7.3 stores these through HDF5 and `h5py` exposes the
reversed shape `(block, frequency, sample, channel, condition)`. Only
`Matlab73Dataset` knows this storage detail; all downstream arrays use
`(channel, sample)` for a single trial.

Preprocessing iterates over one subject, condition, frequency, and block at a
time. Output HDF5 datasets use the documented logical order and include axis
labels, selected values, source identity, configuration JSON, and completion
state. An incomplete output can be resumed.

Compatibility behavior is explicit. `matlab_compatible` sample selection is
kept separate from anti-aliased `polyphase` resampling so a scientific choice
cannot change silently.


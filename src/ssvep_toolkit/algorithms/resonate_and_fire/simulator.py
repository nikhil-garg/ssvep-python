from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .model import ResonateAndFireParameters


def _compiled_simulator():
    """Create the optional Numba kernel lazily, keeping Numba an extra."""
    try:
        from numba import njit, prange
    except ImportError:
        return None

    @njit(parallel=True, cache=True)
    def kernel(data, frequencies, stops, sampling_rate, alpha, threshold, gain,
               transient, refractory, refractory_cycles, substeps, normalized, divide_input_by_resonance,
               notebook_reset, exact_solver, upward_crossing):
        duration_count=stops.size; trials,channels,_=data.shape; neurons=frequencies.size
        output=np.zeros((duration_count,trials,neurons),np.int32); maximum=stops[-1]
        for unit in prange(trials*neurons):
            trial=unit//neurons; neuron=unit%neurons; frequency=frequencies[neuron]
            neuron_refractory=max(refractory,int(round(refractory_cycles*sampling_rate/frequency)))*substeps
            if normalized:
                step=frequency/sampling_rate/substeps
                local_alpha=alpha; omega=2*np.pi
                drive_scale=gain/frequency if divide_input_by_resonance else gain
            else:
                step=1.0/sampling_rate/substeps
                local_alpha=alpha*frequency; omega=2*np.pi*frequency; drive_scale=gain
            # These coefficients are constant for a neuron. Computing exp,
            # sin, and cos inside every sample/substep dominated long runs.
            decay=np.exp(-local_alpha*step); co=np.cos(omega*step); si=np.sin(omega*step)
            er=decay*co; ei=decay*si; den=local_alpha*local_alpha+omega*omega
            br=(-local_alpha*(er-1.0)+omega*ei)/den; bi=(-omega*(er-1.0)-local_alpha*ei)/den
            counts=0; snapshot=0
            for channel in range(channels):
                x=0.0; y=0.0; blocked=0; channel_counts=0; snapshot=0
                for sample in range(maximum):
                    for _ in range(substeps):
                        old_y=y
                        spike=(not upward_crossing) and y>threshold and blocked==0
                        if spike and sample>=transient: channel_counts+=1
                        if spike:
                            x=0.0; y=threshold if notebook_reset else 0.0; blocked=neuron_refractory
                        drive=drive_scale*data[trial,channel,sample]
                        if exact_solver:
                            old_x=x; x=er*x-ei*y+br*drive; y=ei*old_x+er*y+bi*drive
                        else:
                            dx=-local_alpha*x-omega*y+drive; dy=omega*x-local_alpha*y; x+=dx*step; y+=dy*step
                        if upward_crossing and old_y<=threshold and y>threshold and blocked==0:
                            if sample>=transient: channel_counts+=1
                            x=0.0; y=threshold if notebook_reset else 0.0; blocked=neuron_refractory
                        if blocked>0: blocked-=1
                    while snapshot<duration_count and sample+1==stops[snapshot]:
                        output[snapshot,trial,neuron]+=channel_counts; snapshot+=1
        return output
    return kernel


_NUMBA_KERNEL = _compiled_simulator()


def _compiled_event_simulator():
    try:
        from numba import njit, prange
    except ImportError:
        return None

    @njit(parallel=True, cache=True)
    def kernel(data,freqs,stops,fs,alpha,threshold,gain,transient,refractory_cycles,substeps,divide_input_by_resonance,notebook_reset):
        trials,channels,_=data.shape;neurons=freqs.size;out=np.zeros((stops.size,trials,neurons,4),np.float32)
        for unit in prange(trials*neurons):
            trial=unit//neurons;ni=unit%neurons;f=freqs[ni];counts=0;first=-1.;cs=0.;ss=0.;states_x=np.zeros(channels);states_y=np.zeros(channels);blocked=np.zeros(channels,np.int32);refr=max(0,int(round(refractory_cycles*fs/f))*substeps);snap=0
            step=f/fs/substeps;omega=2*np.pi;decay=np.exp(-alpha*step);er=decay*np.cos(omega*step);ei=decay*np.sin(omega*step);den=alpha*alpha+omega*omega;br=(-alpha*(er-1)+omega*ei)/den;bi=(-omega*(er-1)-alpha*ei)/den
            for sample in range(stops[-1]):
                for sub in range(substeps):
                    event_time=(sample+sub/substeps)/fs
                    for ch in range(channels):
                        old_y=states_y[ch];drive=(gain/f if divide_input_by_resonance else gain)*data[trial,ch,sample];old_x=states_x[ch];states_x[ch]=er*states_x[ch]-ei*states_y[ch]+br*drive;states_y[ch]=ei*old_x+er*states_y[ch]+bi*drive
                        if old_y<=threshold and states_y[ch]>threshold and blocked[ch]==0:
                            if sample>=transient:
                                counts+=1
                                if first<0:first=event_time
                                phase=2*np.pi*f*event_time;cs+=np.cos(phase);ss+=np.sin(phase)
                            states_x[ch]=0.;states_y[ch]=threshold if notebook_reset else 0.;blocked[ch]=refr
                        if blocked[ch]>0:blocked[ch]-=1
                while snap<stops.size and sample+1==stops[snap]:
                    available=max((stops[snap]-transient)/fs,1/fs);out[snap,trial,ni,0]=counts/available;out[snap,trial,ni,1]=((first-transient/fs)/available if first>=0 else 1.0);out[snap,trial,ni,2]=(cs/counts if counts else 0.);out[snap,trial,ni,3]=(ss/counts if counts else 0.);snap+=1
        return out
    return kernel


_NUMBA_EVENT_KERNEL = _compiled_event_simulator()


def simulate_bank_event_features(signals: Any, resonance_frequencies_hz: Sequence[float], sampling_rate_hz: float,
                                 parameters: ResonateAndFireParameters, duration_samples: Sequence[int]) -> Any:
    """Return rate, normalized TTFS, and circular spike phase per neuron.

    Shape is `(duration, trial, neuron, 4)` with features
    `[spikes/s, TTFS/final-window, mean_cos_phase, mean_sin_phase]`.
    """
    import numpy as np
    if _NUMBA_EVENT_KERNEL is None:
        raise RuntimeError("event-feature simulation requires the performance extra")
    data=np.ascontiguousarray(np.asarray(signals,float));freqs=np.asarray(resonance_frequencies_hz,float);stops=np.sort(np.asarray(duration_samples,int));transient=round(parameters.transient_seconds*sampling_rate_hz)
    result=_NUMBA_EVENT_KERNEL(data,freqs,stops,float(sampling_rate_hz),float(parameters.damping_alpha),float(parameters.threshold),float(parameters.input_gain),int(transient),float(parameters.refractory_cycles),int(parameters.integration_substeps),bool(parameters.normalize_input_by_resonance),parameters.reset_mode=="notebook_compatible")
    return result[np.argsort(np.argsort(np.asarray(duration_samples,int)))]


def simulate_bank(
    signals: Any,
    resonance_frequencies_hz: Sequence[float],
    sampling_rate_hz: float,
    parameters: ResonateAndFireParameters,
    duration_samples: Sequence[int],
) -> Any:
    """Return cumulative spikes `(duration, trial, neuron)`.

    Signals use `(trial, channel, sample)`. Spikes are summed over channels.
    The Euler/reset order intentionally matches the reviewed notebook.
    """
    import numpy as np

    parameters.validate()
    data = np.asarray(signals, dtype=float)
    if data.ndim != 3:
        raise ValueError("signals must have shape (trial, channel, sample)")
    stops = np.asarray(duration_samples, dtype=int)
    if np.any(stops <= 0) or np.any(stops > data.shape[-1]):
        raise ValueError("duration samples are outside the signal")
    dt = 1.0 / sampling_rate_hz
    transient = round(parameters.transient_seconds * sampling_rate_hz)
    refractory = round(parameters.refractory_seconds * sampling_rate_hz)
    snapshots = np.zeros((len(stops), data.shape[0], len(resonance_frequencies_hz)), dtype=np.int32)
    frequencies = np.asarray(resonance_frequencies_hz, dtype=np.float64)
    if _NUMBA_KERNEL is not None:
        order = np.argsort(stops)
        sorted_stops = stops[order]
        compiled = _NUMBA_KERNEL(
            np.ascontiguousarray(data), frequencies, sorted_stops, float(sampling_rate_hz),
            float(parameters.damping_alpha), float(parameters.threshold), float(parameters.input_gain),
            int(transient), int(refractory), float(parameters.refractory_cycles), int(parameters.integration_substeps),
            bool(parameters.normalized_dynamics), bool(parameters.normalize_input_by_resonance),
            parameters.reset_mode == "notebook_compatible", parameters.solver == "exact",
            parameters.spike_detection == "upward_crossing",
        )
        inverse = np.argsort(order)
        return compiled[inverse]
    for neuron, frequency in enumerate(resonance_frequencies_hz):
        # In normalized time tau=f_res*t, the forcing frequency is f/f_res.
        # Time normalization and input-frequency compensation are deliberately
        # separate. The latter is optional because it can flatten raw spike
        # counts across resonance frequencies.
        normalized_step = frequency * dt / parameters.integration_substeps
        x = np.zeros(data.shape[:2], dtype=float)
        y = np.zeros_like(x)
        counts = np.zeros(data.shape[0], dtype=np.int32)
        blocked = np.zeros(data.shape[:2], dtype=np.int32)
        neuron_refractory=max(refractory,round(parameters.refractory_cycles*sampling_rate_hz/frequency))*parameters.integration_substeps
        snapshot_index = 0
        for sample in range(int(stops.max())):
            for _ in range(parameters.integration_substeps):
                spike = (y > parameters.threshold) & (blocked == 0)
                if sample >= transient: counts += spike.sum(axis=1)
                if np.any(spike):
                    x[spike]=0.; y[spike]=parameters.threshold if parameters.reset_mode=="notebook_compatible" else 0.; blocked[spike]=neuron_refractory
                if parameters.normalized_dynamics:
                    step=normalized_step; local_alpha=parameters.damping_alpha; omega=2*np.pi
                    scale = parameters.input_gain/frequency if parameters.normalize_input_by_resonance else parameters.input_gain
                    drive=scale*data[...,sample]
                else:
                    step=dt/parameters.integration_substeps; local_alpha=parameters.damping_alpha*frequency; omega=2*np.pi*frequency; drive=parameters.input_gain*data[...,sample]
                if parameters.solver=="exact":
                    decay=np.exp(-local_alpha*step); er=decay*np.cos(omega*step); ei=decay*np.sin(omega*step); den=local_alpha**2+omega**2; br=(-local_alpha*(er-1)+omega*ei)/den; bi=(-omega*(er-1)-local_alpha*ei)/den; old_x=x.copy(); x=er*x-ei*y+br*drive; y=ei*old_x+er*y+bi*drive
                else:
                    dx=-local_alpha*x-omega*y+drive;dy=omega*x-local_alpha*y;x+=dx*step;y+=dy*step
                blocked=np.maximum(blocked-1,0)
            while snapshot_index < len(stops) and sample + 1 == stops[snapshot_index]:
                snapshots[snapshot_index, :, neuron] = counts
                snapshot_index += 1
    return snapshots


def simulate_trace(
    signal: Any,
    resonance_frequency_hz: float,
    sampling_rate_hz: float,
    parameters: ResonateAndFireParameters,
) -> tuple[Any, Any, Any]:
    """Return `(spike_indices, x, y)` for one one-dimensional signal."""
    import numpy as np

    values = np.asarray(signal, dtype=float).reshape(-1)
    parameters.validate()
    dt = 1 / sampling_rate_hz
    normalized_step = resonance_frequency_hz * dt / parameters.integration_substeps
    x = np.zeros(values.size); y = np.zeros(values.size); spikes = []
    blocked = 0; refractory = max(round(parameters.refractory_seconds * sampling_rate_hz),round(parameters.refractory_cycles*sampling_rate_hz/resonance_frequency_hz))*parameters.integration_substeps
    for sample in range(values.size - 1):
        x_next, y_next = x[sample], y[sample]
        for _ in range(parameters.integration_substeps):
            old_y=y_next
            if parameters.spike_detection=="level" and y_next > parameters.threshold and blocked == 0:
                spikes.append(sample); x_next=0.; y_next=parameters.threshold if parameters.reset_mode=="notebook_compatible" else 0.; blocked=refractory
            if parameters.normalized_dynamics:
                step=normalized_step;local_alpha=parameters.damping_alpha;omega=2*np.pi
                scale=parameters.input_gain/resonance_frequency_hz if parameters.normalize_input_by_resonance else parameters.input_gain
                drive=scale*values[sample]
            else:
                step=dt/parameters.integration_substeps;local_alpha=parameters.damping_alpha*resonance_frequency_hz;omega=2*np.pi*resonance_frequency_hz;drive=parameters.input_gain*values[sample]
            if parameters.solver=="exact":
                decay=np.exp(-local_alpha*step);er=decay*np.cos(omega*step);ei=decay*np.sin(omega*step);den=local_alpha**2+omega**2;br=(-local_alpha*(er-1)+omega*ei)/den;bi=(-omega*(er-1)-local_alpha*ei)/den;old_x=x_next;x_next=er*x_next-ei*y_next+br*drive;y_next=ei*old_x+er*y_next+bi*drive
            else:
                dx=-local_alpha*x_next-omega*y_next+drive;dy=omega*x_next-local_alpha*y_next;x_next+=dx*step;y_next+=dy*step
            if parameters.spike_detection=="upward_crossing" and old_y<=parameters.threshold and y_next>parameters.threshold and blocked==0:
                spikes.append(sample);x_next=0.;y_next=parameters.threshold if parameters.reset_mode=="notebook_compatible" else 0.;blocked=refractory
            blocked=max(blocked-1,0)
        x[sample + 1], y[sample + 1] = x_next, y_next
    return np.asarray(spikes, dtype=int), x, y

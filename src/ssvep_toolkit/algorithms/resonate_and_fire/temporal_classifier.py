from __future__ import annotations
from typing import Any,Sequence
import numpy as np
from .model import ResonateAndFireParameters
from .simulator import simulate_bank_event_features

class TemporalOscillatorBankClassifier:
    """Training-template classifier using rate, TTFS, and circular phase."""
    def __init__(self,frequencies_hz:Sequence[float],sampling_rate_hz:float,parameters:ResonateAndFireParameters,*,spread_hz=(0.,),harmonics=(1,2,3)):
        self.frequencies_hz=tuple(map(float,frequencies_hz));self.sampling_rate_hz=float(sampling_rate_hz);self.parameters=parameters;self.spread_hz=tuple(map(float,spread_hz));self.harmonics=tuple(map(int,harmonics));self.channel_scale_=None;self.feature_center_=None;self.feature_scale_=None;self.templates_=None
    @property
    def neuron_frequencies_hz(self):return tuple(h*f+o for f in self.frequencies_hz for h in self.harmonics for o in self.spread_hz)
    def fit_scaler(self,x):
        x=np.asarray(x,float);s=np.std(x,axis=(0,2),ddof=1);self.channel_scale_=np.where(s>1e-12,s,1.);return self
    def transform(self,x):
        x=np.asarray(x,float);return (x-x.mean(-1,keepdims=True))/self.channel_scale_[None,:,None]
    def features(self,x,stops):
        raw=simulate_bank_event_features(self.transform(x),self.neuron_frequencies_hz,self.sampling_rate_hz,self.parameters,stops);shape=raw.shape[:2]+(len(self.frequencies_hz),len(self.harmonics),len(self.spread_hz),4);return raw.reshape(shape).mean(axis=4).reshape(raw.shape[0],raw.shape[1],-1)
    def fit_calibration(self,x,y,stops):
        y=np.asarray(y,int);z=self.features(x,stops);self.feature_center_=z.mean(1);self.feature_scale_=np.maximum(z.std(1,ddof=1),1e-6);standard=(z-self.feature_center_[:,None,:])/self.feature_scale_[:,None,:];self.templates_=np.stack([standard[:,y==c].mean(1) for c in range(len(self.frequencies_hz))],axis=1);return self
    def decision_scores(self,x,stops):
        z=self.features(x,stops);standard=(z-self.feature_center_[:,None,:])/self.feature_scale_[:,None,:];return -np.mean((standard[:,:,None,:]-self.templates_[:,None,:,:])**2,axis=-1)
    def predict(self,x,stops):return np.argmax(self.decision_scores(x,stops),axis=-1)

from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedModel
from transformers.utils.generic import ModelOutput

from vocal_remover.inference import Separator
from vocal_remover.lib import spec_utils
from vocal_remover.lib.nets import CascadedNet
from vocal_remover.transformer.configuration import VocalRemoverConfig


@dataclass
class VocalRemoverOutput(ModelOutput):
    instruments: np.ndarray | None = None
    vocals: np.ndarray | None = None
    sampling_rate: int | None = None


class VocalRemoverModel(PreTrainedModel):
    config_class = VocalRemoverConfig

    def __init__(self, config: VocalRemoverConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = CascadedNet(
            config.n_fft,
            config.hop_length,
            nout=config.nout,
            nout_lstm=config.nout_lstm,
            is_complex=config.is_complex,
        )
        self.post_init()

    def forward(
        self,
        audio: np.ndarray,
        sample_rate: int,
        batchsize: int = 1,
        cropsize: int = 256,
        tta: bool = True,
    ) -> VocalRemoverOutput:
        X_spec = spec_utils.wave_to_spectrogram(
            audio,
            self.config.hop_length,
            self.config.n_fft,
        )

        sp = Separator(
            self.model,
            device=self.device,
            batchsize=batchsize,
            cropsize=cropsize,
        )

        if tta:
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)

        instruments = spec_utils.spectrogram_to_wave(
            y_spec, hop_length=self.config.hop_length
        )
        vocals = spec_utils.spectrogram_to_wave(
            v_spec, hop_length=self.config.hop_length
        )

        return VocalRemoverOutput(
            instruments=instruments,
            vocals=vocals,
            sampling_rate=sample_rate,
        )

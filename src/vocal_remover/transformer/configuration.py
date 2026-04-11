from transformers import PretrainedConfig


class VocalRemoverConfig(PretrainedConfig):
    model_type = "tsurumeso-vocal-remover"

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 1024,
        nout: int = 32,
        nout_lstm: int = 128,
        is_complex: bool = False,
        **kwargs,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.nout = nout
        self.nout_lstm = nout_lstm
        self.is_complex = is_complex
        super().__init__(**kwargs)

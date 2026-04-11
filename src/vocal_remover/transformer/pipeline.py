from typing import Any

import numpy as np
import requests
from transformers import Pipeline, is_torch_available
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.utils import (
    ModelOutput,  # pyright: ignore[reportPrivateImportUsage]
    is_torchaudio_available,  # pyright: ignore[reportPrivateImportUsage]
    is_torchcodec_available,  # pyright: ignore[reportPrivateImportUsage]
)


class VocalRemoverPipeline(Pipeline):
    def _sanitize_parameters(
        self,
        sample_rate: int | None = None,
        batchsize: int | None = None,
        cropsize: int | None = None,
        tta: bool | None = None,
        **kwargs: dict,
    ):
        preprocess_parameters = {}
        forward_parameters = {}

        if sample_rate is not None:
            preprocess_parameters["sample_rate"] = sample_rate
        if batchsize is not None:
            forward_parameters["batchsize"] = batchsize
        if cropsize is not None:
            forward_parameters["cropsize"] = cropsize
        if tta is not None:
            forward_parameters["tta"] = tta

        return preprocess_parameters, forward_parameters, {}

    def preprocess(
        self,
        input_: Any,
        sample_rate: int = 44100,
        **preprocess_parameters: dict,
    ):
        audio = self._load_audio_input(input_, sample_rate)

        if audio.ndim == 1:
            audio = np.asarray([audio, audio])
        elif audio.ndim > 2:
            raise ValueError(
                f"The pipeline only supports mono or stereo audios as input. "
                f"The provided audio has {audio.ndim} channels."
            )

        return {"audio": audio, "sample_rate": sample_rate}

    def _load_audio_input(self, input_: Any, sample_rate: int):
        if isinstance(input_, str):
            if input_.startswith("http://") or input_.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                input_ = requests.get(input_).content
            else:
                with open(input_, "rb") as f:
                    input_ = f.read()

        if isinstance(input_, bytes):
            input_ = ffmpeg_read(input_, sample_rate)

        if is_torch_available():
            import torch  # pyright: ignore[reportMissingImports]

            if isinstance(input_, torch.Tensor):
                input_ = input_.cpu().numpy()

        if is_torchcodec_available():
            import torch  # pyright: ignore[reportMissingImports]
            import torchcodec  # pyright: ignore[reportMissingImports]

            if isinstance(input_, torchcodec.decoders.AudioDecoder):
                _audio_samples = input_.get_all_samples()  # pyright: ignore[reportAttributeAccessIssue]
                _array = _audio_samples.data
                input_ = {"array": _array, "sampling_rate": _audio_samples.sample_rate}

        if isinstance(input_, dict):
            input_ = (
                input_.copy()
            )  # So we don't mutate the original dictionary outside the pipeline
            # Accepting `"array"` which is the key defined in `datasets` for
            # better integration
            if not (
                "sampling_rate" in input_ and ("raw" in input_ or "array" in input_)
            ):
                raise ValueError(
                    "When passing a dictionary to AudioClassificationPipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array or torch tensor representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            _inputs = input_.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                input_.pop("path", None)
                _inputs = input_.pop("array", None)
            in_sampling_rate = input_.pop("sampling_rate")
            input_ = _inputs
            if in_sampling_rate != sample_rate:
                import torch  # pyright: ignore[reportMissingImports]

                if is_torchaudio_available():
                    from torchaudio import (  # pyright: ignore[reportMissingImports]
                        functional as F,
                    )
                else:
                    raise ImportError(
                        "torchaudio is required to resample audio samples in AudioClassificationPipeline. "
                        "The torchaudio package can be installed through: `pip install torchaudio`."
                    )

                input_ = F.resample(
                    torch.from_numpy(input_)
                    if isinstance(input_, np.ndarray)
                    else input_,
                    in_sampling_rate,
                    sample_rate,
                ).numpy()

        if not isinstance(input_, np.ndarray):
            raise TypeError("We expect a numpy ndarray or torch tensor as input")

        return input_

    def _forward(
        self,
        input_tensors: dict[str, Any],
        batchsize: int = 4,
        cropsize: int = 256,
        tta: bool = True,
        **forward_parameters: dict,
    ):
        return self.model(
            **input_tensors,
            batchsize=batchsize,
            cropsize=cropsize,
            tta=tta,
        )

    def postprocess(
        self,
        model_outputs: ModelOutput,
        **postprocess_parameters: dict,
    ):
        return model_outputs

from pyannote.audio import Pipeline
pipeline=Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

import torch
pipeline.to(torch.device("cpu"))

diarization=pipeline("sound/chunk_5.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
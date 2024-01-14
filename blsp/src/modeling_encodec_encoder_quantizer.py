from transformers.models.encodec.modeling_encodec import *


class EncodecEncoderQuantizer(EncodecModel):
    def __init__(self, config: EncodecConfig):
        super().__init__(config)

    def encode_quantize(
            self,
            input_values: torch.Tensor,
            padding_mask: torch.Tensor = None,
    ):
        encoder_outputs = self.encode(input_values=input_values, padding_mask=padding_mask)
        chunk_length = self.config.chunk_length
        if chunk_length is None:
            if len(encoder_outputs.audio_codes) != 1:
                raise ValueError(f"Expected one frame, got {len(encoder_outputs.audio_codes)}")
            speech_embeds = self.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1))
        else:
            decoded_frames = []

            for frame in encoder_outputs.audio_codes:
                frames = self.encodec_model.quantizer.decode(frame.transpose(0, 1))
                decoded_frames.append(frames)

            speech_embeds = self._linear_overlap_add(decoded_frames, self.config.chunk_stride or 1)

        return speech_embeds

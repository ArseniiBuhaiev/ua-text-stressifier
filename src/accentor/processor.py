import os
import re
from dotenv import load_dotenv
import torch
import json
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.accentor.text_utils import (
    UKRAINIAN_LETTERS,
    clean_text,
    merge_texts,
    split_text_by_whitespace
)
from utils import shift_stress_marks_right, shift_stress_marks_left
from nemo.collections.tts.models.base import G2PModel

load_dotenv()


class UkrainianStressifier:
    """
    Applies stress marks to Ukrainian text using a grapheme-to-phoneme (G2P) model.
    """

    def __init__(
        self,
        model_path: str | None = None,
        hf_token: str | None = None,
        max_chunks_length: int | None = 256,
        max_length: int | None  = 256,
        num_beams: int | None  = 5
    ):
        """
        Initializes the stressifier by loading a G2P model and setting decoding parameters.

        Args:
            model_path (Optional[str]): Path to a local model file.
            hf_token (Optional[str]): Hugging Face token for downloading the model.
            max_chunks_length (Optional[int]): Max chunk length for internal splitting.
            max_length (Optional[int]): Maximum generation length for the model.
            num_beams (Optional[int]): Number of beams for beam search decoding.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_chunks_length = max_chunks_length
        self.max_length = max_length
        self.num_beams = num_beams

        if model_path is None:
            model_path = hf_hub_download(
                repo_id="mouseyy/stressifier-byt5-g2p-model",
                filename="T5G2P.nemo",
                token=hf_token or os.getenv("HF_TOKEN"),
            )
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        map_location = self.device
        model = G2PModel.restore_from(model_path, map_location=map_location)
        return model

    def _text_to_phonemes(self, text: str) -> str:
        """
        Converts graphemes to phonemes using the loaded G2P model.

        Args:
            text (str): The cleaned input string.

        Returns:
            str: The phonetic transcription with stress marks.
        """
        tokenizer = self.model._tokenizer  # T5-based tokenizer
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )

        phonemes = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return phonemes

    def _text_to_phonemes_io(self, text: str) -> str:
        """
            Converts graphemes to phonemes using a G2P model.

            Args:
                text (str): The cleaned input string.
                model: The G2P model object.

            Returns:
                str: The phonetic transcription with stress marks.
            """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_input:
            temp_input.write(json.dumps({"text_graphemes": text}) + "\n")
            input_path = temp_input.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_output:
            output_path = temp_output.name

        try:
            with torch.no_grad():
                self.model.convert_graphemes_to_phonemes(
                    manifest_filepath=input_path,
                    output_manifest_filepath=output_path,
                    grapheme_field="text_graphemes",
                    pred_field="pred_text",
                )

            with open(output_path, "r") as f:
                output_data = [json.loads(line) for line in f]

            if output_data and "pred_text" in output_data[0]:
                return output_data[0]["pred_text"]
            else:
                return ""
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def apply_stress_marks(self, text: str, stress_after_vowel: bool = True) -> str:
        """
        Inserts stress marks into Ukrainian text.

        Args:
            text (str): Input Ukrainian text.
            stress_after_vowel (bool): If True, places stress mark after the stressed vowel;
                                   if False, places it before.

        Returns:
            str: Text with stress marks applied to stressed vowels.
        """
        chunks = split_text_by_whitespace(text, max_length=self.max_chunks_length)
        result_chunks = []

        for chunk in chunks:
            cleaned = clean_text(chunk)
            match = re.search(rf"[{UKRAINIAN_LETTERS}]", cleaned)

            if not match:
                result_chunks.append(chunk)
                continue

            prefix = cleaned[:match.start()]
            core = cleaned[match.start():]
            ends_with_period = core.endswith(".")

            if not ends_with_period:
                core += "."

            stressed = self._text_to_phonemes(core)
            stressed = shift_stress_marks_right(stressed)

            if stressed.endswith(".") and not ends_with_period:
                stressed = stressed[:-1]

            merged = merge_texts(chunk, prefix + stressed)
            result_chunks.append(merged)

        result = "".join(result_chunks)
        if not stress_after_vowel:
            result = shift_stress_marks_left(result)
        return result

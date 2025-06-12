import re
from src.phonemizer.rules import dict_narrow, dict_ipa


class UkrainianPhonemizer:
    """
    A phonemizer for Ukrainian text that converts input into its allophonic and IPA transcription.
    """

    @staticmethod
    def apply_allophonic_rules(text: str) -> str:
        """
        Applies narrow phonological rules to Ukrainian text.

        Args:
            text (str): The input text.

        Returns:
            str: Text transformed with allophonic rules.
        """
        result = text.lower()
        for pattern, replacement in dict_narrow.items():
            result = re.sub(pattern, replacement, result)
        return result

    @staticmethod
    def phonemize(text: str) -> str:
        """
        Converts allophonic transcription of Ukrainian text into IPA.

        Args:
            text (str): The input text.

        Returns:
            str: IPA-transcribed string.
        """
        result = UkrainianPhonemizer.apply_allophonic_rules(text)
        for pattern, replacement in dict_ipa.items():
            result = re.sub(pattern, replacement, result)
        return result

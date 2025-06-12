from src.accentor import UkrainianStressifier
from src.phonemizer import UkrainianPhonemizer

if __name__ == "__main__":
    import sys
    phonemizer = UkrainianPhonemizer()
    stressifier = UkrainianStressifier()
    input_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter text: ")

    text_with_stress = stressifier.apply_stress_marks(input_text)
    phonemizer_test = phonemizer.phonemize(text_with_stress)

    print(phonemizer_test)

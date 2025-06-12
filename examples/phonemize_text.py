from src.phonemizer import UkrainianPhonemizer

if __name__ == "__main__":
    import sys
    phonemizer = UkrainianPhonemizer()
    input_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter text: ")
    print(phonemizer.phonemize(input_text))

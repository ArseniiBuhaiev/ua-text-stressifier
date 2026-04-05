from ua_text_stressifier.accentor import UkrainianStressifier

if __name__ == "__main__":
    import sys
    stressifier = UkrainianStressifier()
    input_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter text: ")
    print(stressifier.apply_stress_marks(input_text))

from src.accentor import UkrainianStressifier
from lexical_stress_benchmark.benchmark import evaluate_stressification

path_to_nemo_model = "./T5G2P.nemo"
model = UkrainianStressifier(path_to_nemo_model)

def custom_stressify(text):
    return model.apply_stress_marks(text, stress_after_vowel=True)


if __name__ == "__main__":
    accuracies = evaluate_stressification(custom_stressify)
    sentence_accuracy, word_accuracy, heteronym_accuracy, unambiguous_accuracy, macro_average_f1_across_heteronyms = (
        accuracies.values()
    )

    print("Byt5 G2P results:")
    print(f"{'Sentence Accuracy:':40} {sentence_accuracy * 100:.2f}%")
    print(f"{'Word Accuracy:':40} {word_accuracy * 100:.2f}%")
    print(f"{'Unambiguous Words Accuracy:':40} {unambiguous_accuracy * 100:.2f}%")
    print(f"{'Heteronym Accuracy:':40} {heteronym_accuracy * 100:.2f}%")
    print(f"{'Macro-Average F1 score (Heteronyms)):':40} {macro_average_f1_across_heteronyms * 100:.2f}%")

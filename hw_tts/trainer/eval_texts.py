from hw_tts.text import text_to_sequence

tests = [
    "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
    "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
    "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
]

EVAL_DATA = list(text_to_sequence(
    test, ['english_cleaners']) for test in tests)

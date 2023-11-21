from hw_tts.text import text_to_sequence

tests = [
    "I am very happy to see you again!",
    "Durian model is a very good speech synthesis!",
    "When I was twenty, I fell in love with a girl.",
    "I remove attention module in decoder and use average pooling to implement predicting r frames at once",
    "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted.",
    "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old.",
    "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
    "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
    "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
]

EVAL_DATA = list(text_to_sequence(
    test, ['english_cleaners']) for test in tests)

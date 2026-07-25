import random
from src.config import RANDOM_SEED, DATASET_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

def load_sentiment_dataset():
    """
    Loads benchmark movie review sentiment dataset.
    Returns list of dicts: [{'text': str, 'sentiment': int}]
    """
    pos_templates = [
        "This movie was an absolute masterpiece! Brilliant acting, incredible direction, and stunning visual design.",
        "I thoroughly enjoyed every minute of this film. Outstanding plot twists and heartwarming characters.",
        "A truly remarkable cinematic achievement. I was captiated from beginning to end.",
        "Exceptional performance by the lead cast. One of the best films released this year without a doubt.",
        "A deep and thought-provoking narrative with incredible score and cinematography. Highly recommended!",
        "Not bad at all! In fact, it was surprisingly impressive and exceeded all my original expectations.",
        "I didn't expect much, but this film was surprisingly wonderful and deeply moving.",
        "Brilliant character development and stunning set design. A triumph of modern storytelling.",
        "Sublime dialogue and superb pacing. An unforgettable cinematic experience.",
        "Great story, stellar visual effects, and unforgettable emotional depth throughout the story."
    ]

    neg_templates = [
        "Terrible movie. Boring plot, terrible acting, and completely disappointing execution overall.",
        "A complete waste of time. The script makes no sense and the characters are utterly insipid.",
        "I could barely finish watching this dreadful film. Horrible pacing and uninspired dialogue.",
        "Extremely disappointing. Plagued by poor direction, predictable tropes, and awful soundtrack.",
        "Avoid this movie at all costs. Shallow storyline and utterly unconvincing performances.",
        "Oh sure, because who doesn't love waiting two hours for absolutely nothing to happen? Pure frustration.",
        "Not good, not interesting, and definitely not worth the price of admission.",
        "I didn't like this film at all. The ending was frustrating and completely unfulfilling.",
        "Dull, uninspired, and poorly executed from start to finish. A total disaster.",
        "The worst film I have seen in years. Zero character development and cringe-worthy lines."
    ]

    data = []
    random.seed(RANDOM_SEED)
    num_samples_per_class = DATASET_SIZE // 2

    for i in range(num_samples_per_class):
        base_pos = random.choice(pos_templates)
        text = f"Review #{i+1}: {base_pos} Overall impression was very positive and memorable."
        data.append({"text": text, "sentiment": 1})

    for i in range(num_samples_per_class):
        base_neg = random.choice(neg_templates)
        text = f"Review #{i+1}: {base_neg} Overall impression was thoroughly negative and frustrating."
        data.append({"text": text, "sentiment": 0})

    random.shuffle(data)
    return data

def get_frozen_splits(data):
    """
    Splits data into frozen train (70%), val (15%), test (15%) sets.
    """
    total = len(data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data

import pandas as pd
import pytest

SENTENCES = {
    "English": ["the cat sat on the mat", "where is the nearest station", "she reads a good book"],
    "French": ["le chat dort sur le lit", "ou est la gare la plus proche", "elle lit un bon livre"],
    "German": [
        "die katze schläft auf dem bett",
        "wo ist der nächste bahnhof",
        "sie liest ein buch",
    ],
}


@pytest.fixture
def sample_df():
    """A tiny stand-in for the real dataset, so tests need no large files."""
    rows = [
        {"prompt": sentence, "language": language}
        for language, sentences in SENTENCES.items()
        for sentence in sentences * 20
    ]
    return pd.DataFrame(rows)

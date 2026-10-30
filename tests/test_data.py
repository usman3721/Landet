import pandas as pd

from src.data_model_pipeline.data_ingestion import clean_data, save_processed_data, split_data
from src.data_model_pipeline.data_processing import clean_text


def test_clean_text_removes_noise():
    assert clean_text("Hello, World! 123 http://a.com @me") == "hello world"


def test_clean_text_handles_missing_value():
    assert clean_text(float("nan")) == ""


def test_clean_text_keeps_non_latin_letters():
    assert clean_text("مرحبا بالعالم") == "مرحبا بالعالم"


def test_clean_data_keeps_only_useful_columns():
    df = pd.DataFrame(columns=["new_id", "prompt", "label", "data_source", "old_id", "language"])
    assert list(clean_data(df).columns) == ["prompt", "language"]


def test_split_is_80_10_10_and_keeps_every_language(sample_df):
    train_df, val_df, test_df = split_data(sample_df)

    assert len(train_df) == int(len(sample_df) * 0.8)
    assert len(val_df) == len(test_df) == int(len(sample_df) * 0.1)
    for split in (train_df, val_df, test_df):
        assert set(split["language"]) == set(sample_df["language"])


def test_save_processed_data_writes_three_files(sample_df, tmp_path):
    save_processed_data(*split_data(sample_df), output_dir=tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["test.csv", "train.csv", "val.csv"]

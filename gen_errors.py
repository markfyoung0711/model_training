import pandas as pd
import random
import tempfile

from datetime import datetime
from google.cloud import storage

GOOGLE_BUCKET_NAME = 'north-dakota-daily-oil-gas-data-monthly'


def introduce_errors(df, column_name, min_error=-10, max_error=10):
    """
    Introduces random errors into a specified column of a DataFrame.
    Errors are percentage changes based on the original value.

    Args:
    df (DataFrame): The DataFrame containing the data.
    column_name (str): The column to introduce errors in.
    min_error (int): Minimum error percentage.
    max_error (int): Maximum error percentage.
    """
    # Applying random error
    df[column_name] = df[column_name].apply(
        lambda x: x * (1 + random.uniform(min_error / 100, max_error / 100))
    )
    return df


def download_blob_to_temporary_file(bucket_name, source_blob_name):
    """Downloads a blob from the bucket into a temporary file."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    # Download the blob into the temporary file
    blob.download_to_file(temp_file)

    # Close the temporary file
    temp_file.close()

    print(f"Blob {source_blob_name} downloaded to temporary file {temp_file.name}.")
    return temp_file.name


def test_functions():
    date = datetime(2015, 5, 1)
    bucket_name = "north-dakota-daily-oil-gas-data-monthly"
    source_blob_name = f"{date:%Y-%m-01/%Y_%m}.xlsx"
    temp_file_name = download_blob_to_temporary_file(bucket_name, source_blob_name)
    print(f"Reading: Temporary file path: {temp_file_name}")
    df = pd.read_excel(temp_file_name, sheet_name='Oil')
    # Introduce errors in the 'Gas' column
    df_with_errors = introduce_errors(df, 'Gas', -10, 10)
    print(df_with_errors)

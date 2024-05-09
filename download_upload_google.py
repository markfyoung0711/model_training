import os
import requests
import tempfile

from google.cloud import storage
from datetime import datetime
from urllib.parse import urlparse


def download_file_from_uri(uri, file):
    response = requests.get(uri, stream=True)
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)


def upload_file_to_gcs(file_path, file_date, bucket_name, file_name):
    # Create a client object
    client = storage.Client()

    # Get the bucket object, create it if it doesn't exist
    try:
        bucket = client.get_bucket(bucket_name)
    except Exception as e:
        print(f"Bucket {bucket_name} does not exist, creating it...\n{e}")
        bucket = client.create_bucket(bucket_name)

    # Construct the path on Google Cloud Storage
    gcs_path = f'{file_date:%Y-%m-%d}/{file_name}'

    # Upload the file to Google Cloud Storage
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(file_path)

    print(f'File uploaded to {gcs_path}')


# Example usage:
uri = 'https://www.dmr.nd.gov/oilgas/mpr/2015_05.xlsx'
file_name = os.path.basename(urlparse(uri).path)
with tempfile.NamedTemporaryFile() as file:
    download_file_from_uri(uri, file)
    file.flush()
    upload_file_to_gcs(file.name, datetime(2015, 5, 1), 'north-dakota-daily-oil-gas-data-monthly', file_name)

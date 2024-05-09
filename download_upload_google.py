import os
import requests
import tempfile

from google.cloud import storage
from datetime import datetime
from dateutil.relativedelta import relativedelta
from urllib.parse import urlparse


def download_file_from_uri(uri, file):
    response = requests.get(uri, stream=True)
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)


def upload_file_to_gcs(file_path, file_date, bucket_name, file_name, dry_run=False):

    # Construct the path on Google Cloud Storage
    gcs_path = f'{file_date:%Y-%m-%d}/{file_name}'

    if not dry_run:
        # Create a client object
        client = storage.Client()

        # Get the bucket object, create it if it doesn't exist
        try:
            bucket = client.get_bucket(bucket_name)
        except Exception as e:
            print(f"Bucket {bucket_name} does not exist, creating it...\n{e}")
            bucket = client.create_bucket(bucket_name)


        # Upload the file to Google Cloud Storage
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(file_path)

    print(f'File uploaded to {gcs_path}')


def generate_month_first_days(start_date, end_date):
    # Initialize the list to store the dates
    dates = []

    # Iterate over the months
    current_date = start_date
    while current_date <= end_date:
        # Add the first day of the month to the list
        dates.append(current_date)
        # Move to the next month
        current_date += relativedelta(months=1)

    return dates

def store_north_dakota_files(start_date, end_date):
    # first day of every month since and including 2015-05-01 until today
    dates = generate_month_first_days(start_date, end_date)
    for current_date in dates:
        uri = f'https://www.dmr.nd.gov/oilgas/mpr/{current_date:%Y_%m}.xlsx'
        file_name = os.path.basename(urlparse(uri).path)
        with tempfile.NamedTemporaryFile() as file:
            try:
                download_file_from_uri(uri, file)
                file.flush()
            except Exception as e:
                print(f'cannot load file for {uri}\n{e}')
                next
            upload_file_to_gcs(file.name, current_date, 'north-dakota-daily-oil-gas-data-monthly', file_name)

store_north_dakota_files(start_date=datetime(2015, 5, 1), end_date=datetime.today())

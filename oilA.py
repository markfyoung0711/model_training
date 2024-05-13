'''
This program trains BigQuery using an AUTOENCODE model using North Dakota monthly well production daily summaries.
The data elements being trained are Oil, Wtr, Days, Runs, Gas, GasSold and Flared
The timeseries is the ReportDate field.
The key of the well is the API_WELLNO.

'''
import pandas as pd

from google.cloud import bigquery, error_reporting
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account

import gen_errors

# constants for communicating with google api
PROJECT_ID = 'vertex-422616'
DATASET_ID = 'oil_production_dataset'
MODEL_ID = 'oil_production_model'
TABLE_ID = 'oil_production_table'

# constants about the input test data (from an Excel file)
XLSX_FILE_PATH = '/Users/markyoung/oil_data/north-dakota-dmr-nd-gov/2023_01.xlsx'
CREDENTIALS = '/Users/markyoung/.config/gcloud/application_default_credentials.json'
SA_CREDENTIALS = '/Users/markyoung/models/vertex-422616-fecb43e3649a.json'
DATE_FIELD_NAME = 'ReportDate'
OAUTH_CREDENTIALS_FILE = '/Users/markyoung/models/client_secret_913999341270-kgi6b8kvcom433pa5lhveil600um92b7.apps.googleusercontent.com.json'


def cleanup(df):
    # Clean up data
    for col in ['API_WELLNO', 'Oil', 'Wtr', 'Days', 'Runs', 'Gas', 'GasSold', 'Flared']:
        df.loc[df[col] == 'NR', col] = 0
        df.loc[df[col].isnull(), col] = 0

    return df


def train_time_series_model(introduce_errors=False):
    # Load data from XLSX file
    oil_df = pd.read_excel(XLSX_FILE_PATH, sheet_name='Oil')
    skimmed_crude_recovery_df = pd.read_excel(XLSX_FILE_PATH, sheet_name='SkimmedCrudeRecovery', dtype=str)

    # Merge dataframes
    df = pd.concat([oil_df, skimmed_crude_recovery_df], axis=0)

    # Convert date column to datetime format
    df[DATE_FIELD_NAME] = pd.to_datetime(df[DATE_FIELD_NAME])

    df = df.rename(columns={DATE_FIELD_NAME: 'date'})

    '''
    # load credentials
    # A local server is used as the callback URL in the auth flow.
    appflow = flow.InstalledAppFlow.from_client_secrets_file(
        OAUTH_CREDENTIALS_FILE, scopes=["https://www.googleapis.com/auth/bigquery"]
    )

    # This launches a local server to be used as the callback URL in the desktop
    # app auth flow. If you are accessing the application remotely, such as over
    # SSH or a remote Jupyter notebook, this flow will not work. Use the
    # `gcloud auth application-default login --no-browser` command or workload
    # identity federation to get authentication tokens, instead.
    appflow.run_local_server()

    credentials = appflow.credentials

    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
    '''

    credentials = service_account.Credentials.from_service_account_file(SA_CREDENTIALS)
    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

    dataset_ref = client.dataset(DATASET_ID)
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        print('creating dataset')
        client.create_dataset(DATASET_ID)

    table_ref = dataset_ref.table(TABLE_ID)
    client.delete_table(table_ref)
    client.create_table(table_ref)

    # load only necessary fields
    fields = ['date', 'API_WELLNO', 'Oil', 'Wtr', 'Days', 'Runs', 'Gas', 'GasSold', 'Flared']
    df = df[fields]

    # Data Cleanup
    df = cleanup(df)

    if introduce_errors:
        df = gen_errors.introduce_errors(df, 'Gas', min_error=-10, max_error=10)

    # Upload data to BigQuery
    # Generate the schema dynamically
    schema = [
        bigquery.SchemaField('date', 'TIMESTAMP'),
        bigquery.SchemaField('API_WELLNO', 'FLOAT64'),
        bigquery.SchemaField('Oil', 'FLOAT64'),
        bigquery.SchemaField('Wtr', 'FLOAT64'),
        bigquery.SchemaField('Days', 'FLOAT64'),
        bigquery.SchemaField('Runs', 'FLOAT64'),
        bigquery.SchemaField('Gas', 'FLOAT64'),
        bigquery.SchemaField('GasSold', 'FLOAT64'),
        bigquery.SchemaField('Flared', 'FLOAT64'),
    ]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV
    )

    # Load the data via the client
    load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    load_job.result()

    # Use BigQuery ML to create a time series forecasting model
    query = f"""
        CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.{MODEL_ID}`
        OPTIONS (
            MODEL_TYPE='AUTOENCODER',
            ACTIVATION_FN='RELU'
        ) AS (
            SELECT
                API_WELLNO,
                Oil,
                Wtr,
                Days,
                Runs,
                Gas,
                GasSold,
                Flared
            FROM
                `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        )

    """

    try:
        job = client.query(query)
        job.result()
    except Exception as e:
        print(f'reporting exception:\n{e}')
        err_client = error_reporting.Client()
        err_client.report_exception()


train_time_series_model(introduce_errors=True)

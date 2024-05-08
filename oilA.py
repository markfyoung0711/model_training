'''
This program trains BigQuery using an AUTOENCODE model using North Dakota monthly well production daily summaries.
The data elements being trained are Oil, Wtr, Days, Runs, Gas, GasSold and Flared
The timeseries is the ReportDate field.
The key of the well is the API_WELLNO.

'''
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# constants for communicating with google api
PROJECT_ID = 'vertex-422616'
DATASET_ID = 'oil_production_dataset'
MODEL_ID = 'oil_production_model'
TABLE_ID = 'oil_production_table'

# constants about the input test data (from an Excel file)
XLSX_FILE_PATH = '/Users/markyoung/oil_data/north-dakota-dmr-nd-gov/2023_01.xlsx'
CREDENTIALS = '/Users/markyoung/.config/gcloud/application_default_credentials.json'
DATE_FIELD_NAME = 'ReportDate'


def train_time_series_model():
    # Load data from XLSX file
    oil_df = pd.read_excel(XLSX_FILE_PATH, sheet_name='Oil')
    skimmed_crude_recovery_df = pd.read_excel(XLSX_FILE_PATH, sheet_name='SkimmedCrudeRecovery', dtype=str)

    # Merge dataframes
    df = pd.concat([oil_df, skimmed_crude_recovery_df], axis=0)

    # Convert date column to datetime format
    df[DATE_FIELD_NAME] = pd.to_datetime(df[DATE_FIELD_NAME])

    df = df.rename(columns={DATE_FIELD_NAME: 'date'})

    client = bigquery.Client(project=PROJECT_ID)

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

    # Clean up data
    for col in ['API_WELLNO', 'Oil', 'Wtr', 'Days', 'Runs', 'Gas', 'GasSold', 'Flared']:
        df.loc[df[col] == 'NR', col] = 0
        df.loc[df[col].isnull(), col] = 0

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
            ACTIVATION_FN='RELU',
            HIDDEN_UNITS=[64, 32],
            BATCH_SIZE=128
        ) AS (
            SELECT
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

    job = client.query(query)
    job.result()


train_time_series_model()

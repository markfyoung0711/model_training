import requests
from simple_salesforce import Salesforce

consumer_key = "3MVG9JJwBBbcN47I3w518okLS8zi5zEs.920B0Q.1LFO5_SqS25kJntY.HzS.ifPLPgUqkRH1CJl8hFzlj1ML"
consumer_secret = "9ECFE10BA3196E317699518DB0E1FB144602A3E5A07E0582F0F4EBE28E3613BF"

oauth_endpoint = f'https://login.salesforce.com/services/oauth2/authorize?response_type=code&client_id={consumer_key}&redirect_uri=https://login.salesforce.com/services/oauth2/success'

response = requests.get(oauth_endpoint)
# redirect
response = requests.get(response.url)
print(response.text)

# Replace these with your own values
username = 'mark.francis.young@creative-impala-wo5j6l.com'
password = 'MFQ9ryh@xph6buc3dwb'
instance_url = 'https://creative-impala-wo5j6l-dev-ed.trailblaze.lightning.force.com'

sf = Salesforce(username=username,
                password=password,
                consumer_key=consumer_key,
                consumer_secret=consumer_secret)

# If the authentication is successful, the sf object will be created and you can use it to query Salesforce
print(sf)

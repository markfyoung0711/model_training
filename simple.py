import requests
from simple_salesforce import Salesforce

consumer_key = 
consumer_secret = 

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

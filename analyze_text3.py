'''
This code calculates the accuracy of the model on the test set by comparing the predicted labels with the true labels.

As for telling Google how the model did in recognizing things in the text, you can use the Google Cloud AI Platform's Natural Language API to analyze the text and provide feedback on the model's performance. The Natural Language API provides a range of tools for text analysis, including entity recognition, sentiment analysis, and topic modeling.

You can use the API to analyze the text and get feedback on the model's performance by creating a LanguageServiceClient instance and calling the analyze_entities method, for example:

This code analyzes the text and prints out the recognized entities, along with their types.
You can use this information to evaluate the model's performance and provide feedback to Google.
'''

from google.cloud import language

client = language.LanguageServiceClient()

# Load your text data into a Pandas DataFrame
df = pd.read_csv('your_text_data.csv')

# Preprocess the text data
text_data = df['text_column_name'].tolist()

response = client.analyze_entities(
    document=language.Document(content=text_data, type_=language.Document.Type.PLAIN_TEXT),
    encoding_type=language.EncodingType.UTF8,
)

entities = response.entities
for entity in entities:
    print(entity.name, entity.type_)

from google.cloud import language

client = language.LanguageServiceClient()


def test_entities(text_data):

    response = client.analyze_entities(
        document=language.Document(content=text_data, type_=language.Document.Type.PLAIN_TEXT),
        encoding_type=language.EncodingType.UTF8,
    )

    entities = response.entities
    for entity in entities:
        print(entity.name, entity.type_)


def test_sentiment(text_data):

    print(f'{text_data} is being analyzed for sentiment')
    response = client.analyze_sentiment(
        document=language.Document(content=text_data, type_=language.Document.Type.PLAIN_TEXT),
        encoding_type=language.EncodingType.UTF8,
    )

    sentiment = response.document_sentiment
    if sentiment.score > 0:
        print("Positive sentiment")
    elif sentiment.score < 0:
        print("Negative sentiment")
    else:
        print("Neutral sentiment")
    print(f'{sentiment} is the sentiment info')


text_data = '''
The CEO of Google, Sundar Pichai, announced a new product today.
'''

test_entities(text_data)
test_sentiment("Wow, that was really awful toast!")
test_sentiment("That car is fantastic!")
test_sentiment("Well, it wasn't as if it was bad")
test_sentiment("I cannot say it wasn't the best I've ever had")

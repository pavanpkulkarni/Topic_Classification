from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

# Instantiates a client
client = language.LanguageServiceClient()

def AnalyzeText(text):
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    response = client.analyze_entities(document=document)

    entity_type = ('UNKNOWN', 'PERSON', 'LOCATION', 'ORGANIZATION',
                   'EVENT', 'WORK_OF_ART', 'CONSUMER_GOOD', 'OTHER')

    for entity in response.entities:
        print('=' * 20)
        print('         name: {0}'.format(entity.name))
        print('         type: {0}'.format(entity_type[entity.type]))
        print('     metadata: {0}'.format(entity.metadata))
        print('     salience: {0}'.format(entity.salience))
        print('=' * 20)



str = '''Researchers find "simple" way to hack Amazon Key '''

AnalyzeText(str)

#Question to prof : Do we need to submit the entities for all the titles from Mongo? or sample of top n titles will do?

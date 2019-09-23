
def get_labels(dataset='FERPlus'):
    if dataset == 'FERPlus':
        return ['neutral', 'happiness', 'surprise', 'sadness',
                'anger', 'disgust', 'fear', 'contempt']
    elif dataset == 'FER2013':
        return ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise',
                'neutral']
    elif dataset == 'IMDB':
        return ['man', 'woman']
    else:
        raise ValueError('Invalid dataset:', dataset)

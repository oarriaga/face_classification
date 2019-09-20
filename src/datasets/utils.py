
def get_labels(dataset='FERPlus'):
    if dataset == 'FERPlus':
        return ['neutral', 'happiness', 'surprise', 'sadness',
                'anger', 'disgust', 'fear', 'contempt']
    elif dataset == 'IMDB':
        return ['man', 'woman']
    else:
        raise ValueError('Invalid dataset:', dataset)

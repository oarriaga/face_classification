from scipy.io import loadmat

dataset_path = '../datasets/imdb_crop/imdb.mat'
dataset = loadmat(dataset_path)
image_names_array = dataset['imdb'][0][0][2][0]
gender_classes = dataset['imdb'][0][0][3][0].tolist()
image_names = []
for image_name_arg in range(image_names_array.shape[0]):
    image_name = image_names_array[image_name_arg][0]
    image_names.append(image_name)
data =  dict(zip(image_names, gender_classes))



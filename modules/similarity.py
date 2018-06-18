import numpy as np
import pickle

from modules import img_util
from modules import net


def similarity_on_single_img(img_path, model):
    test_img = img_util.get_single_img(img_path, grey=True)
    test_img = np.array(test_img, 'uint8')
    test_img = np.expand_dims(test_img, axis=2)
    test_img = np.expand_dims(test_img, axis=0)

    predict = model.predict_proba(test_img, batch_size=10)
    print(predict, np.argmax(predict))

    with open('class_dict.pkl', 'rb') as f:
        name_dict = pickle.load(f)
    print(name_dict)
    found_person = name_dict[np.argmax(predict)]
    print("Looks like " + found_person)

    """
    match = ''
    unmatch = ''
    for name, idx in train_data.class_indices.items():  # for name, age in list.items():  (for Python 3.x)
        if idx == (predict.argmax(axis=-1)):
            match = name
        if idx == (predict.argmin(axis=-1)):
            unmatch = name

    print("most similar: " + match)
    print("least similar: " + unmatch)
    """

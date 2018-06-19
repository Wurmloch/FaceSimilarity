import numpy as np
import pickle
import os
from PIL import Image
import webbrowser

from modules import img_util
from modules import net

norm_face_folder = "E:\\EigeneDateien\\Dokumente\\Projekte\\image_recognition\\thumbnails_features_deduped_publish\\norm"


def draw_stacked_faces(person):
    img_dir = os.path.join(norm_face_folder, person)
    img_list = os.listdir(img_dir)
    pil_img_list = list(map(lambda x: Image.open(os.path.join(img_dir, x)), img_list))
    idx = 0
    while len(pil_img_list) > 1:
        if idx + 1 < len(pil_img_list):
            pil_img_list[idx] = Image.blend(pil_img_list[idx], pil_img_list[idx + 1], 0.5)
            del pil_img_list[idx + 1]
            idx += 1
        else:
            idx = 0

    pil_img_list[0].show()


def similarity_on_single_img(img_path, model):
    test_img = img_util.get_single_img(img_path, grey=True)
    test_img = [np.array(test_img, 'uint8')]
    test_img = np.asarray(test_img)
    test_img = test_img[:, :, :, np.newaxis] / 255.0

    predict = model.predict_proba(test_img, batch_size=1, verbose=0)
    # only 1 image -- thus first array el
    predict = predict[0]

    best_results = np.argpartition(predict, -5)[-5:]
    best_results = np.flipud(best_results)
    worst_result = np.argmin(predict)

    with open('class_dict.pkl', 'rb') as f:
        name_dict = pickle.load(f)

    found_person = name_dict[np.argmax(predict)]
    print("Looks like " + found_person + "! \n")
    print("best results: " + ", ".join(list(map(lambda x: name_dict[x], best_results))))
    print(np.array(predict)[best_results])
    print("worst result: " + name_dict[worst_result])
    print(predict[worst_result])

    draw_stacked_faces(found_person)

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

import numpy as np
import pickle
import os
from PIL import Image

norm_face_folder = "C:\\Users\\Dominik\\Documents\\Projekte\\Bilderkennung\\cfw_norm\\norm"


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


def similarity_on_img_path(img_path, model):
    try:
        raw_img = Image.open(img_path).convert('L')
        similarity_on_single_img(raw_img, model)
    except OSError:
        return


def similarity_on_single_img(raw_img, model):
    sim_img = [np.array(raw_img, 'uint8')]
    sim_img = np.asarray(sim_img)
    sim_img = sim_img[:, :, :, np.newaxis] / 255.0

    prediction = model.predict(sim_img, batch_size=1, verbose=0)
    # only 1 image -- thus first array el
    prediction = prediction[0]

    best_results = np.argpartition(prediction, -5)[-5:]
    best_results = np.flipud(best_results)
    worst_result = np.argmin(prediction)

    with open('class_dict.pkl', 'rb') as f:
        name_dict = pickle.load(f)

    found_person = name_dict[np.argmax(prediction)]
    print("Looks like " + found_person + "! \n")
    print("best results: " + ", ".join(list(map(lambda x: name_dict[x], best_results))))
    print(np.array(prediction)[best_results])
    print("worst result: " + name_dict[worst_result])
    print(prediction[worst_result])

    draw_stacked_faces(found_person)

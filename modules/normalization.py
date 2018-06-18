import os
import dlib
import cv2
import numpy as np
import time
import shutil

from modules import progress

cur_dir = os.path.dirname(__file__)
predictor_path = os.path.join(cur_dir, '../assets/predictor/face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

INV_TEMPLATE = np.float32([
    (-0.04099179660567834, -0.008425234314031194, 2.575498465013183),
    (0.04062510634554352, -0.009678089746831375, -1.2534351452524177),
    (0.0003666902601348179, 0.01810332406086298, -0.32206331976076663)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

#: Landmark indices corresponding to the inner eyes and bottom lip.
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

#: Landmark indices corresponding to the outer eyes and nose.
OUTER_EYES_AND_NOSE = [36, 45, 33]


def find_landmarks(rgb_img, bb):
    """
    Find the landmarks of a face.
    :param rgb_img: RGB image to process. Shape: (height, width, 3)
    :type rgb_img: numpy.ndarray
    :param bb: Bounding box around the face to find landmarks for.
    :type bb: dlib.rectangle
    :return: Detected landmark locations.
    :rtype: list of (x,y) tuples
    """
    assert rgb_img is not None
    assert bb is not None

    points = predictor(rgb_img, bb)
    # return list(map(lambda p: (p.x, p.y), points.parts()))
    return [(p.x, p.y) for p in points.parts()]


# pylint: disable=dangerous-default-value
def align(img_dim, rgb_img, bb=None,
          landmarks=None, landmark_indices=INNER_EYES_AND_BOTTOM_LIP,
          skip_multi=False, scale=1.0):
    r"""align(imgDim, rgbImg, bb=None, landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP)
    Transform and align a face in an image.
    :param img_dim: The edge length in pixels of the square the image is resized to.
    :type img_dim: int
    :param rgb_img: RGB image to process. Shape: (height, width, 3)
    :type rgb_img: numpy.ndarray
    :param bb: Bounding box around the face to align. \
               Defaults to the largest face.
    :type bb: dlib.rectangle
    :param landmarks: Detected landmark locations. \
                      Landmarks found on `bb` if not provided.
    :type landmarks: list of (x,y) tuples
    :param landmark_indices: The indices to transform to.
    :type landmark_indices: list of ints
    :param skip_multi: Skip image if more than one face detected.
    :type skip_multi: bool
    :param scale: Scale image before cropping to the size given by imgDim.
    :type scale: float
    :return: The aligned RGB image. Shape: (imgDim, imgDim, 3)
    :rtype: numpy.ndarray
    """
    assert img_dim is not None
    assert rgb_img is not None
    assert landmark_indices is not None

    if bb is None:
        bb = get_largest_face_bounding_box(rgb_img, skip_multi)
        if bb is None:
            return None, None

    if landmarks is None:
        landmarks = find_landmarks(rgb_img, bb)

    np_landmarks = np.float32(landmarks)
    np_landmark_indices = np.array(landmark_indices)

    # pylint: disable=maybe-no-member
    H = cv2.getAffineTransform(np_landmarks[np_landmark_indices],
                               img_dim * MINMAX_TEMPLATE[np_landmark_indices] * scale + img_dim * (1 - scale) / 2)
    thumbnail = cv2.warpAffine(rgb_img, H, (img_dim, img_dim))

    return thumbnail, landmarks


def get_all_face_bounding_boxes(rgb_img):
    """
    Find all face bounding boxes in an image.
    :param rgb_img: RGB image to process. Shape: (height, width, 3)
    :type rgb_img: numpy.ndarray
    :return: All face bounding boxes in an image.
    :rtype: dlib.rectangles
    """
    assert rgb_img is not None

    try:
        return detector(rgb_img, 1)
    except Exception as e:  # pylint: disable=broad-except
        print("Warning: {}".format(e))
        # In rare cases, exceptions are thrown.
        return []


def get_largest_face_bounding_box(rgb_img, skip_multi=False):
    """
    Find the largest face bounding box in an image.
    :param rgb_img: RGB image to process. Shape: (height, width, 3)
    :type rgb_img: numpy.ndarray
    :param skip_multi: Skip image if more than one face detected.
    :type skip_multi: bool
    :return: The largest face bounding box in an image, or None.
    :rtype: dlib.rectangle
    """
    assert rgb_img is not None

    faces = get_all_face_bounding_boxes(rgb_img)
    if (not skip_multi and len(faces) > 0) or len(faces) == 1:
        return max(faces, key=lambda rect: rect.width() * rect.height())
    else:
        return None


def normalize_face(face_img, size, should_show, skip_multi=False):
    """
    normalize a given image
    :param face_img: RGB image to process. Shape: (height, width, 3)
    :type face_img: numpy.ndarray
    :param size: size for the output image in height & width px
    :type size: number
    :param should_show: if the normalized face should be shown in a frame
    :type should_show: boolean
    :param skip_multi: Skip image if more than one face detected.
    :type skip_multi: bool
    :return: landmarks and aligned face
    """
    bb = get_largest_face_bounding_box(face_img, skip_multi=skip_multi)

    aligned, landmarks = align(size, face_img, bb, landmark_indices=INNER_EYES_AND_BOTTOM_LIP, skip_multi=skip_multi)
    if aligned is not None:
        if should_show:
            cv2.imshow("aligned", aligned)
        return landmarks, aligned
    return None, None


def norm_image(image_root, norm_root, image_dir, image, undetected_dir):
    """
    finds a normalized image in the given source image
    :param image_root: the image root where all image folders are stored in
    :type image_root: str
    :param norm_root: the root folder of the normalized images
    :type norm_root: str
    :param image_dir: the current image directory with the  person name
    :type image_dir: str
    :param image: the image name
    :type image: str
    :param undetected_dir: the folder to save images with undetected faces to
    :type undetected_dir: str
    :return:
    """
    cur_image_path = os.path.join(image_root, image_dir, image)
    crop_dir = os.path.join(norm_root, image_dir)
    norm_file_path = os.path.join(crop_dir, image)

    if not os.path.exists(norm_file_path):
        image_data = cv2.imread(os.path.join(image_root, image_dir, image))
        if image_data is not None:
            # image dimensions --> 96 x 96
            landmarks, detected = normalize_face(image_data, 128, False, skip_multi=True)

            # move image with undetected face to separate folder
            if detected is None or len(detected) == 0:
                print('no face detected in ' + norm_file_path)
                undetected_sub_folder = os.path.join(undetected_dir, image_dir)
                if not os.path.exists(undetected_sub_folder):
                    os.mkdir(undetected_sub_folder)
                shutil.move(cur_image_path, os.path.join(undetected_sub_folder, image))
                return False
            else:
                if not os.path.exists(crop_dir):
                    os.makedirs(crop_dir)
                cv2.imwrite(norm_file_path, detected)
                return True
    else:
        print(norm_file_path + ' already exists')


def move_to(from_path, to_path):
    shutil.move(from_path, to_path)


def start_normalization():
    """
    initiates the normalization phase for your images, also initiates a logger to follow the batch process
    """
    dir_name = input('Enter image root folder:\n--> ')
    image_root = os.path.abspath(dir_name)
    norm_dir = input('Enter the normalized images output folder:\n--> ')
    norm_root = os.path.abspath(norm_dir)
    min_image_count = int(input('Enter the minimum images per folder to normalize:\n--> '))
    image_dir_list = os.listdir(dir_name)

    print('Normalizing all images... ')
    start_time = time.time()

    undetected_dir = os.path.join(image_root, '_undetected')
    if not os.path.isdir(undetected_dir):
        os.mkdir(undetected_dir)
    if not os.path.isdir(norm_root):
        os.makedirs(norm_root)

    for index, image_dir in enumerate(image_dir_list):
        progress.set_progress(1, len(image_dir_list), index + 1)
        image_list = os.listdir(os.path.join(dir_name, image_dir))

        if len(image_list) >= min_image_count:
            for image in image_list:
                if os.path.isfile(os.path.join(image_root, image_dir, image)):
                    norm_image(image_root, norm_root, image_dir, image, undetected_dir)

    print('Completed normalization in {} seconds'.format(time.time() - start_time))


if __name__ == 'main':
    start_normalization()

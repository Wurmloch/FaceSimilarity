from keras.models import load_model

from modules import normalization, capture, net, similarity


def decision():
    """
    starts the user input, what tool to run
    """
    answer = input('Which tool should be executed?\n(1) Normalization\n(2) Train and evaluate NN\n'
                   '(3) Live Capture\n(4) Similarity on images\n--> ')
    if answer == '1':
        normalization.start_normalization()
    elif answer == '2':
        net.init_net()
    elif answer == '3':
        capture.live_capture()
    elif answer == '4':
        full_model = load_model("model_full")
        img_path = input("Enter the image path:\n--> ")
        similarity.similarity_on_single_img(img_path, full_model)
    elif answer == 'x':
        return
    else:
        decision()
    # execute again recursively
    decision()


decision()

if __name__ == 'main':
    decision()

from keras.models import load_model

from modules import normalization, capture, net, similarity

global num_classes
num_classes = 1578


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
        try:
            full_model = load_model("model_full")
        except OSError:
            full_model = net.load_weights(num_classes)
        capture.live_capture(full_model)
    elif answer == '4':
        try:
            full_model = load_model("model_full")
        except OSError:
            full_model = net.load_weights(num_classes)
        img_path = input("Enter the image path:\n--> ")
        similarity.similarity_on_img_path(img_path, full_model)
    elif answer == 'x':
        return
    else:
        decision()
    # execute again recursively
    decision()


decision()

if __name__ == 'main':
    decision()

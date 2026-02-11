import numpy as np

def preprocess_image(image):
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict(processed_image):
    # ⚠️ Fake prediction pour test
    import random
    prob = random.random()

    if prob > 0.5:
        return 1, prob
    else:
        return 0, 1 - prob
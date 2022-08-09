import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


def getPrediction(filename):

    classes = ["1", "2", "3", "4", "5",
               "6", "7", "8", "9", "10"]

    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])

    my_model = load_model("model/model_save_fromTrained.h5")

    SIZE = 64
    img_path = 'static/images/'+filename
    img = np.asarray(Image.open(img_path).convert('L').resize((SIZE, SIZE)))

    img = img/255.

    img = np.expand_dims(img, axis=0)

    pred = my_model.predict(img)

    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("The number is:", pred_class)
    return pred_class


# a = getPrediction('8.png')

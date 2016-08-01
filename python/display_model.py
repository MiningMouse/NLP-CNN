from keras.utils.visualize_util import plot
from keras.models import model_from_json

model = model_from_json(open('imdb_CNN-rand7_arch.json').read())
plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)
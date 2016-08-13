from keras.utils.visualize_util import plot
from keras.models import model_from_json

model = model_from_json(open('twitter_CNN-static7_arch.json').read())
plot(model, to_file='model_twitter_static.png', show_shapes=True, show_layer_names=True)
import os
import matplotlib.pyplot as plt
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as sk_mae
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate
from keras.models import Model
from keras.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np

img_dir = "boneage2/"
csv_path = "boneage2.csv"
age_df = pd.read_csv(csv_path)
age_df['path'] = age_df['id'].map(lambda x: img_dir+"{}.png".format(x))
age_df['exists'] = age_df['path'].map(os.path.exists)
age_df['gender'] = age_df['male'].map(lambda x: "male" if x else "female")
mu = age_df['boneage'].mean()
sigma = age_df['boneage'].std()
age_df['zscore'] = age_df['boneage'].map(lambda x: (x-mu)/sigma)
age_df.dropna(inplace=True)


age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)

raw_train_df, test_df = train_test_split(age_df, 
                                         test_size=0.2,
                                         random_state=2019,
                                         stratify=age_df['boneage_category'])

raw_train_df, valid_df = train_test_split(raw_train_df, 
                                          test_size=0.1,
                                          random_state=2019,
                                          stratify=raw_train_df['boneage_category'])

train_df = raw_train_df.groupby(['boneage_category', 'male']).apply(lambda x: x.sample(100, replace = True)).reset_index(drop=True)
print(train_df)

train_size = train_df.shape[0]
valid_size = valid_df.shape[0]
test_size = test_df.shape[0]

print("# Training images:   {}".format(train_size))
print("# Validation images: {}".format(valid_size))
print("# Testing images:    {}".format(test_size))

def preprocess_input_inception(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

IMG_SIZE = (224, 224)
def datagen(in_df, path_col, y_col, gender_col, batch_size, **dflow_args):
    img_data_gen = ImageDataGenerator(samplewise_center=False, 
                                      samplewise_std_normalization=False,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      height_shift_range=0.15,
                                      width_shift_range=0.15,
                                      rotation_range=45,
                                      shear_range=0.01,
                                      fill_mode='reflect',
                                      zoom_range=0.2,
                                      preprocessing_function=preprocess_input_inception)
    base_dir = os.path.dirname(in_df[path_col].values[0])
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode='sparse', batch_size=batch_size, shuffle=True, **dflow_args)
    df_gen.filenames = [x.split("/")[1] for x in in_df[path_col].values]
    df_gen.classes = np.column_stack([in_df[y_col].values,in_df[gender_col].values.astype(float)])
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    for batch in df_gen:
      yield [batch[0], batch[1][:, 1]], batch[1][:, 0]

train_gen = datagen(train_df, 
                    path_col='path',
                    y_col='zscore',
                    gender_col='male',
                    batch_size=10,
                    target_size=IMG_SIZE,
                    color_mode='rgb',
                    seed=2275)

valid_X, valid_Y = next(datagen(valid_df, 
                        path_col='path',
                        y_col='zscore',
                        gender_col='male',
                        batch_size=valid_size,
                        target_size=IMG_SIZE,
                        color_mode='rgb',
                        seed=2275))

IMG_SHAPE = valid_X[0][0, :, :, :].shape


img = Input(shape=IMG_SHAPE)
gender = Input(shape=(1,))
cnn_vec = InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')(img)
cnn_vec = GlobalAveragePooling2D()(cnn_vec)
cnn_vec = Dropout(0.2)(cnn_vec)
gender_vec = Dense(32, activation='relu')(gender)
features = Concatenate(axis=-1)([cnn_vec, gender_vec])
dense_layer = Dense(1024, activation='relu')(features)
dense_layer = Dropout(0.2)(dense_layer)
dense_layer = Dense(1024, activation='relu')(dense_layer)
dense_layer = Dropout(0.2)(dense_layer)
output_layer = Dense(1, activation='linear')(dense_layer)
bone_age_model = Model(inputs=[img, gender], outputs=output_layer)

def mae_months(in_gt, in_pred):
    return mean_absolute_error(mu+sigma*in_gt, mu+sigma*in_pred)




#lrlist = [0.0005, 0.0003, 0.0001, 0.00005, 0.00003, 0.00001, 0.000005, 0.000003, 0.000001]

lrlist = [0.000003, 0.000001]
patiencelist = [5, 10, 15, 20, 25]
i = 35

for i in range(1, 36):
        bone_age_model.compile(optimizer='adam', loss='mse', metrics=[mae_months])
        print('Age ' + str(i))
        weight_path='models/age-' + str(i) + '.hdf5'

        bone_age_model.load_weights(weight_path)

        test_X, test_Y = next(datagen(test_df,
                                      path_col='path',
                                      y_col='zscore',
                                      gender_col='male',
                                      batch_size=test_size,
                                      target_size=IMG_SIZE,
                                      color_mode='rgb',
                                      seed=2275))

        pred_val_Y = mu+sigma*bone_age_model.predict(x=valid_X, batch_size=25, verbose=0)
        valid_Y_months = mu+sigma*valid_Y
        print("Valid MAE: " + str(sk_mae(valid_Y_months, pred_val_Y)))

        pred_Y = mu+sigma*bone_age_model.predict(x=test_X, batch_size=25, verbose=0)
        test_Y_months = mu+sigma*test_Y
        print("Test MAE: " + str(sk_mae(test_Y_months, pred_Y)))
        print('-' * 10)

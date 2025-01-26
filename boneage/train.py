import os
import matplotlib.pyplot as plt
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from keras.applications.densenet import DenseNet169
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate
from keras.models import Model
from keras.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np

img_dir = "images/"
csv_path = "boneage2.csv"
age_df = pd.read_csv(csv_path)
age_df['path'] = age_df['id'].map(lambda x: img_dir + 'boneage2/' + "{}.png".format(x))
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
    df_gen.filenames = [x.split("/")[2] for x in in_df[path_col].values]
    df_gen.classes = np.column_stack([in_df[y_col].values, in_df[gender_col].values.astype(float)])
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
#cnn_vec = InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')(img)
cnn_vec = DenseNet169(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')(img)
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




lrlist = [0.0005, 0.0003, 0.0001, 0.00005, 0.00003, 0.00001, 0.000005, 0.000003, 0.000001]
patiencelist = [10]
i = 0

for lr in lrlist:
    for patience in patiencelist:
        adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        bone_age_model.compile(optimizer=adam, loss='mse', metrics=[mae_months])
        bone_age_model.summary()
        i = i + 1
        weight_path='models/age2-' + str(i) + '.hdf5'
        checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=patience, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.000001)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=50)
        callbacks_list = [checkpoint, early, reduceLROnPlat]
        if not os.path.exists(weight_path):
            bone_age_model.fit_generator(train_gen,
                                         steps_per_epoch=train_size/10,
                                         validation_data=(valid_X, valid_Y),
                                         epochs=200,
                                         callbacks=callbacks_list,
                                         verbose=1)
        bone_age_model.load_weights(weight_path)

        test_X, test_Y = next(datagen(test_df,
                                      path_col='path',
                                      y_col='zscore',
                                      gender_col='male',
                                      batch_size=test_size,
                                      target_size=IMG_SIZE,
                                      color_mode='rgb',
                                      seed=2275))

        pred_val_Y = mu+sigma*bone_age_model.predict(x=valid_X,batch_size=25,verbose=1)
        valid_Y_months = mu+sigma*valid_Y
        print("Valid MAE: " + str(sk_mae(valid_Y_months,pred_val_Y)))

        pred_Y = mu+sigma*bone_age_model.predict(x=test_X,batch_size=25,verbose=1)
        test_Y_months = mu+sigma*test_Y
        print("Test MAE: " + str(sk_mae(test_Y_months,pred_Y)))


# fig, ax1 = plt.subplots(1,1, figsize = (6,6))
# ax1.plot(test_Y_months, pred_Y, 'r.', label = 'predictions')
# ax1.plot(test_Y_months, test_Y_months, 'b-', label = 'actual')
# ax1.legend()
# ax1.set_xlabel('Actual Age (Months)')
# ax1.set_ylabel('Predicted Age (Months)')
# ord_idx = np.argsort(test_Y)
# ord_idx = ord_idx[np.linspace(0, len(ord_idx)-1, num=16).astype(int)]
# fig, m_axs = plt.subplots(2, 4, figsize = (16, 32))
# for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):
#     c_ax.imshow(test_X[0][idx, :, :, 0], cmap='bone')
#     title = 'Age: %2.1f\nPredicted Age: %2.1f\nGender: ' % (test_Y_months[idx], pred_Y[idx])
#     if test_X[1][idx]==0:
#         title += "Female\n"
#     else:
#         title += "Male\n"
#     c_ax.set_title(title)
#     c_ax.axis('off')
# plt.show()

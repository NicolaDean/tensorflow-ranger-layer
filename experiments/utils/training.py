import tensorflow.keras as keras
import tensorflow as tf

def train_model(model,DATASET,NUM_CATEGORIES,EPOCHS):

    print(f'MODEL HAS : [{NUM_CATEGORIES}] classes')
   
    (x_train,x_val,y_train,y_val,DATASET_NAME) = DATASET

    y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
    y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)


    history = model.fit(x_train, y_train, batch_size=64, epochs=EPOCHS, validation_data=(x_val, y_val))



def load_model(MODEL = "vgg16",NUM_CLASSES=3,INPUT_SHAPE=(32,32,3)):

    if MODEL == "vgg16":
        head = tf.keras.applications.VGG16(include_top=False,weights="imagenet",input_shape=INPUT_SHAPE,classes=NUM_CLASSES)
        preprocess_fn = tf.keras.applications.vgg16.preprocess_input
    elif MODEL == "vgg19":
        head = tf.keras.applications.VGG19(include_top=False,weights="imagenet",input_shape=INPUT_SHAPE,classes=NUM_CLASSES)
        preprocess_fn = tf.keras.applications.vgg19.preprocess_input
    elif MODEL == "xception":#SERVE RESIZE INPUT
        head = tf.keras.applications.Xception(include_top=False,weights="imagenet",input_shape=INPUT_SHAPE,classes=NUM_CLASSES)
        preprocess_fn = tf.keras.applications.xception.preprocess_input
    elif MODEL == "resnet50":
        head = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",input_shape=INPUT_SHAPE,classes=NUM_CLASSES)
        preprocess_fn = tf.keras.applications.resnet50.preprocess_input
    elif MODEL == "inceptionv3":
        head = tf.keras.applications.InceptionV3(include_top=False,weights="imagenet",input_shape=INPUT_SHAPE,classes=NUM_CLASSES)
        preprocess_fn = tf.keras.applications.inception_v3.preprocess_input
    elif MODEL == "mobilenetv2":
        head = tf.keras.applications.MobileNetV2(include_top=False,weights="imagenet",input_shape=INPUT_SHAPE,classes=NUM_CLASSES)
        preprocess_fn = tf.keras.applications.mobilenet.preprocess_input
    elif MODEL == "efficientnet":
        head = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False,weights="imagenet",input_shape=INPUT_SHAPE,classes=NUM_CLASSES)
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
    elif MODEL == "convnettiny":
        head = tf.keras.applications.ConvNeXtTiny(include_top=False,weights="imagenet",input_shape=INPUT_SHAPE,classes=NUM_CLASSES)
        preprocess_fn = tf.keras.applications.convnext.preprocess_input
    elif MODEL == "densenet":
        head = tf.keras.applications.DenseNet121(include_top=False,weights="imagenet",input_shape=INPUT_SHAPE,classes=NUM_CLASSES)
        preprocess_fn = tf.keras.applications.densenet.preprocess_input
    elif MODEL == "nasnet":
        head = tf.keras.applications.NASNetMobile(include_top=False,weights="imagenet",input_shape=INPUT_SHAPE,classes=NUM_CLASSES)
        preprocess_fn = tf.keras.applications.nasnet.preprocess_input
        
    else:
        print(f"\033[0;31mSELECT A VALID MODEL\033[0m")
        exit()

    layers = tf.keras.layers
    

    model = layers.BatchNormalization()(head.output)
    model = layers.Flatten()(model)
    model = layers.Dense(512, activation='relu')(model)
    out   = layers.Dense(NUM_CLASSES, activation='softmax')(model)

    model = tf.keras.Model(inputs=head.input, outputs=out)
    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    #model.summary()

    return model,preprocess_fn
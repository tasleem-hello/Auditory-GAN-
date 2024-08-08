class ACGAN:

  def __init__(self, rows=28, cols=28, channels=1):
    self.rows = rows
    self.cols = cols
    self.channels = channels
    self.shape = (self.rows, self.cols, self.channels)
    self.latent_size = 100
    self.sample_rows = 1
    self.sample_cols = 2
    self.sample_path = 'images'
    self.num_classes = 2

    #optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
    optimizer = Adam(0.0002, 0.5)

    image_shape = self.shape
    seed_size = self.latent_size

    #Get the discriminator and generator Models
    print("Build Discriminator")
    self.discriminator = self.build_discriminator()

    self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print("Build Generator")

    self.generator = self.build_generator()

    random_input = Input(shape=(seed_size,))
    label = Input(shape=(1,))

    #Pass noise/random_input and label as input to the generator
    generated_image = self.generator([random_input,label])

    #Put discriminator.trainable to False. We do not want to train the discriminator at this point in time
    self.discriminator.trainable =False

    #Pass generated image and label as input to the discriminator
    validity, label_out = self.discriminator(generated_image)
    print('validity',validity)
    print('label_discr',label_out)
    #Pass radom input and label as input to the combined model
    self.combined_model = Model([random_input,label], [validity,label_out])
    self.combined_model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

  def build_discriminator(self):

    input_shape = self.shape
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=2, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128,(3,3), strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3,3), strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))

    output = model
    inp = Input(shape=input_shape)

    model.summary()

    input_image = Input(shape=input_shape)

    #Extrating features from the model
    features = model(input_image)

    # AC GAN has 2 outputs, 1 for real or Fake using sigmoid activation. Another for class prediction using softmax.

    validity = Dense(1, activation='sigmoid', name='Dense_validity')(features)
    print('validity',validity)
    aux = Dense(self.num_classes, activation='softmax', name ='Dense_Aux')(features)
    print('aux',aux)
    return Model(input_image,[validity,aux])

  def build_generator(self):

    seed_size = self.latent_size
    model = Sequential()
    model.add(Dense(7*7*256, input_dim=seed_size))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Reshape((7,7,256)))
    model.add(Dropout(0.4))

    model.add(Conv2DTranspose(128,(5,5),padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(64,(3,3),padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(32,(3,3),padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(1,(3,3),padding='same'))
    model.add(Activation('sigmoid'))

    noise = Input(shape=(seed_size,))
    label = Input(shape = (1,), dtype='int32')

    label_embeddings = Flatten()(Embedding(self.num_classes,self.latent_size)(label))

    model_input = multiply([noise,label_embeddings])

    generated_image = model(model_input)

    model.summary()

    return(Model([noise,label],generated_image))
### Final Validation accuracy for Base Network ###

Accuracy on test data is: 82.52

### Model definition (model.add... ) with output channel size and receptive field ###

model = Sequential()
model.add(SeparableConv2D(48, kernel_size=(3,3), input_shape=(32, 32, 3))) #30X30X48  #RF=3
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

model.add(SeparableConv2D(48, kernel_size=(3,3))) #28X28X48   #RF=5
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2))) #14X14X48  #RF=6


model.add(SeparableConv2D(64, kernel_size=(3,3), border_mode='same')) #12X12X96  #RF=10
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(SeparableConv2D(96, kernel_size=(3,3))) #12X12X96  #RF=14
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2))) #6X6X96 #RF=16
model.add(Dropout(0.2))

model.add(SeparableConv2D(128, kernel_size=(3,3), border_mode='same')) #4X4X192  #RF=24
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(SeparableConv2D(192, kernel_size=(3,3))) #4X4X192  #RF=32
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2))) #2X2X192  #RF=36


model.add(SeparableConv2D(num_classes,kernel_size=(2,2))) #1X1X10  #RF=44
model.add(Flatten())

model.add(Activation('softmax'))


### 50 epoch logs ###


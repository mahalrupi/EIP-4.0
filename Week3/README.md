### Final Validation accuracy for Base Network ###

Accuracy on test data is:
### 83.30 ###

### Model definition (model.add... ) with output channel size and receptive field ###

model = Sequential()
model.add(SeparableConv2D(48, kernel_size=(3,3), input_shape=(32, 32, 3))) #30X30X48 #RF=3X3
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(48, kernel_size=(3,3))) #28X28X48 #RF=5X5
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) #14X14X48 #RF=6X6
model.add(Dropout(0.25))

model.add(SeparableConv2D(96, kernel_size=(3,3), border_mode='same')) #14X14X96 #RF=10X10
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(96, kernel_size=(3,3))) #12X12X96 #RF=14X14
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=(2, 2))) #6X6X96 #RF=16X16
model.add(Dropout(0.25))

model.add(SeparableConv2D(192, kernel_size=(3,3), border_mode='same')) #6X6X192 #RF=24X24
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(192, kernel_size=(3,3))) #4X4X192 #RF=32X32
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) #2X2X192 #RF=36X36
model.add(Dropout(0.25))

model.add(SeparableConv2D(num_classes,kernel_size=(2,2))) #1X1X10 #RF=44X44
model.add(Flatten())

model.add(Activation('softmax'))


### 50 epoch logs ###
Epoch 1/50
390/390 [==============================] - 30s 77ms/step - loss: 1.4969 - acc: 0.4584 - val_loss: 1.3550 - val_acc: 0.5192

Epoch 2/50
390/390 [==============================] - 27s 68ms/step - loss: 1.1446 - acc: 0.5919 - val_loss: 1.0737 - val_acc: 0.6171

Epoch 3/50
390/390 [==============================] - 27s 69ms/step - loss: 1.0193 - acc: 0.6380 - val_loss: 0.9340 - val_acc: 0.6674

Epoch 4/50
390/390 [==============================] - 27s 68ms/step - loss: 0.9389 - acc: 0.6660 - val_loss: 1.0707 - val_acc: 0.6322

Epoch 5/50
390/390 [==============================] - 27s 68ms/step - loss: 0.8717 - acc: 0.6929 - val_loss: 0.9505 - val_acc: 0.6730

Epoch 6/50
390/390 [==============================] - 27s 69ms/step - loss: 0.8197 - acc: 0.7115 - val_loss: 0.8282 - val_acc: 0.7151

Epoch 7/50
390/390 [==============================] - 27s 69ms/step - loss: 0.7810 - acc: 0.7253 - val_loss: 0.8656 - val_acc: 0.7018

Epoch 8/50
390/390 [==============================] - 27s 69ms/step - loss: 0.7455 - acc: 0.7377 - val_loss: 0.7635 - val_acc: 0.7338

Epoch 9/50
390/390 [==============================] - 27s 69ms/step - loss: 0.7193 - acc: 0.7476 - val_loss: 0.6917 - val_acc: 0.7619

Epoch 10/50
390/390 [==============================] - 27s 69ms/step - loss: 0.6950 - acc: 0.7552 - val_loss: 0.6756 - val_acc: 0.7650

Epoch 11/50
390/390 [==============================] - 27s 69ms/step - loss: 0.6754 - acc: 0.7632 - val_loss: 0.7502 - val_acc: 0.7462

Epoch 12/50
390/390 [==============================] - 27s 69ms/step - loss: 0.6566 - acc: 0.7702 - val_loss: 0.6869 - val_acc: 0.7586

Epoch 13/50
390/390 [==============================] - 27s 69ms/step - loss: 0.6413 - acc: 0.7749 - val_loss: 0.6770 - val_acc: 0.7673

Epoch 14/50
390/390 [==============================] - 27s 69ms/step - loss: 0.6285 - acc: 0.7779 - val_loss: 0.6898 - val_acc: 0.7621

Epoch 15/50
390/390 [==============================] - 27s 69ms/step - loss: 0.6092 - acc: 0.7877 - val_loss: 0.6384 - val_acc: 0.7820

Epoch 16/50
390/390 [==============================] - 27s 69ms/step - loss: 0.6042 - acc: 0.7880 - val_loss: 0.6063 - val_acc: 0.7894

Epoch 17/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5857 - acc: 0.7942 - val_loss: 0.6491 - val_acc: 0.7798

Epoch 18/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5866 - acc: 0.7947 - val_loss: 0.6532 - val_acc: 0.7800

Epoch 19/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5653 - acc: 0.8026 - val_loss: 0.6244 - val_acc: 0.7882

Epoch 20/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5571 - acc: 0.8059 - val_loss: 0.6256 - val_acc: 0.7889

Epoch 21/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5466 - acc: 0.8084 - val_loss: 0.7123 - val_acc: 0.7629

Epoch 22/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5433 - acc: 0.8102 - val_loss: 0.5354 - val_acc: 0.8140

Epoch 23/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5337 - acc: 0.8120 - val_loss: 0.6248 - val_acc: 0.7889

Epoch 24/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5229 - acc: 0.8167 - val_loss: 0.6716 - val_acc: 0.7757

Epoch 25/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5186 - acc: 0.8170 - val_loss: 0.6699 - val_acc: 0.7712

Epoch 26/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5173 - acc: 0.8177 - val_loss: 0.5303 - val_acc: 0.8157

Epoch 27/50
390/390 [==============================] - 27s 69ms/step - loss: 0.5114 - acc: 0.8211 - val_loss: 0.5293 - val_acc: 0.8217

Epoch 28/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4995 - acc: 0.8243 - val_loss: 0.6009 - val_acc: 0.7966

Epoch 29/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4940 - acc: 0.8277 - val_loss: 0.5733 - val_acc: 0.8062

Epoch 30/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4925 - acc: 0.8263 - val_loss: 0.5542 - val_acc: 0.8134

Epoch 31/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4856 - acc: 0.8301 - val_loss: 0.5840 - val_acc: 0.8068

Epoch 32/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4776 - acc: 0.8339 - val_loss: 0.5462 - val_acc: 0.8127

Epoch 33/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4785 - acc: 0.8339 - val_loss: 0.5543 - val_acc: 0.8103

Epoch 34/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4710 - acc: 0.8334 - val_loss: 0.5275 - val_acc: 0.8230

Epoch 35/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4668 - acc: 0.8366 - val_loss: 0.5336 - val_acc: 0.8224

Epoch 36/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4666 - acc: 0.8377 - val_loss: 0.5190 - val_acc: 0.8244

Epoch 37/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4600 - acc: 0.8398 - val_loss: 0.5344 - val_acc: 0.8187

Epoch 38/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4551 - acc: 0.8407 - val_loss: 0.5338 - val_acc: 0.8213

Epoch 39/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4566 - acc: 0.8395 - val_loss: 0.5269 - val_acc: 0.8249

Epoch 40/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4477 - acc: 0.8435 - val_loss: 0.5432 - val_acc: 0.8198

Epoch 41/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4453 - acc: 0.8434 - val_loss: 0.5281 - val_acc: 0.8202

Epoch 42/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4420 - acc: 0.8439 - val_loss: 0.4743 - val_acc: 0.8390

Epoch 43/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4343 - acc: 0.8459 - val_loss: 0.5318 - val_acc: 0.8165

Epoch 44/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4349 - acc: 0.8476 - val_loss: 0.5344 - val_acc: 0.8253

Epoch 45/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4314 - acc: 0.8484 - val_loss: 0.5168 - val_acc: 0.8240

Epoch 46/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4261 - acc: 0.8488 - val_loss: 0.5871 - val_acc: 0.8065

Epoch 47/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4299 - acc: 0.8486 - val_loss: 0.4812 - val_acc: 0.8366

Epoch 48/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4214 - acc: 0.8529 - val_loss: 0.4879 - val_acc: 0.8390

Epoch 49/50
390/390 [==============================] - 27s 68ms/step - loss: 0.4221 - acc: 0.8517 - val_loss: 0.5315 - val_acc: 0.8265

Epoch 50/50
390/390 [==============================] - 27s 69ms/step - loss: 0.4187 - acc: 0.8529 - val_loss: 0.4905 - val_acc: 0.8330

Model took 1343.49 seconds to train
Accuracy on test data is: 83.30

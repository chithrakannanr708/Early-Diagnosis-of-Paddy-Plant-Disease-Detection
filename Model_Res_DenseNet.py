import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input

from Evaluation import evaluation


def dilated_residual_dense_attention_block(x, filters, dilation_rate=2):
    # Dense block
    y = layers.Conv2D(3, (3, 3), padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    y = layers.BatchNormalization()(y)
    x = layers.Concatenate()([x, y])

    # Residual connection
    res = layers.Conv2D(3, (1, 1), padding='same')(x)

    # Attention mechanism
    attn = layers.GlobalAveragePooling2D()(res)
    attn = layers.Dense(filters // 4, activation='relu')(attn)
    attn = layers.Dense(16, activation='sigmoid')(attn)
    attn = layers.Multiply()([res, attn])

    # Residual addition
    x = layers.Add()([x, attn])
    x = layers.ReLU()(x)

    return x


def Model_Res_DenseNet(input_shape, num_classes, Test_Data, Test_Target, sol=None):
    if sol is None:
        sol =[5, 5, 5]
    for n in range(input_shape.shape[0]):
        input = input_shape[n]
        inputs = Input(shape=input.shape)
        x = dilated_residual_dense_attention_block
        # Initial Conv Layer
        x = layers.Conv2D(3, (3, 3), padding='same')(inputs)
        # Classification Head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes.shape[1], activation='softmax')(x)

        model = models.Model(inputs, outputs)
        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()
        model.fit(input_shape, num_classes, epochs=1) # epochs=sol[0], steps_per_epoch=sol[1]
        pred = model.predict(Test_Data)
        Eval = evaluation(pred, Test_Target)

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return Eval, pred

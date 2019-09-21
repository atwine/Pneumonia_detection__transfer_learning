# Pneumonia_detection_using_transfer_learning

In this repository, I am working on a Kaggle dataset for Pneumonia.

I am using tranfer learning using some modles that have already pre-trained by Google an some other entities to help practice transfer learning.

So far the models used are:
1- Resnet50
2- EfficientnetB5
3-

The list will continue to grow as time and resources allow.

Transfer learning can be seen in some specific cells of the notebooks

```
# setup model
    from efficientnet.model import EfficientNetB5
    base_model = EfficientNetB5(input_tensor = input_tensor,weights='imagenet', include_top=False) #include_top=False excludes final FC layer
    model = add_new_last_layer(base_model, nb_classes)

```

Here we use the EfficientnetB5 as the base model and its used in the function add_new_last_layer

```

def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = Dropout(0.5)(x)
    x = AveragePooling2D((8, 8), border_mode='valid', name='avg_pool')(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    predictions = Dense(2, activation='sigmoid')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

```

That is where the magic happens. Some of the notebooks are still under training so ignore some of the mistakes that you may notice.

To work with this repo just do the following:

1- Clone the repo to your hard disk

```
git clone https://github.com/atwine/Pneumonia_detection_resnet50_transfer_learning.git

```

2- Fire up jupyter notebooks and run the notebook locally.

I have made it easy to follow through the steps so run the cells as they come. To use the kaggle function please find an authentication token from your kaggle account and keep it in the same folder you downloaded.

Happy coding!

# Symbol Recognition

## Training and predicting

Train with `python train.py <epochs>`. Default epochs is 50.

Predict with `python predict.py <path_to_image / path_to_validation_directory> <path_to_model>`

## Dataset

The dataset must follow the following file structure:
```
/dataset
|
|---- /category1
|     |
|     |---- /image1.png
|     |---- /image2.png
|     ...
|
|---- /category2
|     |
|     |---- /image1.png
|     |---- /image2.png
|     ...
...
```
Same goes for the `validation_data` directory.
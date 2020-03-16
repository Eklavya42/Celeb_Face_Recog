# Face Recognition Using Facenet

This repository is forked from original [facenet implementation](https://github.com/davidsandberg/facenet). I have added some files for custom data based face detection.

### Dataset taken from kaggle : [Face Celeb Data](https://www.kaggle.com/tanvirshanto/facedatasetcelebrity)

#### Dataset Compressed [here](data/celeb.tar.gz)


> Folder Structure for data

```
data
└───Celeb
│   └───raw
│   |   └───Brad
│   |   └───Daniel
│   |   └───...
|   |
│   └───processed
    |   └───Brad
    |   └───Daniel
    |   └───...

```



### Setting Python Path

Set the environment variable PYTHONPATH to point to the src directory of the cloned repo. This is typically done something like this

```
export PYTHONPATH=[...]/facenet/src
```

where [...] should be replaced with the directory where the cloned facenet repo resides.


## Preprocessing


#### Aligning the Dataset
Alignment of the FaceCeleb dataset can be done using `align_dataset_mtcnn` in the `align` module.

Can be done like this :

```
python src/align/align_dataset_mtcnn.py \
./data/celeb/raw \
./data/celeb/processed \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.25

```

#### Download Pre-trained Model

Download and extract the model  given here : [Facenet Model](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) and
create a folder Models and place it in the facenet folder. After extracting the archive there should be a new folder 20180402-114759 with the contents
 ```
20180402-114759.pb
model-20180402-114759.ckpt-275.data-00000-of-00001
model-20180402-114759.ckpt-275.index
model-20180402-114759.meta
```

## Train the Model on your Data

Now we can train the data on the model using `classifier.py`  in the src folder.

Can be done like this:

```
python src/classifier.py \
TRAIN ./data/celeb/processed \
./Models/facenet/20180402-114759.pb \
./Models/celeb/celeb.pkl \
--batch_size 100
```

## Face Recognition

Run the `recog_celeb.py` file for face detection on a video file.

```
>python recog_celeb.py
```

Path for input and ouput video file can be edited in `recog_celeb.py` file.



## Result

![Ouput gif](ouput.gif "Output GIF")

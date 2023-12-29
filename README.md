# TmNoiseClassification

> Deep Learning Project for Sound Classification Based on ECAPA-TDNN

#### Description

After a period of studying machine learning techniques, IAfter a period of studying machine learning techniques, I
decided to make my own deep learning project for sound classification. The name is TieMClassification.

## Implementation Steps

1. Prepare the dataset
2. Data preprocessing
3. Build ecapa-tdnn model
4. Train the model
5. Apply the model

## Model deployment correlation

> the ecapa_tdnn_80.model is my train result.The next time I need to step up in RockChip board.

Now want to deploy on the RockChip board.But no support pytorch.Now I need some toolkit help my model change to knn
model and setup in the RockChip board.
This is official description by [knn toolkit](https://github.com/rockchip-linux/rknn-toolkit2?tab=readme-ov-file).

## dataSets Collection

### UrbanSound8K

Urban Sound 8K is an audio dataset that contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes:
air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and
street_music. The classes are drawn from the urban sound taxonomy. All excerpts are taken from field recordings uploaded
to www.freesound.org.

### ESC-50

The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of
environmental sound classification.
The dataset consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class)
loosely arranged into 5 major categories:

| <sub>Animals</sub>          | <sub>Natural soundscapes & water sounds </sub> | <sub>Human, non-speech sounds</sub> | <sub>Interior/domestic sounds</sub> | <sub>Exterior/urban noises</sub> |
|:----------------------------|:-----------------------------------------------|:------------------------------------|:------------------------------------|:---------------------------------|
| <sub>Dog</sub>              | <sub>Rain</sub>                                | <sub>Crying baby</sub>              | <sub>Door knock</sub>               | <sub>Helicopter</sub></sub>      |
| <sub>Rooster</sub>          | <sub>Sea waves</sub>                           | <sub>Sneezing</sub>                 | <sub>Mouse click</sub>              | <sub>Chainsaw</sub>              |
| <sub>Pig</sub>              | <sub>Crackling fire</sub>                      | <sub>Clapping</sub>                 | <sub>Keyboard typing</sub>          | <sub>Siren</sub>                 |
| <sub>Cow</sub>              | <sub>Crickets</sub>                            | <sub>Breathing</sub>                | <sub>Door, wood creaks</sub>        | <sub>Car horn</sub>              |
| <sub>Frog</sub>             | <sub>Chirping birds</sub>                      | <sub>Coughing</sub>                 | <sub>Can opening</sub>              | <sub>Engine</sub>                |
| <sub>Cat</sub>              | <sub>Water drops</sub>                         | <sub>Footsteps</sub>                | <sub>Washing machine</sub>          | <sub>Train</sub>                 |
| <sub>Hen</sub>              | <sub>Wind</sub>                                | <sub>Laughing</sub>                 | <sub>Vacuum cleaner</sub>           | <sub>Church bells</sub>          |
| <sub>Insects (flying)</sub> | <sub>Pouring water</sub>                       | <sub>Brushing teeth</sub>           | <sub>Clock alarm</sub>              | <sub>Airplane</sub>              |
| <sub>Sheep</sub>            | <sub>Toilet flush</sub>                        | <sub>Snoring</sub>                  | <sub>Clock tick</sub>               | <sub>Fireworks</sub>             |
| <sub>Crow</sub>             | <sub>Thunderstorm</sub>                        | <sub>Drinking, sipping</sub>        | <sub>Glass breaking</sub>           | <sub>Hand saw</sub>              |

### The torch model transformation

The rknnToolkit needs to run on Ubuntu 20.4, and is currently built using py311 + 1.6.0 to successfully convert the
model. The Oracle VM VirtualBox found so far is when a virtual machine is used. There will be strange problems with
model conversion, and illegal instructions will suddenly occur when the weight layer is too sparse. At present, there is
no problem with the replacement of the VMware Workstation Pro. The input tensor is initially [1,80,3000 * 160]. Model
conversion failed. Conversion errors may occur because the data is too large. Now it becomes 500 \ * 160. It is the
audio of 5S. At present, we are waiting for the deployment of connecting boards.
# AnimeFacesGenerate: Use Deep Convolution Generative Adversarial Networks(DCGAN)

## Target
Use Deep Convolution Generative Adversarial Networks(DCGAN) for Anime-faces Generation.
The structure of the Generator G(z) is like the following:
![image](imgs/Gz_structure.png)

## Dataset
50,000 Anime-faces pictures of size 96\*96\*3.(would be finally transformed to 64\*64\*3)
* Samples
![image](imgs/DatasetSamples.png)

## Instruments
* pytorch 1.0.1
* matplotlib 2.2.2

## Usage
```
usage: python DCGAN.py
```
## Result
Epoch_numbers : 100
![image](imgs/fake_samples_epoch_099.jpg)
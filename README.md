# Ball Tracking
* Detect the presence of the colored ball using computer vision techniques. 
* Track the ball as it moves around in the video or camera

This is part of the bigger project called Remotely Operated underwater Vehicle (ROV). I am a member of computer vision team for this project. Specifically, I am in charge of the object tracking algorithm of the vehicle. Other 2 people are responsible for depth map algorithm and data transfer.

## Demo
https://youtu.be/FnqBbC2Fa6w
[![Demo CountPages alpha](https://j.gifs.com/y8Kz1V.gif)](https://youtu.be/FnqBbC2Fa6w)

## How to run
Open CV 3.0
```
g++ main.cpp Utils.cpp Contour.cpp Noise.cpp `pkg-config --cflags --libs opencv`
```

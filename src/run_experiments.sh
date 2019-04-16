#!/bin/bash


# XCEPTION EXPERIMENTS --------------------------------------------------------

##---------------------
## IMAGE SIZE: 48x48 ##
##---------------------

### small kernels sizes
python3 train.py -m xception -d FERPlus -is 48 -sk 8 16 -bd 32
python3 train.py -m xception -d FERPlus -is 48 -sk 8 16 -bd 32 32 
python3 train.py -m xception -d FERPlus -is 48 -sk 8 16 -bd 32 32 64 
python3 train.py -m xception -d FERPlus -is 48 -sk 8 16 -bd 32 32 64 64 
python3 train.py -m xception -d FERPlus -is 48 -sk 8 16 -bd 32 32 64 64 128
python3 train.py -m xception -d FERPlus -is 48 -sk 8 16 -bd 32 32 64 64 128 128

### medium kernel sizes
python3 train.py -m xception -d FERPlus -is 48 -sk 16 32 -bd 64
python3 train.py -m xception -d FERPlus -is 48 -sk 16 32 -bd 64 64
python3 train.py -m xception -d FERPlus -is 48 -sk 16 32 -bd 64 64 128
python3 train.py -m xception -d FERPlus -is 48 -sk 16 32 -bd 64 64 128 128
python3 train.py -m xception -d FERPlus -is 48 -sk 16 32 -bd 64 64 128 128 256
python3 train.py -m xception -d FERPlus -is 48 -sk 16 32 -bd 64 64 128 128 256 256

### big kernel sizes
python3 train.py -m xception -d FERPlus -is 48 -sk 32 64 -bd 128 
python3 train.py -m xception -d FERPlus -is 48 -sk 32 64 -bd 128 128 
python3 train.py -m xception -d FERPlus -is 48 -sk 32 64 -bd 128 128 256 
python3 train.py -m xception -d FERPlus -is 48 -sk 32 64 -bd 128 128 256 256 
python3 train.py -m xception -d FERPlus -is 48 -sk 32 64 -bd 128 128 256 256 512 
python3 train.py -m xception -d FERPlus -is 48 -sk 32 64 -bd 128 128 256 256 512 512

##-----------------------
## IMAGE SIZE: 128x128 ##
##-----------------------

## small kernel sizes
python3 train.py -m xception -d FERPlus -is 128 -sk 8 16 -bd 32
python3 train.py -m xception -d FERPlus -is 128 -sk 8 16 -bd 32 32 
python3 train.py -m xception -d FERPlus -is 128 -sk 8 16 -bd 32 32 64 
python3 train.py -m xception -d FERPlus -is 128 -sk 8 16 -bd 32 32 64 64 
python3 train.py -m xception -d FERPlus -is 128 -sk 8 16 -bd 32 32 64 64 128
python3 train.py -m xception -d FERPlus -is 128 -sk 8 16 -bd 32 32 64 64 128 128
python3 train.py -m xception -d FERPlus -is 128 -sk 8 16 -bd 32 32 64 64 128 128 256

## medium kernel sizes
python3 train.py -m xception -d FERPlus -is 128 -sk 16 32 -bd 64
python3 train.py -m xception -d FERPlus -is 128 -sk 16 32 -bd 64 64
python3 train.py -m xception -d FERPlus -is 128 -sk 16 32 -bd 64 64 128
python3 train.py -m xception -d FERPlus -is 128 -sk 16 32 -bd 64 64 128 128
python3 train.py -m xception -d FERPlus -is 128 -sk 16 32 -bd 64 64 128 128 256
python3 train.py -m xception -d FERPlus -is 128 -sk 16 32 -bd 64 64 128 128 256 256
python3 train.py -m xception -d FERPlus -is 128 -sk 16 32 -bd 64 64 128 128 256 256 512

## big kernels sizes
python3 train.py -m xception -d FERPlus -is 128 -sk 32 64 -bd 128 
python3 train.py -m xception -d FERPlus -is 128 -sk 32 64 -bd 128 128 
python3 train.py -m xception -d FERPlus -is 128 -sk 32 64 -bd 128 128 256 
python3 train.py -m xception -d FERPlus -is 128 -sk 32 64 -bd 128 128 256 256 
python3 train.py -m xception -d FERPlus -is 128 -sk 32 64 -bd 128 128 256 256 512 
python3 train.py -m xception -d FERPlus -is 128 -sk 32 64 -bd 128 128 256 256 512 512
python3 train.py -m xception -d FERPlus -is 128 -sk 32 64 -bd 128 128 256 256 512 512 1024


# VGG EXPERIMENTS --------------------------------------------------------

##---------------------
## IMAGE SIZE: 48x48 ##
##---------------------

### small kernels sizes
python3 train.py -m vgg -d FERPlus -is 48 -sk 8 16 -bd 32
python3 train.py -m vgg -d FERPlus -is 48 -sk 8 16 -bd 32 32 
python3 train.py -m vgg -d FERPlus -is 48 -sk 8 16 -bd 32 32 64 
python3 train.py -m vgg -d FERPlus -is 48 -sk 8 16 -bd 32 32 64 64 
python3 train.py -m vgg -d FERPlus -is 48 -sk 8 16 -bd 32 32 64 64 128
python3 train.py -m vgg -d FERPlus -is 48 -sk 8 16 -bd 32 32 64 64 128 128

### medium kernel sizes
python3 train.py -m vgg -d FERPlus -is 48 -sk 16 32 -bd 64
python3 train.py -m vgg -d FERPlus -is 48 -sk 16 32 -bd 64 64
python3 train.py -m vgg -d FERPlus -is 48 -sk 16 32 -bd 64 64 128
python3 train.py -m vgg -d FERPlus -is 48 -sk 16 32 -bd 64 64 128 128
python3 train.py -m vgg -d FERPlus -is 48 -sk 16 32 -bd 64 64 128 128 256
python3 train.py -m vgg -d FERPlus -is 48 -sk 16 32 -bd 64 64 128 128 256 256

### big kernel sizes
python3 train.py -m vgg -d FERPlus -is 48 -sk 32 64 -bd 128 
python3 train.py -m vgg -d FERPlus -is 48 -sk 32 64 -bd 128 128 
python3 train.py -m vgg -d FERPlus -is 48 -sk 32 64 -bd 128 128 256 
python3 train.py -m vgg -d FERPlus -is 48 -sk 32 64 -bd 128 128 256 256 
python3 train.py -m vgg -d FERPlus -is 48 -sk 32 64 -bd 128 128 256 256 512 
python3 train.py -m vgg -d FERPlus -is 48 -sk 32 64 -bd 128 128 256 256 512 512

##-----------------------
## IMAGE SIZE: 128x128 ##
##-----------------------

## small kernel sizes
python3 train.py -m vgg -d FERPlus -is 128 -sk 8 16 -bd 32
python3 train.py -m vgg -d FERPlus -is 128 -sk 8 16 -bd 32 32 
python3 train.py -m vgg -d FERPlus -is 128 -sk 8 16 -bd 32 32 64 
python3 train.py -m vgg -d FERPlus -is 128 -sk 8 16 -bd 32 32 64 64 
python3 train.py -m vgg -d FERPlus -is 128 -sk 8 16 -bd 32 32 64 64 128
python3 train.py -m vgg -d FERPlus -is 128 -sk 8 16 -bd 32 32 64 64 128 128
python3 train.py -m vgg -d FERPlus -is 128 -sk 8 16 -bd 32 32 64 64 128 128 256

## medium kernel sizes
python3 train.py -m vgg -d FERPlus -is 128 -sk 16 32 -bd 64
python3 train.py -m vgg -d FERPlus -is 128 -sk 16 32 -bd 64 64
python3 train.py -m vgg -d FERPlus -is 128 -sk 16 32 -bd 64 64 128
python3 train.py -m vgg -d FERPlus -is 128 -sk 16 32 -bd 64 64 128 128
python3 train.py -m vgg -d FERPlus -is 128 -sk 16 32 -bd 64 64 128 128 256
python3 train.py -m vgg -d FERPlus -is 128 -sk 16 32 -bd 64 64 128 128 256 256
python3 train.py -m vgg -d FERPlus -is 128 -sk 16 32 -bd 64 64 128 128 256 256 512

## big kernels sizes
python3 train.py -m vgg -d FERPlus -is 128 -sk 32 64 -bd 128 
python3 train.py -m vgg -d FERPlus -is 128 -sk 32 64 -bd 128 128 
python3 train.py -m vgg -d FERPlus -is 128 -sk 32 64 -bd 128 128 256 
python3 train.py -m vgg -d FERPlus -is 128 -sk 32 64 -bd 128 128 256 256 
python3 train.py -m vgg -d FERPlus -is 128 -sk 32 64 -bd 128 128 256 256 512 
python3 train.py -m vgg -d FERPlus -is 128 -sk 32 64 -bd 128 128 256 256 512 512
python3 train.py -m vgg -d FERPlus -is 128 -sk 32 64 -bd 128 128 256 256 512 512 1024

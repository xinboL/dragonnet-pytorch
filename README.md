# Dragonnet-PyTorch Setup Guide

This guide will help you set up and run the Dragonnet-PyTorch project on your local machine.

## 1. Set Up Environment

### 1.1 Download Dragonnet-PyTorch
Download the repository from GitHub to your PC:  
[Dragonnet-PyTorch Repository](https://github.com/farazmah/dragonnet-pytorch/tree/master)

### 1.2 Install Dependencies
1. Open the command prompt.
2. Navigate to the project directory:
   ```bash
   cd C:\Users\x.liu\Desktop\Xinbo\dragonnet-pytorch-master\dragonnet-pytorch-master
   pip install wheel
   python setup.py bdist_wheel
   pip install dist/dragonnet-0.1-py3-none-any.whl
   
3. Install wheel package
   
   ```bash
   pip install wheel

4. Build the wheel distribution

   ```bash
   python setup.py bdist_wheel

5. Install the Dragonnet package

   ```bash
   pip install dist/dragonnet-0.1-py3-none-any.whl

## 2. Set Working Directory

In Spyder (or any Python interpreter you are using), set the working directory to …\dragonnet-pytorch-master.

## 3. Run the Main Script

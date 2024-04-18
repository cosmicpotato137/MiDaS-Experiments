# Depth Estimation with MiDaS

This project uses the MiDaS model for depth estimation from a camera feed. The model takes an RGB image as input and outputs a depth map, which estimates the distance from the camera for each pixel in the image.

Source:
[MiDaS](https://pytorch.org/hub/intelisl_midas_v2/)

## Requirements

This project requires Python and the following Python libraries installed:

- OpenCV
- PyTorch
- NumPy

You can install these dependencies using pip. I recommend using a virtual environemnt:

```bash
python -m venv MiDaSvenv
pip freeze > requirements.txt
```

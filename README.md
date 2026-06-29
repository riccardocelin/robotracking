# RoboTracking

The goal is to design a lightweight computer vision system capable of detecting and tracking sports balls in real time using a camera mounted on a Raspberry Pi. MobileNet-based object detection model was selected to balance accuracy and computational constraints, converting it for efficient inference on the device. Once an object is detected, the system computes its position in the frame and controls servomotors to physically track the object.

## Roadmap
- [x] Base inference pipeline (model loading on edge device, video frame loop, object detection)
- [ ] Model optimization (TFLite conversion, speedup inference and data wrangling)
- [ ] Controller coding
- [ ] Final optimizations

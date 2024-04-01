# POSE ESTIMATION


## Requirements
* Xcode IDE: Download Xcode IDE from the link: ​​https://xcodereleases.com/
* Cocoapods: https://guides.cocoapods.org/using/getting-started.html

## How to get the iOS code?
* Clone the project from the following repo: https://github.com/shrutisuryawanshi29/PoseEstimation_iOS
* After it gets cloned, double click the file: PoseEstimation-CoreML.xcworkspace
* This will open in your Xcode.
* Connect your Mac to your device via a cable.
* You will need your Provisioning profiles, so create that. (You may need to change your bundle identifier to a unique one)
* Run it on any target device.

## Model Conversion Code
The Model is converted into CoreML format using the following code:
```
import coremltools as ct
# Load model with pretrained weights
from super_gradients.training import models
from super_gradients.common.object_names import Models




model = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights="coco_pose")


# Prepare model for conversion
# Input size is in format of [Batch x Channels x Width x Height] where 640 is the standard COCO dataset dimensions
model.eval()
model.prep_model_for_conversion(input_size=[1, 3, 640, 640])


print("\ngoing to convert ..")
# Create dummy_input
dummy_input = torch.rand(1, 3, 640, 640)
traced_model = torch.jit.trace(model, dummy_input)
# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.
model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=dummy_input.shape)]
 )
# Save the converted model.
model.save("yolo_nas_s.mlmodel")
```


The Model is converted into ONNX format using the following code:

```
import torch
# Load model with pretrained weights
from super_gradients.training import models
from super_gradients.common.object_names import Models




model = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights="coco_pose")


# Prepare model for conversion
# Input size is in format of [Batch x Channels x Width x Height] where 640 is the standard COCO dataset dimensions
model.eval()
model.prep_model_for_conversion(input_size=[1, 3, 640, 640])


print("\ngoing to convert ..")
# Create dummy_input


dummy_input = torch.rand(1, 3, 640, 640)


# Define input / output names
input_names = ["my_input"]
output_names = ["my_output"]
model.eval()
print(model)
# Convert model to onnx
torch.onnx.export(model, dummy_input,  "yolo_nas_s.onnx",verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,)
print("\nconverted!")
```

The above is an example in which YOLO-NAS pose. Any models can be converted in the above way and used in the project.


## Branch Information
1. main - which contains the dropdown selection of models
2. AD_YoloNas - which contains the implementation of yolo_nas_pose_l

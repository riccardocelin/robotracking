to download the model:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

reference to download:
ssd_mobilenet_v2_320x320_coco17_tpu-8.tar
>> SSD MobileNet V2 320x320 trained on COCO dataset

working tree:

RoboTracking
    Computer_vision
        .py scripts
        [ssd_mobilenet_v2_320x320_coco17_tpu-8]
            [checkpoint]    # checkpoint original model
            [saved_model]   # saved_model original model file
            [TFLite]
                [coco_subsampl_val2017]         # validation set cocodataset 2017 for tflit conv purposes
                [prepro_model_nodynamicinput]   # original model preprocessed for tflite conv purposes
                ssd_mobilenet_v2_320x320_coco17_tpu.tflite      # tflite-converted model (from prepro model)



# STEP1: download the original model
# STEP2: preprocess the original model (fix the input shape at [1,320,320,3])
# STEP3: tflite-conversion of the preprocessed model

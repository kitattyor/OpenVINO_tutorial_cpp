# OpenVINO_tutorial_cpp

A complete code sample on how to use the OpenVINO toolkit to infer images using pre-trained models in ONNX format, or from the OpenVINO Model Optimizer.

### Requirements
* [Intel OpenVINO toolkit](https://software.intel.com/en-us/openvino-toolkit/choose-download)
* [jsonCPP](https://github.com/open-source-parsers/jsoncpp)
* [Boost](https://www.boost.org/)

jsonCPP and Boost are not essential for understanding the general idea of the code. 
jsonCPP is used because the class labels corresponding to the classIDs were in a json file. 
Boost is used for cross-platform filesystem handling and command line options implementation.
I used the OpenCV version that comes packed with the OpenVINO installer.

#### Environment vars
* BOOST_ROOT points to the boost directory
* BOOST_LIB64 points to the 64 bit Boost libs (most probably $(BOOST_ROOT)\stage\lib)
* OPENCV_411IR_DIR points to the OpenCV dir that ships with OpenVINO toolkit

### Usage

`ImageInferenceSample --help` for help

`ImageInferenceSample -m <Path to xml>,<Path to bin> or <Path to onnx> -i <Path to Image Dataset> -j <Path to class labels json file>`

### Test 

Assuming normal build paths (project_dir\release or debug\ImageInferenceSample.exe)

`.\ImageInferenceSample -m ..\..\test\ONNX_Models\resnet18v2.onnx -i ..\..\test\ImageNetDataset -j ..\..\test\ImageNetDataset\imagenet1000_clsid_to_human.json`

Predicted output should be something like - 

> --  Dummy image --  
Load net time taken = 499 ms  
Time taken = 26 ms  
Class ID = 398 and confidence = 17.0721  
Class name = "abacus"  
Image file = "..\..\test\ImageNetDataset\abacus.jpg"  
Time taken = 28 ms  
Class ID = 124 and confidence = 16.7009  
Class name = "crayfish, crawfish, crawdad, crawdaddy"  
Image file = "..\..\test\ImageNetDataset\bibLobster.jpg"  
Time taken = 28 ms  
Class ID = 534 and confidence = 14.1346  
Class name = "dishwasher, dish washer, dishwashing machine"  
Image file = "..\..\test\ImageNetDataset\dishwasher.jpg"  
Time taken = 30 ms  
Class ID = 120 and confidence = 16.7068  
Class name = "fiddler crab"  
Image file = "..\..\test\ImageNetDataset\fidlercrab.jpg"  
Time taken = 29 ms  
Class ID = 604 and confidence = 32.6894  
Class name = "hourglass"  
Image file = "..\..\test\ImageNetDataset\hourglass.jpg"  
Time taken = 27 ms  
Class ID = 119 and confidence = 15.943  
Class name = "rock crab, Cancer irroratus"  
Image file = "..\..\test\ImageNetDataset\rockcrab.jpg"  
Time taken = 30 ms  
Class ID = 6 and confidence = 16.713  
Class name = "stingray"  
Image file = "..\..\test\ImageNetDataset\stingray.jpg"  
Time taken = 27 ms  
Class ID = 897 and confidence = 18.8466  
Class name = "washer, automatic washer, washing machine"  
Image file = "..\..\test\ImageNetDataset\washingmachine.jpg"  
Average time taken = 28 ms  

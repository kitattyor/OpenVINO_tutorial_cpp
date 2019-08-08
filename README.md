# OpenVINO_tutorial_cpp

### Requirements
* [Intel OpenVINO toolkit](https://software.intel.com/en-us/openvino-toolkit/choose-download)
* [jsonCPP](https://github.com/open-source-parsers/jsoncpp)
* [Boost](https://www.boost.org/)

jsonCPP and Boost are not essential for understanding the general idea of the code. 
jsonCPP is used because the class names corresponding to the classIDs were in a json file. 
Boost is used for cross-platform filesystem handling and command line options implementation.
I used the OpenCV version that comes packed with the OpenVINO installer.

The project is built using MS Visual Studio 2019.

#### Environment vars
* BOOST_ROOT points to the boost directory
* BOOST_LIB64 points to the 64 bit Boost libs (most probably $(BOOST_ROOT)\stage\lib)
* OPENCV_411IR_DIR points to the OpenCV dir that ships with OpenVINO toolkit

### Usage

`ImageInferenceSample --help` for help

`ImageInferenceSample -m <Path to xml>,<Path to bin> or <Path to onnx> -i <Path to Image Dataset> -j <Path to class labels json file>`

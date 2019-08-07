// ImageInferenceSample.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>
#include <fstream>
#include <numeric>

#include "opencv2/opencv.hpp"
#include "json/json.h"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "impreproc.h"

#define WRONG_OPTIONS_SELECTED  std::cout << "Use --help to see help screen on how to use this code." << std::endl

namespace fs = boost::filesystem;
namespace po = boost::program_options;

//struct to deal with command line options
struct CMDLINEOPTS 
{
	std::string pathToModel;
	std::string pathToImageSet;
	std::string pathToClassIDJSON;
};

static CMDLINEOPTS opts;

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path& root, const std::string& ext, std::vector<fs::path>& ret)
{
	if (!fs::exists(root) || !fs::is_directory(root)) return;

	fs::recursive_directory_iterator it(root);
	fs::recursive_directory_iterator endit;

	while (it != endit)
	{
		if (fs::is_regular_file(*it) && it->path().extension() == ext) 
			ret.push_back(it->path());
		
		++it;

	}

}

//parses command line args, passes to command line struct obj if valid options, if not, gracefully exits program
int parseCommandLineArgs(int argc, char* argv[])
{

	using namespace po;
	options_description desc{ "Options" };
	
	/*boost::program_options::options_description defines a member function add() that expects a parameter of type 
	boost::program_options::option_description.You call this function to describe each command - line option.
	Instead of calling this function for every command - line option, Example 63.1 calls the member function add_options(), which makes that task easier.

	add_options() returns a proxy object representing an object of type boost::program_options::options_description.
	The type of the proxy object doesn’t matter.It’s more interesting that the proxy object simplifies defining many command - line options.
	It uses the overloaded operator operator(), which you can call to pass the required data to define a command - line option.
	This operator returns a reference to the same proxy object, which allows you to call operator() multiple times.
	
	From https://theboostcpplibraries.com/boost.program_options
	*/

	desc.add_options()
		("help,h",
			"To use this program, in cmd pass: ImageInferenceSample --model <Path_to_model_file(s)> --imageset <Path_to_imageset> --jsonfile <Path_to_json_with_classIDs>")
		("model,m", value<std::string>(), "Model Path")
		("imageset,i", value<std::string>(), "Path to Image Set folder")
		("jsonfile,j", value<std::string>(), "Path to json class labels");

	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);
	notify(vm);

	if (vm.count("help"))
	{
		std::cout << desc << std::endl;
		return 2;
	}
	else if(vm.count("model") && vm.count("imageset") && vm.count("jsonfile"))
	{
		opts.pathToModel = vm["model"].as<std::string>();
		opts.pathToImageSet = vm["imageset"].as<std::string>();
		opts.pathToClassIDJSON = vm["jsonfile"].as<std::string>();
	}
	else
	{
		WRONG_OPTIONS_SELECTED;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

int main(int argc, char * argv[])
{
	if (argc < 7)
	{
		if (argc == 2 && 
			(std::string(argv[1]) == std::string("--help") || 
				std::string(argv[1]) == std::string("-h"))) //to let help pass
		{			}
		else
		{
			WRONG_OPTIONS_SELECTED;
			return EXIT_FAILURE;
		}
		
	}

	if (parseCommandLineArgs(argc, argv) != EXIT_SUCCESS)
		return EXIT_FAILURE;

	if (!fs::exists(opts.pathToModel) ||
		!fs::is_directory(opts.pathToImageSet) ||
		!fs::exists(opts.pathToClassIDJSON))
	{
		std::cout << "One or more of the arguments are invalid" << std::endl;
		return EXIT_FAILURE;
	}

	std::string onnx_file = opts.pathToModel;
	auto net = cv::dnn::readNetFromONNX(onnx_file);

	net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	auto tp_0 = std::chrono::high_resolution_clock::now();

	//get labels from json file
	std::string classIdFile = opts.pathToClassIDJSON;
	std::ifstream ifs;
	ifs.open(classIdFile);

	Json::Reader reader;
	Json::Value dictClassID;

	if (ifs.good())
	{
		reader.parse(ifs, dictClassID);
		ifs.close();
	}
	
	std::vector<cv::Mat> imageSet;

	//get jpegs from the set
	std::string imageSetF = opts.pathToImageSet;
	std::vector<fs::path> allImageFiles;
	get_all(fs::path(imageSetF), ".jpg", allImageFiles);

	//get images
	for (auto& imageFile : allImageFiles)
	{
		cv::Mat image = cv::imread(imageFile.string(), cv::IMREAD_COLOR);
		imageSet.push_back(image);
	}
	
	std::vector<long long> timeTaken;

	//dummy image to kickstart network because net.forward() takes time for the first image
	std::cout << "  --  Dummy image -- " << std::endl;
	cv::Mat dummyImage = cv::Mat(224, 224, CV_8UC3);
	randu(dummyImage, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
	auto dummyInput = imgToBlob(dummyImage);
	net.setInput(dummyInput);
	auto dummyProb = net.forward();
	//dont care about the result

	auto tspent_0 = std::chrono::high_resolution_clock::now() - tp_0;
	auto duration_0 = std::chrono::duration_cast<std::chrono::milliseconds>(tspent_0);
	std::cout << "Load net time taken = " << duration_0.count() << " ms" << std::endl;

	for (int i = 0; i < imageSet.size(); ++i)
	{
		cv::Mat im = imageSet[i];

		if (im.empty())
			continue;

		auto input = imgToBlob(im);

		auto tp_1 = std::chrono::high_resolution_clock::now();

		net.setInput(input);
		auto prob = net.forward();

		cv::Point classIdPoint;
		double confidence;
		minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
		int classId = classIdPoint.x;

		auto tspent = std::chrono::high_resolution_clock::now() - tp_1;
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tspent);
		std::cout << "Time taken = " << duration.count() << " ms" << std::endl;
		timeTaken.push_back(duration.count());

		std::cout << "Class ID = " << classId << " and confidence = " << confidence << std::endl;
		std::cout << "Class name = " << dictClassID[std::to_string(classId)] << std::endl;
		std::cout << "Image file = " << allImageFiles[i] << std::endl;
	}

	if (timeTaken.size() == 0)
		return EXIT_FAILURE;

	auto avgTime  = std::accumulate(timeTaken.begin(), timeTaken.end(), 0) / timeTaken.size();
	std::cout << "Average time taken = " << avgTime << " ms" << std::endl;

	return EXIT_SUCCESS;
}

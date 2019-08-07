#pragma once

cv::Mat normalizeImage(cv::Mat image)
{
	cv::Mat imNorm;
	image.convertTo(imNorm, CV_32FC3, 1.0f / 255);
	return imNorm;
}

cv::Mat standardizeImage(cv::Mat image)
{
	cv::Mat standardizedImage = image;
	//BGR because Image will be read in BGR
	auto mean = cv::Scalar(0.406, 0.456, 0.485);
	auto stddev = cv::Scalar(0.225, 0.224, 0.229);

	cv::Mat subMean = image - cv::Scalar(0.406, 0.456, 0.485);
	standardizedImage = subMean / cv::Scalar(0.225, 0.224, 0.229);
	
	return standardizedImage;
}


cv::Mat imgToBlob(cv::Mat image)
{
	cv::Mat imNorm = normalizeImage(image);
	cv::Mat imStNorm = standardizeImage(imNorm);
	
	cv::Size inputSz(0, 0);

	inputSz = cv::Size(224, 224);

	auto input = cv::dnn::blobFromImage(
		imStNorm,
		1.0,
		inputSz,
		cv::Scalar(0, 0, 0),
		true, //BGR to RGB
		false,
		CV_32F
	);

	return input;

}
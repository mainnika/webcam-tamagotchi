#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class ScreenDetector
{
protected:
    static constexpr double screenIdealCoefficient = 216.0 / 106.0;
    static constexpr double screenCoefficientTolerance = 0.1;

private:
    cv::Mat edgePattern;

public:
    ScreenDetector(std::string edgePatternPath);
    virtual ~ScreenDetector() = default;
    cv::Rect getScreen(cv::Mat &image, cv::Mat &edgePattern);
    bool validateScreen(cv::Mat &image, cv::Rect screenRect);
    cv::Rect detectScreen(cv::Mat &image);
};
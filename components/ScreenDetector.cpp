#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>

#include "ScreenDetector.h"

constexpr double ScreenDetector::screenIdealCoefficient;
constexpr double ScreenDetector::screenCoefficientTolerance;

ScreenDetector::ScreenDetector(std::string edgePatternPath) : edgePattern(cv::imread(edgePatternPath))
{
    cv::cvtColor(edgePattern, edgePattern, cv::COLOR_BGR2GRAY);
}

cv::Rect ScreenDetector::getScreen(cv::Mat &image, cv::Mat &edgePattern)
{
    cv::TemplateMatchModes method = cv::TM_CCOEFF_NORMED;
    cv::Mat imageLeft = image(cv::Rect(0, 0, image.cols / 2, image.rows));
    cv::Mat imageRight = image(cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows));

    cv::Point topLeft, bottomRight;

    cv::Mat matches;
    cv::matchTemplate(imageLeft, edgePattern, matches, method);
    cv::minMaxLoc(matches, nullptr, nullptr, nullptr, &topLeft);
    topLeft.y += edgePattern.rows;

    matches = cv::Mat();
    cv::matchTemplate(imageRight, edgePattern, matches, method);
    cv::minMaxLoc(matches, nullptr, nullptr, nullptr, &bottomRight);
    bottomRight.x += imageLeft.cols + edgePattern.cols;

    return cv::Rect(topLeft, bottomRight);
}

bool ScreenDetector::validateScreen(cv::Mat &image, cv::Rect screenRect)
{
    double screenCoefficient = screenRect.width / static_cast<double>(screenRect.height);

    if (screenRect.width < screenRect.height)
    {
        return false;
    }
    if (std::abs(screenCoefficient - ScreenDetector::screenIdealCoefficient) > ScreenDetector::screenCoefficientTolerance)
    {
        return false;
    }

    return true;
}

cv::Rect ScreenDetector::detectScreen(cv::Mat &image)
{
    cv::Rect screen = this->getScreen(image, edgePattern);
    bool validScreenDetected = this->validateScreen(image, screen);
    if (validScreenDetected)
    {
        return screen;
    }

    return cv::Rect();
}

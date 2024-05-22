#include <cmath>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include "components/ScreenDetector.h"
#include "components/ScreenExtractor.h"

#define PATH_TO_PATTERN "/home/mainnika/Develop/mainnika/webcam-tamagotchi/pattern.png"

int main()
{
    auto cap = std::make_unique<cv::VideoCapture>(0);
    if (!cap->isOpened())
    {
        spdlog::error("error opening the camera-0");
        return -1;
    }

    auto screenDetector = std::make_unique<ScreenDetector>(PATH_TO_PATTERN);
    auto screenExtractor = std::make_unique<ScreenExtractor>(std::move(cap), std::move(screenDetector), 30);

    screenExtractor->start();

    while (cv::waitKey(10) != 27)
    {
        if (screenExtractor->getLastRendered().empty())
        {
            continue;
        }

        cv::imshow("last rendered", screenExtractor->getLastRendered());
    }

    screenExtractor->stop();

    return 0;
}
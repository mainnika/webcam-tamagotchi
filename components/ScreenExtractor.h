#pragma once

#include <memory>
#include <thread>
#include <array>
#include <atomic>
#include <list>
#include <deque>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include "ScreenDetector.h"

class ScreenExtractor
{
protected:
    static constexpr size_t contoursRenderHistorySize = 7;
    static constexpr size_t screenRectsSize = 128;
    static constexpr size_t screenCapBufferSize = 5;

    cv::Rect validScreenRect;

    std::deque<cv::Mat> contoursRenderHistory;
    std::deque<cv::Rect> screenRects;
    std::deque<cv::Mat> screenCapBuffer;

private:
    std::unique_ptr<cv::VideoCapture> cap;
    std::unique_ptr<ScreenDetector> screenDetector;
    std::unique_ptr<std::jthread> extractorThread;

    size_t fps;

    std::atomic_bool needStop;
    std::atomic_size_t nextRenderTime;

    std::pair<std::shared_ptr<cv::Mat>, std::shared_ptr<cv::Mat>> lastRendered;
    std::mutex lastRenderedMutex;

    void extractor();
    cv::Rect getValidScreenRect();
    cv::Mat getHistogram(cv::Mat &image);
    std::pair<size_t, std::pair<size_t,size_t>> getColorThreshhold(cv::Mat &histogram);

public:
    ScreenExtractor(std::unique_ptr<cv::VideoCapture> cap, std::unique_ptr<ScreenDetector> screenDetector, size_t fps = 30);
    virtual ~ScreenExtractor();

    void start();
    void stop();

    cv::Mat getLastRendered();
    size_t getNextRenderTime();
};

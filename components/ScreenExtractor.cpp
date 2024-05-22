#include "ScreenExtractor.h"

constexpr size_t ScreenExtractor::contoursRenderHistorySize;
constexpr size_t ScreenExtractor::screenRectsSize;
constexpr size_t ScreenExtractor::screenCapBufferSize;

ScreenExtractor::ScreenExtractor(std::unique_ptr<cv::VideoCapture> cap, std::unique_ptr<ScreenDetector> screenDetector, size_t fps)
    : validScreenRect(),
      cap(std::move(cap)),
      screenDetector(std::move(screenDetector)),
      extractorThread(nullptr),
      fps(fps),
      needStop(false),
      nextRenderTime(0),
      lastRendered(std::make_pair(std::make_shared<cv::Mat>(), std::make_shared<cv::Mat>()))
{
}

ScreenExtractor::~ScreenExtractor()
{
    if (this->extractorThread)
    {
        this->extractorThread->join();
    }
}

void ScreenExtractor::start()
{
    this->extractorThread = std::make_unique<std::jthread>(&ScreenExtractor::extractor, this);
}

void ScreenExtractor::stop()
{
    this->needStop.store(true);
}

cv::Mat ScreenExtractor::getLastRendered()
{
    std::lock_guard<std::mutex> lock(this->lastRenderedMutex);
    return *this->lastRendered.first;
}

size_t ScreenExtractor::getNextRenderTime()
{
    return this->nextRenderTime;
}

cv::Rect ScreenExtractor::getValidScreenRect()
{
    typedef std::pair<int, int> XY_t;
    typedef std::pair<XY_t, XY_t> ScreenRect_t;

    if (this->screenRects.empty())
    {
        return cv::Rect();
    }

    std::map<ScreenRect_t, size_t> screenRectsCounter;
    for (auto &screenRect : this->screenRects)
    {
        ScreenRect_t screenRectKey = ScreenRect_t(XY_t(screenRect.x, screenRect.y), XY_t(screenRect.width, screenRect.height));
        screenRectsCounter[screenRectKey]++;
    }

    auto screenRect = std::max_element(screenRectsCounter.begin(), screenRectsCounter.end(),
                                       [](const std::pair<ScreenRect_t, size_t> &a, const std::pair<ScreenRect_t, size_t> &b)
                                       { return a.second < b.second; })
                          ->first;

    return cv::Rect(screenRect.first.first, screenRect.first.second, screenRect.second.first, screenRect.second.second);
}

cv::Mat ScreenExtractor::getHistogram(cv::Mat &image)
{
    // do color histogram
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // smooth histogram
    cv::Mat histSmoothed;
    cv::GaussianBlur(hist, histSmoothed, cv::Size(21, 21), 0);

    // // draw histogram
    // int histW = 512, histH = 400;
    // int binW = cvRound((double)histW / histSize);
    // cv::Mat histImage(histH, histW, CV_8UC3, cv::Scalar(0, 0, 0));
    // cv::normalize(histSmoothed, histSmoothed, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    // for (int i = 1; i < histSize; i++)
    // {
    //     cv::line(histImage, cv::Point(binW * (i - 1), histH - cvRound(histSmoothed.at<float>(i - 1))),
    //              cv::Point(binW * (i), histH - cvRound(histSmoothed.at<float>(i))),
    //              cv::Scalar(255, 0, 0), 2, 8, 0);
    // }

    return histSmoothed;
}

std::pair<size_t, std::pair<size_t, size_t>> ScreenExtractor::getColorThreshhold(cv::Mat &histogram)
{
    // find local maxima
    std::vector<int> peaks;
    for (int i = 1; i < histogram.rows - 1; i++)
    {
        if (histogram.at<float>(i) > histogram.at<float>(i - 1) &&
            histogram.at<float>(i) > histogram.at<float>(i + 1))
        {
            peaks.push_back(i);
        }
    }

    // find two peaks
    int peak1 = 0, peak2 = 0;
    for (int i = 0; i < peaks.size(); i++)
    {
        if (histogram.at<float>(peaks[i]) > histogram.at<float>(peak1))
        {
            peak2 = peak1;
            peak1 = peaks[i];
        }
        else if (histogram.at<float>(peaks[i]) > histogram.at<float>(peak2))
        {
            peak2 = peaks[i];
        }
    }
    if (peak2 == 0 || peak2 > peak1)
    {
        return std::make_pair(0, std::make_pair(0, 0));
    }

    // find local minima in between peaks
    int min1 = peak2;
    for (int i = peak2; i < peak1; i++)
    {
        if (histogram.at<float>(i) < histogram.at<float>(min1))
        {
            min1 = i;
        }
    }

    return std::make_pair(min1, std::make_pair(peak1, peak2));
}

void ScreenExtractor::extractor()
{
    bool expectFalse = false;
    nextRenderTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    while (this->needStop.compare_exchange_weak(expectFalse, false))
    {
        size_t currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if (currentTime < nextRenderTime)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(nextRenderTime - currentTime));
        }
        nextRenderTime = currentTime + 1000 / this->fps;

        cv::Mat image;
        this->cap->read(image);
        if (image.empty())
        {
            spdlog::error("error reading frame");
            break;
        }

        cv::Mat imageGray;
        cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);

        this->screenCapBuffer.push_back(imageGray);
        if (this->screenCapBuffer.size() > ScreenExtractor::screenCapBufferSize)
        {
            this->screenCapBuffer.pop_front();
        }

        cv::Mat screenCapResult = cv::Mat::zeros(imageGray.size(), CV_8UC1);
        for (auto &frame : this->screenCapBuffer)
        {
            cv::addWeighted(screenCapResult, 1.0, frame, 1.0 / this->screenCapBuffer.size(), 0.0, screenCapResult);
        }

        cv::Rect detected = this->screenDetector->detectScreen(screenCapResult);
        if (!detected.empty())
        {
            this->screenRects.push_back(detected);
            if (this->screenRects.size() > ScreenExtractor::screenRectsSize)
            {
                this->screenRects.pop_front();
            }

            auto validScreenRect = this->getValidScreenRect();
            bool screenChanged = this->validScreenRect.x != validScreenRect.x || this->validScreenRect.y != validScreenRect.y ||
                                 this->validScreenRect.width != validScreenRect.width || this->validScreenRect.height != validScreenRect.height;
            if (screenChanged)
            {
                this->validScreenRect = validScreenRect;
                this->contoursRenderHistory.clear();
                spdlog::info("screen changed: x={}, y={}, width={}, height={}", this->validScreenRect.x, this->validScreenRect.y, this->validScreenRect.width, this->validScreenRect.height);
            }
        }

        if (this->validScreenRect.area() == 0)
        {
            spdlog::info("no valid screen detected");
            continue;
        }

        cv::Mat roi = screenCapResult(this->validScreenRect);
        cv::Mat histogram = this->getHistogram(roi);
        auto min1 = this->getColorThreshhold(histogram);

        // // draw peaks
        // cv::line(histImage, cv::Point(binW * min1.second.first, 0), cv::Point(binW * min1.second.first, histH), cv::Scalar(0, 0, 255), 2, 8, 0);
        // cv::line(histImage, cv::Point(binW * peak2, 0), cv::Point(binW * peak2, histH), cv::Scalar(0, 0, 255), 2, 8, 0);
        // cv::line(histImage, cv::Point(binW * min1, 0), cv::Point(binW * min1, histH), cv::Scalar(0, 255, 0), 2, 8, 0);

        // show histogram
        // cv::imshow("histogram", histImage);

        // threshold on average of two peaks
        cv::Mat thresholded;
        cv::threshold(roi, thresholded, min1.first, 255, cv::THRESH_BINARY);

        // remove noise
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(thresholded, thresholded, cv::MORPH_OPEN, kernel);

        // show thresholded
        // cv::imshow("thresholded", thresholded);

        // // find contours
        // std::vector<std::vector<cv::Point>> contours;
        // std::vector<cv::Vec4i> hierarchy;
        // cv::findContours(thresholded, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // // remove small contours
        // std::vector<std::vector<cv::Point>> contoursFiltered;
        // for (int i = 0; i < contours.size(); i++)
        // {
        //     if (cv::contourArea(contours[i]) > 33)
        //     {
        //         contoursFiltered.push_back(contours[i]);
        //     }
        // }

        // // draw contours
        // cv::Mat drawing = cv::Mat::zeros(thresholded.size(), CV_8UC3);
        // for (int i = 0; i < contoursFiltered.size(); i++)
        // {
        //     cv::drawContours(drawing, contoursFiltered, i, cv::Scalar(0, 255, 0), 2, 8, hierarchy, 0);
        // }

        // // put to history
        // contoursRenderHistory.push_back(drawing);
        // if (contoursRenderHistory.size() > ScreenExtractor::contoursRenderHistorySize)
        // {
        //     contoursRenderHistory.pop_front();
        // }
        // if (contoursRenderHistory.empty())
        // {
        //     continue;
        // }

        // // merge all history together
        // cv::Mat result = cv::Mat::zeros(thresholded.size(), CV_8UC3);
        // for (auto &frame : contoursRenderHistory)
        // {
        //     cv::addWeighted(result, 1.0, frame, 1.0 / contoursRenderHistory.size(), 0.0, result);
        // }

        // // draw a rectangle around the screen
        // cv::rectangle(result, cv::Point(0, 0), cv::Point(result.cols - 1, result.rows - 1), cv::Scalar(0, 255, 0), 3);

        // // keep only bright pixels
        // cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
        // cv::threshold(result, result, 64, 255, cv::THRESH_BINARY);

        // cv::imshow("game screen", result);

        auto nextRendered = this->lastRendered.second;
        thresholded.copyTo(*nextRendered);

        std::lock_guard<std::mutex> lock(this->lastRenderedMutex);
        std::swap(this->lastRendered.first, this->lastRendered.second);
    };
}
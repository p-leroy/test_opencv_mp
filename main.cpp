#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <opencv2\core\mat.hpp>
#include <opencv2/ml.hpp>

#define ROWS 36581
#define COLS 34

float values[ROWS][COLS];
float labels[ROWS][1];

void readCSV()
{
    std::ifstream features("C:/DATA/training/Loire/juin2019/classif_C3_withSS/dalles/features.csv");
    std::ifstream labels("C:/DATA/training/Loire/juin2019/classif_C3_withSS/dalles/labels.csv");
    std::string current_line;
    // vector allows you to add data without knowing the exact size beforehand
    int rows = COLS;
    int cols = ROWS;
    cv::Mat matFeatures(rows, cols, CV_32F);
    cv::Mat matLabels(rows, 1, CV_32F);
    // Start reading lines as long as there are lines in the file
    for (int row = 0; row < rows; row++)
    {
        getline(features, current_line);
        std::stringstream temp(current_line);
        std::string value;
        float* Mi = matFeatures.ptr<float>(row);
        for(int j = 0; j < matFeatures.cols; j++)
        {
            getline(temp, value, ',');
            Mi[j] = atof(value.c_str());
            if (row==0)
                std::cout << value << std::endl;
        }
    }
}

cv::Mat readFeatures(std::string filename)
{
    std::ifstream stream(filename, std::ios::binary);
    stream.read((char *) &values[0], sizeof(float) * ROWS * COLS);
    cv::Mat M(ROWS, COLS, CV_32F, values);

    return M;
}

cv::Mat readLabels(std::string filename)
{
    std::ifstream stream(filename, std::ios::binary);
    stream.read((char *) &labels[0], sizeof(float) * ROWS * 1);
    cv::Mat M(ROWS, 1, CV_32F, labels);

    return M;
}

int main()
{
    std::cout << "[main]" << std::endl;
    std::cout << "sizeof(float) " <<  sizeof(float) << std::endl;

    cv::Mat data = readFeatures("C:/DATA/training/Loire/juin2019/classif_C3_withSS/dalles/features.bin");
    cv::Mat labels = readLabels("C:/DATA/training/Loire/juin2019/classif_C3_withSS/dalles/labels.bin");

    std::cout << "features (10, 10) " << data.at<float>(10, 10) << std::endl;
    std::cout << "features (ROWS - 1, COLS - 1) " << data.at<float>(ROWS - 1, COLS - 1) << std::endl;

    std::cout << "labels (10, 0) " << labels.at<float>(10, 0) << std::endl;
    std::cout << "labels (ROWS - 1, 0) " << labels.at<float>(ROWS - 1, 0) << std::endl;

    // RTrees for classification
    cv::Ptr<cv::ml::RTrees> m_rtrees = cv::ml::RTrees::create();

    m_rtrees->setMaxDepth(25);
    m_rtrees->setMinSampleCount(2);
    m_rtrees->setRegressionAccuracy(0);
    // If true then surrogate splits will be built. These splits allow to work with missing data and compute variable importance correctly. Default value is false.
    m_rtrees->setUseSurrogates(false);
    m_rtrees->setPriors(cv::Mat());
    //m_rtrees->setMaxCategories(params.maxCategories); //not important?
    m_rtrees->setCalculateVarImportance(true);
    m_rtrees->setActiveVarCount(0);
    cv::TermCriteria terminationCriteria(cv::TermCriteria::MAX_ITER, 100, std::numeric_limits<double>::epsilon());
    m_rtrees->setTermCriteria(terminationCriteria);

	std::cout << "[main] start training" << std::endl;
//    m_rtrees->train(data, cv::ml::ROW_SAMPLE, labels);
	m_rtrees->train_MP(data, cv::ml::ROW_SAMPLE, labels);
	std::cout << "[main] training complete" << std::endl;

    return 0;
}

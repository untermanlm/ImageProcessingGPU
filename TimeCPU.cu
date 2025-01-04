#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void BGRToGrayScale(Mat& image, Mat& dest){
    dest = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);
    for (int row = 0; row < image.rows; row++){
        for (int col = 0; col < image.cols; col++){
            Vec3b& pixel = image.at<Vec3b>(row, col);
            unsigned char grayValue = static_cast<uchar>(0.114f * pixel[0]) + (0.587f * pixel[1]) + (0.299f * pixel[2]);
            dest.at<uchar>(row, col) = grayValue;
        }
    }
}

void SauvolaBinary(Mat& image, int windowSize, float k, float R){
    for (int row = 0; row < image.rows; row++){
        for (int col = 0; col < image.cols; col++){
            int halfWindow = windowSize / 2;
            float mean = 0.0f, std_dev = 0.0f;
            int count = 0;

            for (int i = -halfWindow; i <= halfWindow; ++i) {
                for (int j = -halfWindow; j <= halfWindow; ++j) {
                    int nx = min(max(col + j, 0), image.cols - 1);
                    int ny = min(max(row + i, 0), image.rows - 1);
                    float pixel = image.at<uchar>(ny, nx);
                    mean += pixel;
                    std_dev += pixel * pixel;
                    ++count;
                }
            }
            mean /= count;
            std_dev = sqrt((std_dev / count) - (mean * mean));
            float thresh = mean * (1 + k * ((std_dev / R) - 1));
            
            if (image.at<uchar>(row, col) > thresh){
                image.at<uchar>(row, col) = 255;
            }
            else{
                image.at<uchar>(row, col) = 0;
            }
        }
    }
}

void BitwiseNot(Mat& image){
    for (int row = 0; row < image.rows; row++){
        for (int col = 0; col < image.cols; col++){
            uchar& pixel = image.at<uchar>(row, col);
            pixel = ~pixel;
        }
    }
}

void PadImage(Mat& image, Mat& paddedImage, int padSize, uchar padValue){
    for (int row = 0; row < paddedImage.rows; row++){
        for (int col = 0; col < paddedImage.cols; col++){
            uchar value = padValue;
            uchar notInPadRow = row - padSize >= 0;
            uchar notInPadCol = col - padSize >= 0;
            if (notInPadRow && notInPadCol && row < image.rows + padSize && col < image.cols + padSize){
                value = image.at<uchar>(row - padSize, col - padSize);
            }
            paddedImage.at<int>(row, col) = value;
        }
    }
}

tuple<int, Mat>  ConnectedComponents(Mat& image, Mat& labelImage){
    int numRows = image.rows;
    int numCols = image.cols;

    // // Pad image
    // Mat labelImage = Mat::zeros(Size(numCols + 2, numRows + 2), CV_32S);
    // image.copyTo(labelImage(Rect(1,1,numCols,numRows)));
    
    map<int, set<int>> equivalenceTable;
    int k = 1;

    // First pass
    for (int row = 1; row <= numRows; row++){
        for (int col = 1; col <= numCols; col++){
            if (labelImage.at<int>(row, col) == 0){
                continue;
            }

            // 4-connectivity
            vector<int> neighbors = {
                labelImage.at<int>(row, col-1),
                labelImage.at<int>(row-1, col)
            };

            vector<int> nonZeroNeighbors;
            for (int n : neighbors){
                if (n != 0){
                    nonZeroNeighbors.push_back(n);
                }
            }

            // Neighbors don't have white pixels
            if (nonZeroNeighbors.empty()){
                labelImage.at<int>(row, col) = k;
                k++;
            }
            else {
                int minLabel = *min_element(nonZeroNeighbors.begin(), nonZeroNeighbors.end());
                labelImage.at<int>(row, col) = minLabel;
                for (int n : nonZeroNeighbors){
                    if (n !=  minLabel){
                        equivalenceTable[minLabel].insert(n);
                        equivalenceTable[n].insert(minLabel);
                    }
                }

            }
        }
    }

    // Resolve pt. 1
    for (auto& [key, values] : equivalenceTable){
        vector<int> toMerge(values.begin(), values.end());
        for (int v : toMerge){
            if (equivalenceTable.count(v)) {
                values.insert(equivalenceTable[v].begin(), equivalenceTable[v].end());
            }
        }
    }

    // Resolve pt. 2
    unordered_map<int, int> flatEquivalenceMap;
    for (const auto& [key, values] : equivalenceTable){
        int root = key;
        for (int v : values){
            flatEquivalenceMap[v] = root;
        }
    }

    // Second pass
    for (int row = 1; row <= numRows; row++){
        for (int col = 1; col <= numCols; col++){
            int label = labelImage.at<int>(row, col);
            if (flatEquivalenceMap.count(label)){
                labelImage.at<int>(row, col) = flatEquivalenceMap[label];
            }
        }
    }
    // Remove padding
    labelImage = labelImage(Rect(1, 1, numCols, numRows));


    // Get all unique labels after downsizing upper bound of k
    set<int> unique; 
    for (int row = 0; row < numRows; row++){
        for (int col = 0; col < numCols; col++){
            if (labelImage.at<int>(row, col) != 0){
                unique.insert(labelImage.at<int>(row, col));
            }
        }
    }
    return make_tuple(unique.size(), labelImage);
}


int main(int argc, char **argv)
{
	if(argc != 4) {
		cerr << "Usage: ./program input.jpg output.jpg downsizeby" << endl;
		return 1;
	}

	string inputImageFileName = string(argv[1]);
	string outputImageFileName = string(argv[2]);
    string down = string(argv[3]);
    int downsize =  stoi(down);

    // Read image
	cv::Mat inputImage = cv::imread(inputImageFileName, IMREAD_COLOR);
    resize(inputImage, inputImage, Size(inputImage.cols / downsize, inputImage.rows / downsize));
    printf("Input image dim: rows=%d, cols=%d\n", inputImage.rows, inputImage.cols);
	Mat frame1C;

    int padding = 1;
    int padNumRows = inputImage.rows + 2 * padding;
    int padNumCols = inputImage.cols + 2 * padding;
    Mat paddedImage(padNumRows, padNumCols, CV_32S);

    BGRToGrayScale(inputImage, frame1C);
    SauvolaBinary(frame1C, 5, 0.2, 128.0);
    BitwiseNot(frame1C);
    PadImage(frame1C, paddedImage, padding, 0);

    auto startTime = chrono::steady_clock::now();
    tuple<int, Mat> componentTuple = ConnectedComponents(frame1C, paddedImage);
    int uniqueCount = get<0>(componentTuple);
    Mat labels = get<1>(componentTuple);

    auto endTime = chrono::steady_clock::now();
    double elapsedSec = chrono::duration<double, milli>(endTime - startTime).count();
    printf("time elapsed (ms): %.4f\n", elapsedSec);  

    cv::imwrite(outputImageFileName, labels);

    return 0;
}
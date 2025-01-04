#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

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


void GrayScaleToBinary(Mat& image, int thresh){
    for (int row = 0; row < image.rows; row++){
        for (int col = 0; col < image.cols; col++){
            uchar& pixel = image.at<uchar>(row, col);
            if (pixel < thresh){
                pixel = 0;
            }
            else{
                pixel = 255;
            }
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

void PadImage(Mat& image, Mat& paddedImage, int padSize){
    copyMakeBorder(image, paddedImage, padSize, padSize, padSize, padSize, BORDER_CONSTANT, Scalar(0));
}

//https://www.google.com/books/edition/Handbook_of_Image_and_Video_Processing/UM_GCfJe88sC?q=region+labeling+algorithm&gbpv=1#f=false
//Ignores pixels in border=1 of image unless padded
//Returns number of labels and label image
tuple<int, Mat>  ConnectedComponents(Mat& image){
    int numRows = image.rows;
    int numCols = image.cols;

    // Pad image
    Mat labelImage = Mat::zeros(Size(numCols + 2, numRows + 2), CV_32S);
    image.copyTo(labelImage(Rect(1,1,numCols,numRows)));
    
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

//Citation: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
int main(int argc, char **argv){
	if(argc != 3) {
		cerr << "Usage: ./program input.mov output_images/" << endl;
		return 1;
	}
    string inputVideoFileName = string(argv[1]);
	string outputImageFileName = string(argv[2]);

    cv::Mat frame;
    cv::VideoCapture cap;
    int apiID = cv::CAP_ANY;
    cap.open(inputVideoFileName, apiID);
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    printf("total frames in video: %d\n", totalFrames);
    int count = 0;
    auto startTime = chrono::steady_clock::now();
    for (;;){
        string frameFileName;
        ostringstream oss;
        oss << "image_" << count << ".jpg";
        frameFileName = outputImageFileName + oss.str();
        count++;
        cap.read(frame);
        if (frame.empty()){
            cerr << "Empty frame grabbed \n";
            break;
        }
        //resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));
        Mat frame1C;
        BGRToGrayScale(frame, frame1C);
        SauvolaBinary(frame1C, 7, 0.5f, 110.0f);
        // // GrayScaleToBinary(frame1C, thresh);
        BitwiseNot(frame1C);
        tuple<int, Mat> componentTuple = ConnectedComponents(frame1C);
        int uniqueCount = get<0>(componentTuple);
        Mat labels = get<1>(componentTuple);
        
        // // Make sure retrieved is not padded still
        if (labels.size != frame.size){
            cerr << "labelMat and ImageFrame must have same size!" << endl;
            exit(1);
        }

        // Plot connected components
        double minVal, maxVal;
        minMaxLoc(labels, &minVal, &maxVal);
        Mat normLabels;
        labels.convertTo(normLabels, CV_8U, 255.0 / (maxVal - minVal), -255.0 * minVal / (maxVal - minVal));
        applyColorMap(normLabels, frame, COLORMAP_AUTUMN);

        // Get image with background removed
        for (int row = 0; row < frame.rows; row++){
            for (int col = 0; col < frame.cols; col++){
                if (labels.at<int>(row, col) == 0){
                    frame.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
                }
            }
        }

        imwrite(frameFileName, frame);
        //imwrite(frameFileName, paddedImage);
        if (waitKey(5) >= 0){
            break;
        }
    }
    auto endTime = chrono::steady_clock::now();
    double elapsedSec = chrono::duration<double, milli>(endTime - startTime).count() / 1000;
    double fps = totalFrames / elapsedSec;

    printf("frames per second: %.2f\n", fps);  
    return 0;


}
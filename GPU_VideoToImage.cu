#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

__global__ void BGRToGrayScale(uchar3 *BGRImage, unsigned char *grayImage, int numberRows, int numberColumns)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int    row = blockIdx.y * blockDim.y + threadIdx.y;

    if(column < numberColumns && row < numberRows)
    {
		uchar3 pixel = BGRImage[row * numberColumns + column];

		unsigned char grayValue = (0.114f * pixel.x) + (0.587f * pixel.y) + (0.299f * pixel.z); // BGR to gray

		grayImage[row * numberColumns + column] = grayValue;
    }
}

__global__ void GrayScaleToBinary(const uchar *grayImage, uchar *binaryImage, int numberRows, int numberColumns, int thresh){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int    row = blockIdx.y * blockDim.y + threadIdx.y;
    if(column < numberColumns && row < numberRows){
        uchar pixel = grayImage[row * numberColumns + column];
        if (pixel < thresh){
            binaryImage[row * numberColumns + column] = 0;
        }
        else{
            binaryImage[row * numberColumns + column] = 255;
        }
    }
}

__global__ void BitwiseNot(uchar *binary, uchar *binaryFlip, int numberRows, int numberColumns){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int    row = blockIdx.y * blockDim.y + threadIdx.y;
    if(column < numberColumns && row < numberRows){
        uchar pixel = binary[row * numberColumns + column];
        binaryFlip[row * numberColumns + column] = ~pixel;
    }
}

__global__ void SauvolaBinary(const uchar *grayImage, uchar *binaryImage, int numberRows, int numberCols, int windowSize, float k) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int    row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column < numberCols && row < numberRows) {
        int halfWindow = windowSize / 2;
        float mean = 0.0f, std_dev = 0.0f;
        int count = 0;

        for (int i = -halfWindow; i <= halfWindow; ++i) {
            for (int j = -halfWindow; j <= halfWindow; ++j) {
                int nx = min(max(column + j, 0), numberCols - 1);
                int ny = min(max(row + i, 0), numberRows - 1);
                float pixel = grayImage[ny * numberCols + nx];
                mean += pixel;
                std_dev += pixel * pixel;
                ++count;
            }
        }

        mean /= count;
        std_dev = sqrt((std_dev / count) - (mean * mean));

        float threshold = mean * (1 + k * ((std_dev / 110.0f) - 1));

        binaryImage[row * numberCols + column] = (grayImage[row * numberCols + column] > threshold) ? 255 : 0;
    }
}

// Naive, branch divergence
__global__ void DepConnectedComponents(const int *labelImage, int *outputImage, int numberRows, int numberColumns, int *k, int *equivalenceTable){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int    row = blockIdx.y * blockDim.y + threadIdx.y;
    int  pixel = labelImage[row * numberColumns + column];
    if(column < numberColumns && row < numberRows && pixel != 0 && row >= 1 && column >= 1){
        int leftNeighbor = labelImage[row * numberColumns + column - 1];
        int upNeighbor = labelImage[(row - 1) * numberColumns + column];
        int minLabel = 0;

        if (leftNeighbor != 0) minLabel = leftNeighbor;
        if (upNeighbor != 0 && (minLabel == 0 || upNeighbor < minLabel))  minLabel = upNeighbor;
        if (minLabel == 0){
            outputImage[row * numberColumns + column] = *k;
            atomicAdd(k, 1);
        }
        else{
            outputImage[row * numberColumns + column] = minLabel;
            if (leftNeighbor != 0 && leftNeighbor != minLabel){
                equivalenceTable[minLabel * numberColumns + leftNeighbor] = 1;
                equivalenceTable[leftNeighbor *  numberColumns + minLabel] = 1;
            }
            if (upNeighbor != 0 && upNeighbor != minLabel){
                equivalenceTable[minLabel * numberColumns + upNeighbor] = 1;
                equivalenceTable[upNeighbor *  numberColumns + minLabel] = 1;
            }
        }
    }
}

__global__ void ConnectedComponents(int *labelImage, int *equivalenceMap, int numberRows, int numberColumns){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int    row = blockIdx.y * blockDim.y + threadIdx.y;
    int    idx = row * numberColumns + column;
    if(row < numberRows && column < numberColumns && column >= 1 && row >= 1){
        int label = labelImage[idx];
        if (equivalenceMap[label] > 0){
            labelImage[idx] = equivalenceMap[label];
        }
    }
}

__global__ void PadImage(uchar *input, int *output, int numRows, int numCols, int padNumRows, int padNumCols, int padSize, uchar padValue){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int    row = blockIdx.y * blockDim.y + threadIdx.y;
    if(column < padNumCols && row < padNumRows){
        uchar value = padValue;
        uchar notInPadRow = row - padSize >= 0;
        uchar notInPadCol = column - padSize >= 0;

        // If within regular image bounds, and not in padding area, set value to original image pixel
        if (notInPadRow && notInPadCol && row < numRows + padSize && column < numCols + padSize){
            value = input[(row - padSize) * numCols + (column - padSize)];
        }

        // Set value to either original image pixel or padValue depending on conditional
        output[row * padNumCols + column] = value;
    }

}

__global__ void RemoveBackground(uchar3 *BGRImage, uchar *binaryImage, int numberRows, int numberColumns){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int    row = blockIdx.y * blockDim.y + threadIdx.y;
    if(column < numberColumns && row < numberRows){
        unsigned char pixel = binaryImage[row * numberColumns + column];
        if (pixel == 0){
            BGRImage[row * numberColumns + column] = make_uchar3(0, 0, 0);
        }
    }

}

tuple<int, unordered_map<int, int>> ConnectedComponentsInit(Mat &image, Mat &labelImage){
    int numRows = image.rows;
    int numCols = image.cols;

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
    int mapMax = -1;
    unordered_map<int, int> flatEquivalenceMap;
    for (const auto& [key, values] : equivalenceTable){
        int root = key;
        for (int v : values){
            flatEquivalenceMap[v] = root;
        }
        if (root > mapMax){
            mapMax = root;
        }
    }
    return make_tuple(mapMax, flatEquivalenceMap);
}

//Citation: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
int main(int argc, char **argv){
	if(argc != 3) {
		cerr << "Usage: ./program input.mov output_images/" << endl;
		return 1;
	}

    // Initialization steps
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
    int count = 0;
    auto startTime = chrono::steady_clock::now();

    // Loop continues until last frame reached
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
        int kLabel = 1;
        int padding = 1;
        int padNumRows = frame.rows + 2 * padding;
        int padNumCols = frame.cols + 2 * padding;

        Mat outputImage(frame.rows, frame.cols, CV_8UC3);
        //Mat outputImage(frame.rows, frame.cols, CV_8UC1);
        Mat paddedImage(padNumRows, padNumCols, CV_32S);

        // Allocate GPU memory
        uchar3 *d_frame;
        unsigned char *d_grayscaleImage;
        unsigned char *d_binaryImage;
        unsigned char *d_flipBinaryImage;
        int *d_padded;
        int *d_equivalence;

        // Allocate required variables
        cudaMalloc(&d_frame, sizeof(uchar3) * frame.rows * frame.cols);
        cudaMalloc(&d_grayscaleImage, sizeof(unsigned char) * frame.rows * frame.cols);
        cudaMalloc(&d_binaryImage, sizeof(unsigned char) * frame.rows * frame.cols);
        cudaMalloc(&d_flipBinaryImage, sizeof(unsigned char) * frame.rows * frame.cols);
        cudaMalloc(&d_padded, sizeof(int) * padNumRows * padNumCols);

        // Move input image to d_frame, set equivalence table to 0
        cudaMemcpy(d_frame, (uchar3*) frame.ptr<uchar3>(0), sizeof(uchar3) * frame.rows * frame.cols, cudaMemcpyHostToDevice);

        int threadsPerBlockPerDim = 16;
        int gridDimx = (frame.cols + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
        int gridDimy = (frame.rows + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
        int padGridDimx = (padNumCols + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
        int padGridDimy = (padNumRows + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;

        dim3 blockDim(threadsPerBlockPerDim, threadsPerBlockPerDim);
        dim3 gridDim(gridDimx, gridDimy);
        dim3 padGridDim(padGridDimx, padGridDimy);

        // Image Processing on the GPU
        BGRToGrayScale<<<gridDim, blockDim>>> (d_frame, d_grayscaleImage, frame.rows, frame.cols);
        int windowSize = 50;
        float k = 0.05f;
        SauvolaBinary<<<gridDim, blockDim>>> (d_grayscaleImage, d_binaryImage, frame.rows, frame.cols, windowSize, k);
        BitwiseNot<<<gridDim, blockDim>>> (d_binaryImage, d_flipBinaryImage, frame.rows, frame.cols);
        RemoveBackground<<<gridDim, blockDim>>> (d_frame, d_flipBinaryImage, frame.rows, frame.cols);
        PadImage<<<padGridDim, blockDim>>> (d_flipBinaryImage, d_padded, frame.rows, frame.cols, padNumRows, padNumCols, padding, 0);
        cudaMemcpy((int*) paddedImage.ptr<int>(0), d_padded, sizeof(int) * padNumRows * padNumCols, cudaMemcpyDeviceToHost);
    
        tuple<int, unordered_map<int, int>> ccTuple = ConnectedComponentsInit(frame, paddedImage);
        int mapMax = get<0>(ccTuple);
        unordered_map<int, int> equivalenceMap = get<1>(ccTuple);
        vector<int> flatEquivalences(mapMax + 1, 0);
        for (const auto& [key, value] : equivalenceMap){
            flatEquivalences[key] = value;
        }
        cudaMalloc(&d_equivalence, sizeof(int) * flatEquivalences.size());
        cudaMemcpy(d_equivalence, flatEquivalences.data(), sizeof(int) * flatEquivalences.size(), cudaMemcpyHostToDevice);
        ConnectedComponents<<<padGridDim, blockDim>>> (d_padded, d_equivalence, padNumRows, padNumCols);
        cudaMemcpy((int*) paddedImage.ptr<int>(0), d_padded, sizeof(int) * padNumRows * padNumCols, cudaMemcpyDeviceToHost);

        imwrite(frameFileName, paddedImage);

        // Free Memory
        cudaFree(d_frame);
        cudaFree(d_grayscaleImage);
        cudaFree(d_binaryImage);
        cudaFree(d_flipBinaryImage);
        cudaFree(d_padded);
        cudaFree(d_equivalence);

        cudaError_t cudaError = cudaGetLastError();

        if(cudaError != cudaSuccess)
        {
            fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
            exit(EXIT_FAILURE);
        }

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
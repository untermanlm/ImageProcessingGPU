#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

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

__global__ void SauvolaBinary(const unsigned char *gray, unsigned char *binary, int width, int height, int window_size, float k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int half_window = window_size / 2;
        float mean = 0.0f, std_dev = 0.0f;
        int count = 0;

        for (int wy = -half_window; wy <= half_window; ++wy) {
            for (int wx = -half_window; wx <= half_window; ++wx) {
                int nx = min(max(x + wx, 0), width - 1);
                int ny = min(max(y + wy, 0), height - 1);
                float pixel = gray[ny * width + nx];
                mean += pixel;
                std_dev += pixel * pixel;
                ++count;
            }
        }

        mean /= count;
        std_dev = sqrt((std_dev / count) - (mean * mean));

        float threshold = mean * (1 + k * ((std_dev / 110.0f) - 1));

        binary[y * width + x] = (gray[y * width + x] > threshold) ? 255 : 0;
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

    // Padding
    int kLabel = 1;
    int padding = 1;
    int padNumRows = inputImage.rows + 2 * padding;
    int padNumCols = inputImage.cols + 2 * padding;

	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);
    Mat paddedImage(padNumRows, padNumCols, CV_32S);

    // Allocate GPU memory
    uchar3 *d_BGRImage;
    unsigned char *d_grayscaleImage;
    unsigned char *d_binaryImage;
    unsigned char *d_flipBinaryImage;
    int *d_padded;
    int *d_equivalence;

    cudaMalloc(&d_BGRImage, sizeof(uchar3) * inputImage.rows * inputImage.cols);
    cudaMalloc(&d_grayscaleImage, sizeof(unsigned char) * inputImage.rows * inputImage.cols);
    cudaMalloc(&d_binaryImage, sizeof(unsigned char) * inputImage.rows * inputImage.cols);
    cudaMalloc(&d_flipBinaryImage, sizeof(unsigned char) * inputImage.rows * inputImage.cols);
    cudaMalloc(&d_padded, sizeof(int) * padNumRows * padNumCols);

	cudaMemcpy(d_BGRImage, (uchar3*) inputImage.ptr<uchar3>(0), sizeof(uchar3) * inputImage.rows * inputImage.cols, cudaMemcpyHostToDevice);

	int threadsPerBlockPerDim = 1;
	int gridDimx = (inputImage.cols + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
	int gridDimy = (inputImage.rows + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
    int padGridDimx = (padNumCols + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
    int padGridDimy = (padNumRows + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
    dim3 padGridDim(padGridDimx, padGridDimy);

	dim3 blockDim(threadsPerBlockPerDim, threadsPerBlockPerDim);
	dim3 gridDim(gridDimx,gridDimy);

    int windowSize = 5;
    float k = 0.1f;

    auto g_s1 = chrono::steady_clock::now();
	BGRToGrayScale <<< gridDim, blockDim >>> (d_BGRImage, d_grayscaleImage, inputImage.rows, inputImage.cols);
    cudaDeviceSynchronize();
    auto g_e1 = chrono::steady_clock::now();

    auto g_s2 = chrono::steady_clock::now();
    SauvolaBinary <<<gridDim, blockDim >>> (d_grayscaleImage, d_binaryImage, inputImage.cols, inputImage.rows, windowSize, k);
    cudaDeviceSynchronize();
    auto g_e2 = chrono::steady_clock::now();

    auto g_s3 = chrono::steady_clock::now();
    BitwiseNot<<<gridDim, blockDim>>> (d_binaryImage, d_flipBinaryImage, inputImage.rows, inputImage.cols);
    cudaDeviceSynchronize();
    auto g_e3 = chrono::steady_clock::now();

    auto g_s4 = chrono::steady_clock::now();
    PadImage<<<padGridDim, blockDim>>> (d_flipBinaryImage, d_padded, inputImage.rows, inputImage.cols, padNumRows, padNumCols, padding, 0);
    cudaDeviceSynchronize();
    auto g_e4 = chrono::steady_clock::now();

    cudaMemcpy((int*) paddedImage.ptr<int>(0), d_padded, sizeof(int) * padNumRows * padNumCols, cudaMemcpyDeviceToHost);

    auto g_s5 = chrono::steady_clock::now();
    tuple<int, unordered_map<int, int>> ccTuple = ConnectedComponentsInit(inputImage, paddedImage);
    int mapMax = get<0>(ccTuple);
    unordered_map<int, int> equivalenceMap = get<1>(ccTuple);
    vector<int> flatEquivalences(mapMax + 1, 0);
    for (const auto& [key, value] : equivalenceMap){
        flatEquivalences[key] = value;
    }
    cudaMalloc(&d_equivalence, sizeof(int) * flatEquivalences.size());
    cudaMemcpy(d_equivalence, flatEquivalences.data(), sizeof(int) * flatEquivalences.size(), cudaMemcpyHostToDevice);
    ConnectedComponents<<<padGridDim, blockDim>>> (d_padded, d_equivalence, padNumRows, padNumCols);
    cudaDeviceSynchronize();
    auto g_e5 = chrono::steady_clock::now();

    cudaMemcpy((int*) paddedImage.ptr<int>(0), d_padded, sizeof(int) * padNumRows * padNumCols, cudaMemcpyDeviceToHost);
    //cudaMemcpy((unsigned char*) outputImage.ptr<unsigned char>(0), d_flipBinaryImage, sizeof(unsigned char) * inputImage.rows * inputImage.cols, cudaMemcpyDeviceToHost);
    
    //cudaMemcpy((int*) paddedImage.ptr<int>(0), d_padded, sizeof(int) * padNumRows * padNumCols, cudaMemcpyDeviceToHost);

    double elapsedGray = chrono::duration<double, milli>(g_e1 - g_s1).count();
    double elapsedBW = chrono::duration<double, milli>(g_e2 - g_s2).count();
    double elapsedFlip = chrono::duration<double, milli>(g_e3 - g_s3).count();
    double elapsedPad = chrono::duration<double, milli>(g_e4 - g_s4).count();
    double elapsedCCL = chrono::duration<double, milli>(g_e5 - g_s5).count();

    printf("img2gray time elapsed (ms): %.4f\n", elapsedGray);  
    printf("grayToBW time elapsed (ms): %.4f\n", elapsedBW);  
    printf("BWtoWB time elapsed (ms): %.4f\n", elapsedFlip);  
    printf("WBtoPad time elapsed (ms): %.4f\n", elapsedPad);  
    printf("CCL time elapsed (ms): %.4f\n", elapsedCCL);  

    cv::imwrite(outputImageFileName, paddedImage);

    // Free Memory
    cudaFree(d_BGRImage);
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

    return 0;
}

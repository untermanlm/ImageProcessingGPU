all: VideoToImage GPU_VideoToImage TimeCPU TimeGPU
VideoToImage: VideoToImage.cu
	nvcc VideoToImage.cu -o VideoToImage -I/usr/include/opencv4/ -I/opt/opencv-4.8.1/include/opencv4/ -L/opt/opencv-4.8.1/lib64 -l opencv_core -l opencv_imgproc -l opencv_imgcodecs -l opencv_videoio -l opencv_highgui
GPU_VideoToImage: GPU_VideoToImage.cu
	nvcc GPU_VideoToImage.cu -o GPU_VideoToImage -I/usr/include/opencv4/ -I/opt/opencv-4.8.1/include/opencv4/ -L/opt/opencv-4.8.1/lib64 -l opencv_core -l opencv_imgproc -l opencv_imgcodecs -l opencv_videoio -l opencv_highgui
TimeCPU: TimeCPU.cu
	nvcc TimeCPU.cu -o TimeCPU -I/usr/include/opencv4/ -I/opt/opencv-4.8.1/include/opencv4/ -L/opt/opencv-4.8.1/lib64 -l opencv_core -l opencv_imgproc -l opencv_imgcodecs -l opencv_videoio -l opencv_highgui
TimeGPU: TimeGPU.cu
	nvcc TimeGPU.cu -o TimeGPU -I/usr/include/opencv4/ -I/opt/opencv-4.8.1/include/opencv4/ -L/opt/opencv-4.8.1/lib64 -l opencv_core -l opencv_imgproc -l opencv_imgcodecs -l opencv_videoio -l opencv_highgui
clean:
	rm BGRtoGrayScale
#include "Library.h"
#include "Utils.h"
#include "Contour.h"
#include "Noise.h"

using namespace std;

//18 56 86 72 255 255
// 20 100 100 - 30 255 255
//     {Scalar(29, 86, 6), Scalar(64, 255, 255)}, // Green
Scalar colorTable[NUM_COLOR][2] = {
    {Scalar(20, 88, 16), Scalar(80, 255, 255)}, // Yellow
    {Scalar(16, 86, 27), Scalar(72, 255, 255)}, // Green
    {Scalar(138, 154, 47), Scalar(255, 255, 215)}, // Red
};

bool ShapeDetection(Mat &img, Mat &mask) {
    //imwrite("C:/Users/Khanh Le/Desktop/ROV/Code/mask.jpg", mask);
    //double t = (double) getTickCount();  
    Contour contour(mask);
    contour.FindContourArea();
    contour.SortContourByArea();
    bool isDetected = contour.DrawContour(img);
    return isDetected;
    //t = ((double) getTickCount() - t) / getTickFrequency();
    //cout << "time: " << t * 1000 << " ms" << endl;  
}

/*
I want to take a video and create a binary from it, 
I want it so that if the pixel is within a certain range 
it will be included within the binary. 
In other words I want an upper and lower bound like in the inRange() function 
as opposed to a simple cutoff point like in the threshold() function.

I also want to use adaptive thresholding to account for differences 
in lighting in my video. Is there a way to do this? 
I know there is inRange() that does the former and adaptiveThreshold() 
that does the latter, but I don't know if there is a way to do both.
 */
void Preprocessing(Mat &img, Mat &mask) {
    img = Utils().Resize(img, 600, 0);

    /*
    Mat denoiseImg;
    Mat grayImg;
    cvtColor(img, grayImg, CV_BGR2GRAY);
    fastNlMeansDenoising(grayImg, denoiseImg, 3, 7, 21);
    imshow("Denoise Img", denoiseImg);
     */

    // blur the image to remove noise: need to tune the kernel size
    //I used Size(11 11) before
    Mat blurredImg;
    GaussianBlur(img, blurredImg, Size(15, 15), 0, 0);
    // convert to HSV space;
    Mat hsvImg;
    cvtColor(blurredImg, hsvImg, CV_BGR2HSV);
    // construct mask for yellow color
    Mat rangeMask;
    inRange(hsvImg, colorTable[GREEN][LOWER], colorTable[GREEN][UPPER], rangeMask);

    /*
    Mat grayMask;
    cvtColor(blurredImg, grayMask, CV_BGR2GRAY);
    adaptiveThreshold(grayMask, grayMask, 255,
            ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 4);
    rangeMask = grayMask.mul(rangeMask);
     */
    // erode and dilate to remove noise
    // last time I use Size(3, 3) Point(1, 1)
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    Mat erodeMask, dilateMask;
    //morphologyEx(rangeMask, rangeMask, MORPH_CLOSE , element);   
    erode(rangeMask, erodeMask, element);
    erode(erodeMask, erodeMask, element);
    dilate(erodeMask, dilateMask, element);
    dilate(dilateMask, dilateMask, element);
    mask = dilateMask;

    imshow("Display before algorithm", dilateMask);
}

bool Run(Mat &img) {
    Mat mask;
    Preprocessing(img, mask);
    bool isDetected = ShapeDetection(img, mask);
    return isDetected;
}

void RunVideo() {
    VideoCapture cap("C:/Users/Khanh Le/Desktop/ROV/video0.mp4");
    int fourcc = CV_FOURCC('X', 'V', 'I', 'D');
    VideoWriter video("C:/Users/Khanh Le/Desktop/ROV/video0_result.avi", fourcc,
            30, Size(640, 360), true);
    Mat img, noiseImg;
    double snr;
    bool playVideo = true;
    bool putNoise = true;
    int noiseDetect = 0, normalDetect = 0;
    cap >> img;
    while (!img.empty()) {
        if (playVideo) {
            cap >> img;
            
            Noise noise = Noise(img);
            //noise.AddGaussianNoise(noiseImg, 0.1);
            noise.AddSaltPepperNoise(noiseImg, 0.1);
            //noise.AddUniformNoise(noiseImg, 0.1);
            //noise.AddSwapPixelNoise(noiseImg, 0.1);
            snr = noise.GetPSNR(img, noiseImg);
        }

        bool isDetectedImg = Run(img);
        bool isDetectedNoiseImg = Run(noiseImg);

        if (isDetectedImg) {
            normalDetect++;
        }
        if (isDetectedNoiseImg) {
            noiseDetect++;
        }
        
        ostringstream strStream1;
        strStream1 << normalDetect;
        putText(img, "Detection: " + strStream1.str(), Point(img.rows - 100, 60),
                FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2);


        ostringstream strs;
        strs << snr;
        putText(noiseImg, "PSNR: " + strs.str(), Point(img.rows - 100, 30),
                FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2);
        ostringstream strStream2;
        strStream2 << noiseDetect;
        putText(noiseImg, "Detect: " + strStream2.str(), Point(img.rows - 100, 60),
                FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2);
        //video.write(img);
        imshow("Normal Image", img);
        imshow("Noise Image", noiseImg);
        char key = waitKey(1);
        if (key == 'p') {
            playVideo = !playVideo;
        } else if (key == 'e') {
            break;
        }
    }

    cout << "%correctness: " << (double(noiseDetect) / normalDetect * 100) << "%" << endl;
}

void RunImage() {
    Mat img = imread("C:/Users/Khanh Le/Desktop/ROV/Code/ball1.png", IMREAD_COLOR);
    double t = (double) getTickCount();
    Run(img);
    t = ((double) getTickCount() - t) / getTickFrequency();
    cout << "time: " << t << endl;
    imwrite("C:/Users/Khanh Le/Desktop/ROV/Code/ball_result1.jpg", img);
}

void RunCamera() {
    time_t start, end;
    // for camera
    //VideoCapture cap("http://169.254.104.8:8080");
    //VideoCapture cap("http://169.254.104.8:8081");
    VideoCapture cap(0);

    Mat image;
    // Mat image1;
    bool is_close_call = false;

    int count = 0;
    int numFrames = 60;
    double fps = 0;
    time(&start);
    while (true) {
        if (count == numFrames) {
            time(&end);
            double seconds = difftime(end, start);
            fps = numFrames / seconds;
            cout << "FPS: " << fps << endl;
            time(&start);
            count = 0;
        }
        cap >> image; // read frame
        Run(image);
        imshow("cam", image);
        if (waitKey(10) >= 0) { // Delay and key
            break;

        }
        count++;
    }
}

void RunCamera1() {
    //VideoCapture cap("http://169.254.104.8:8080");
    //VideoCapture cap("http://169.254.104.8:8081");
    VideoCapture cap(0);
    //int fourcc = CV_FOURCC('X','V','I','D');
    //VideoWriter video("C:/Users/Khanh Le/Desktop/ROV/out.avi", fourcc, 
    //    30, Size(640, 360), true);

    Mat image;
    int count = 0;
    double fps = 0;
    while (true) {
        double t = (double) getTickCount();
        cap.grab(); // read frame
        //video.write(image)
        t = ((double) getTickCount() - t) / getTickFrequency();
        cout << "time: " << t * 1000 << " ms" << endl;
        cap.retrieve(image);
        Run(image);
        imshow("cam", image);
        if (waitKey(10) >= 0) { // Delay and keys
            break;
        }
    }
}

int main(int argc, char **argv) {
    //RunImage();
    //RunCamera1();
    RunVideo();
    return 0;
}

void detectBall(Mat &frame, std::vector<Vec3f> &circles) {
    Mat hsv; //Mat to store transformed HSV space image
    Mat upLim; //Mat to store HSV image with upper limit applied
    Mat downLim; //Mat to store HSV image with lower limit applied
    Mat redImg; //Mat to store HSV image with combined upper and lower limits

    //capture frame
    resize(frame, frame, Size(640, 360), 0, 0, INTER_CUBIC);

    //convert to HSV space
    cvtColor(frame, hsv, CV_BGR2HSV);

    // <TODO: remove hard coded limits>
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), downLim);
    inRange(hsv, Scalar(160, 100, 100), Scalar(179, 255, 255), upLim);

    //combine two ranges into single image
    addWeighted(downLim, 1.0, upLim, 1.0, 0.0, redImg);

    //apply Gaussian blur to improve detection
    GaussianBlur(redImg, redImg, Size(9, 9), 2, 2);

    //apply Hough transform (configured to only really work at 7m)
    //inputArray, outputArray, method, dp, minDistance, param1, param2, minR, maxR
    //redImg is 320x240
    // Need to tune up to detect other circular objects
    //double t = (double) getTickCount();
    HoughCircles(redImg, circles, CV_HOUGH_GRADIENT, 1, redImg.rows / 2, 75, 24, 10, 300);
    //if circle is found, save image and return true
    if (circles.size() > 0) {

        // clone original frame to draw circle on
        //Mat endFrame = frame.clone();

        // draw circle
        for (size_t i = 0; i < circles.size(); i++) {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            circle(frame, center, 3, Scalar(0, 255, 0), -1, 8, 0);
            // circle outline
            circle(frame, center, radius, Scalar(0, 0, 255), 3, 8, 0);
        }

        // save images
        //imwrite("/home/pi/NGCP/RPI_cpslo/Datalogs/OriginalImg.jpg", frame);
        //imwrite("/home/pi/NGCP/RPI_cpslo/Datalogs/HSVImg.jpg", redImg);
        //imwrite("/home/pi/NGCP/RPI_cpslo/Datalogs/FinalImg.jpg", endFrame);
    }
    //t = ((double) getTickCount() - t) / getTickFrequency();
    //cout << "time: " << t * 1000 << " ms" << endl;
}

/*
int main(int argc, char** argv) {
    //VideoCapture cap(1);
    VideoCapture cap("C:/Users/Khanh Le/Desktop/ROV/BallDetection_Raw.mp4");
     int frame_width=   cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height=   cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    
    // should do some research about codec on your computer
    int fourcc = CV_FOURCC('X','V','I','D');
    //VideoWriter video("C:/Users/Khanh Le/Desktop/BallDetectionResult.avi", fourcc, 
    //        30, Size(640, 360), true);
    
    std::vector<Vec3f> circles;
    Mat image = imread("C:/Users/Khanh Le/Desktop/ROV/Code/ball1.png", IMREAD_COLOR);
    double t = (double) getTickCount();

    detectBall(image, circles);
    t = ((double) getTickCount() - t) / getTickFrequency();
    cout << "time: " << t << endl;
    //imwrite("C:/Users/Khanh Le/Desktop/ROV/Code/ball_result1.jpg", image);
  
    
    return 0;
} 
 */


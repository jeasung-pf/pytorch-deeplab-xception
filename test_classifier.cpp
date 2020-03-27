#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <Windows.h>
#include <string>
#include <vector>
#include <io.h>
#include <conio.h>


#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>


using namespace cv::ml;
using namespace cv::xfeatures2d;
using namespace cv;
using namespace std;

// Datasets: https://drive.google.com/open?id=1DB42nE9y8TcY15K1UzDO4Ly62euN5en0
// CIFAR-10
const string TRAIN_DATASET_PATH = "cifar\\trainSet\\";
const string TEST_DATASET_PATH = "cifar\\testSet\\";

// FIDS30
//const string TRAIN_DATASET_PATH = "FIDS30\\";
//const string TEST_DATASET_PATH = "FIDS30\\";


vector<string> get_files_inDirectory(string _path) // 디렉토리 안에 있는 파일 리스트
{
    vector<string> file_list;
    _finddatai64_t c_file;
    intptr_t hFile;
    char path[100] = "";
    strcpy_s(path, (_path + "*.*").c_str());


    if ((hFile = _findfirsti64(path, &c_file)) == -1L) {
        switch (errno) {
        case ENOENT:
            printf(":: Not exist file ::\n"); break;
        case EINVAL:
            fprintf(stderr, "Invalid path name.\n"); exit(1); break;
        case ENOMEM:
            fprintf(stderr, "Not enough memory or file name too long.\n"); exit(1); break;
        default:
            fprintf(stderr, "Unknown error.\n"); exit(1); break;
        }
    } // end if
    else {
        do {
            file_list.push_back(c_file.name);
        } while (_findnexti64(hFile, &c_file) == 0);
        _findclose(hFile); // _findfirsti64(), _findnexti64()에 사용된 메모리를 반환
    } // end else

    return file_list;
}

void readDetectComputeimage(const string& path, const string& className, int classLable, vector<Mat>& allDescPerImg, vector<int> & allClassPerImg, int file_cnt) { // 이미지 불러오기
    vector<string> file_list = get_files_inDirectory(path + className + "\\*.*");
    //cout << "Start className: " << className << ", file_cnt: " << file_list.size() - 2 << endl;

    if (file_cnt == -1) {
        file_cnt = file_list.size();
    }
    else {
        file_cnt += 2;
    }

    for (int i = 2; i < file_cnt; i++) {
            Mat img = imread(path + className + "\\" + file_list[i], COLOR_BGR2RGB);
            if (img.rows != 0) {
                resize(img, img, Size(32, 32)); // Resize 필요하면 하기

                allDescPerImg.push_back(img);
                allClassPerImg.push_back(classLable);

            }
    }
    //cout << "End className: " << className << ", file_cnt: " << file_list.size() - 2 << endl;
}

HOGDescriptor hog;
void HogInit() { // hog 초기화 설정
    hog.winSize = Size(32, 32);
    hog.blockSize = Size(8, 8);
    hog.blockStride = Size(8, 8);
    hog.cellSize = Size(8, 8);
    hog.nbins = 16;
    hog.derivAperture = 1;
    hog.L2HysThreshold = 0.2;
    hog.gammaCorrection = 0;
    hog.signedGradient = 1;
}

float *ColorName(Mat& mat) { // ColorName 함수
    Mat data, image, kLabels, kCenters;
    mat.convertTo(data, CV_32F);
    data = data.reshape(1, mat.total());

    // 원본 이미지에서 clusterCount개의 색으로 추상화.
    const int clusterCount = 60;
    int attempts = 3, iterationNumber = 1e4;
    kmeans(data, clusterCount, kLabels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, iterationNumber, 1e-4), attempts, KMEANS_PP_CENTERS, kCenters);

    // 벡터 정리
    kCenters = kCenters.reshape(3, kCenters.rows);
    data = data.reshape(3, data.rows);

    // ColorName 논문에서 제안한 11가지 색 사용
    string color_name[11] = { "Black", "Blue", "Brown", "Grey", "Green", "Orange", "Pink", "Purple", "Red", "White", "Yellow" };
    int color[11][3] = { {0, 0, 0}, {0, 0, 255}, {165, 42, 42}, {128, 128, 128}, {0, 255, 0},
    {155, 165, 0}, {255, 192, 203}, {128, 0, 128}, {255, 0, 0}, {255, 255, 255}, {255, 255, 0} }; // rgb
    float *color_his = new float[11];

    // 이미지를 추상화
    Vec3f* p = data.ptr<Vec3f>();
    int arr[clusterCount] = { 0, };
    for (size_t i = 0; i < data.rows; i++) {
        int center_id = kLabels.at<int>(i);
        arr[center_id]++;
        p[i] = kCenters.at<Vec3f>(center_id);
    }

    mat = data.reshape(3, mat.rows);
    mat.convertTo(mat, CV_8U);

    // clusterCount개로 추상화된 색 종류를 11가지 색중 가장 가까운 거리의 색으로 매핑, clusterCount -> 11 색으로 추상화
    for (int i = 0; i < clusterCount; i++) {
        // clusterCount개의 색 각각 11가지 색으로 매핑
        float distance = sqrt(pow(kCenters.at<Vec3f>(i)[0] - color[0][0], 0) + pow(kCenters.at<Vec3f>(i)[1] - color[0][1], 2) + pow(kCenters.at<Vec3f>(i)[2] - color[0][2], 2));
        int cnt = 0;
        for (int j = 1; j < 11; j++) {
            float tmp = sqrt(pow(kCenters.at<Vec3f>(i)[0] - color[j][0], 0) + pow(kCenters.at<Vec3f>(i)[1] - color[j][1], 2) + pow(kCenters.at<Vec3f>(i)[2] - color[j][2], 2));
            if (distance > tmp) {
                distance = tmp;
                cnt = j;
            }
        }
        color_his[cnt] += arr[i];
        //cout << kCenters.at<Vec3f>(i)[0] << arr[i] << endl;
    }

    // 사진의 색이 11가지 색으로 distribution됨, color_his를 모두 더하면 사진 pixel 수와 같음
    // 11가지 색의 distribution을 Normalization함 (Normalization을 하고 안하고의 차이는 많이 난다)
    for (int i = 0; i < 11; i++) {
        color_his[i] = color_his[i] / (mat.size[0] * mat.size[1]);
        //cout << color_name[i] << ": " << color_his[i] << endl;
    }


    return color_his;
}

void CreateDataHog(vector<Mat>& deskewedCells, vector<int>& allClassPerImg, vector<vector<float>>& dataHog, Mat& label) {
    for (Mat& mat : deskewedCells) {
        vector<float> descriptors;
        hog.compute(mat, descriptors);

        // ColorName 적용
        float *color_his = ColorName(mat);
        // hog의 결과 뒤 float 원소를 더한다
        descriptors.insert(descriptors.end(), color_his, color_his + 11);
        
        dataHog.push_back(descriptors);
    }

    // label를 Mat 형식으로 바꾼다
    for (int& i : allClassPerImg) {
        label.push_back(Mat(1, 1, CV_32SC1, i));
    }
}


void ConvertVectortoMatrix(vector<vector<float>>& dataHog, Mat& dataMat) { // Vector를 Matrix 형태로 바꾼다
    int descriptor_size = dataHog[0].size();

    for (int i = 0; i < dataHog.size(); i++) {
        for (int j = 0; j < descriptor_size; j++) {
            dataMat.at<float>(i, j) = dataHog[i][j];
        }
    }
}

Ptr<SVM> svmInit() { // SVM 초기화 함수
    Ptr<SVM> svm = SVM::create();
    svm->setGamma(0.6);
    svm->setC(12.5);
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));
    return svm;
}

void svmTrain(Ptr<SVM> svm, Mat& trainMat, Mat& trainLabel) { // SVM 학습 합수
    Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabel);
    svm->train(td);
    svm->save("svm_model.xml");
}

void svmPredict(Ptr<SVM> svm, Mat& testResponse, Mat& testMat) // SVM 예측 함수
{
    svm->predict(testMat, testResponse);
}

void SVMevaluate(Mat& testResponse, float& count, float& accuracy, vector<int>& testLabels) // SVM 평가 함수 
{
    for (int i = 0; i < testResponse.rows; i++)
    {
        // cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
        if (testResponse.at<float>(i, 0) == testLabels[i])
            count = count + 1;
    }
    accuracy = (count / testResponse.rows) * 100;
}

void PrintParam(Ptr<SVM> svm) { // hog와 SVM Parameters를 출력
    cout << "hog.winSize          : " << hog.winSize << endl;
    cout << "hog.blockSize        : " << hog.blockSize << endl;
    cout << "hog.blockStride      : " << hog.blockStride << endl;
    cout << "hog.cellSize         : " << hog.cellSize << endl;
    cout << "hog.nbins            : " << hog.nbins << endl;
    cout << "hog.derivAperture    : " << hog.derivAperture << endl;
    cout << "hog.L2HysThreshold   : " << hog.L2HysThreshold << endl;
    cout << "hog.gammaCorrection  : " << hog.gammaCorrection << endl;
    cout << "hog.signedGradient   : " << hog.signedGradient << endl;;
    cout << "hog.getKernelType    : " << svm->getKernelType() << endl;
    cout << "svm->Type            : " << svm->getType() << endl;
    cout << "svm->C               : " << svm->getC() << endl;
    cout << "svm->Degree          : " << svm->getDegree() << endl;
    cout << "svm->Nu              : " << svm->getNu() << endl;
    cout << "svm->Gamma           : " << svm->getGamma() << endl;
}

int main() {
    vector<Mat> allDescPerImg;
    vector<int> allClassPerImg;
    const int labels_cnt = 10;

    // CIFAR-10 Labels
    string labels[labels_cnt] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    // FIDS30 Labels
    /*string labels[labels_cnt] = { "acerolas", "apples", "apricots", "avocados", "bananas", 
                                  "blackberries", "blueberries", "cantaloupes", "cherries", "coconuts", 
                                  "figs", "grapefruits", "grapes", "guava", "kiwifruit", 
                                  "lemons", "limes", "mangos", "olives", "oranges", 
                                  "passionfruit","peaches", "pears", "pineapples", "plums", 
                                  "pomegranates", "raspberries", "strawberries", "tomatoes", "watermelons" };*/
    
    clock_t sTime = clock();
    for (int i = 0; i < labels_cnt; i++) {
        readDetectComputeimage(TRAIN_DATASET_PATH, labels[i], i, allDescPerImg, allClassPerImg, 100);
    } 
    cout << "Detect and Describe input in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << "second(s)" << endl;
    

    sTime = clock();
    cout << "HOG..." << endl;
    // Hog 초기화
    HogInit();

    vector<Mat> deskewedCells;
    vector<vector<float>> dataHog;
    Mat HogLabel;
    CreateDataHog(allDescPerImg, allClassPerImg, dataHog, HogLabel);

    Mat trainMat(dataHog.size(), dataHog[0].size(), CV_32FC1);
    ConvertVectortoMatrix(dataHog, trainMat);
    cout << "HOG in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << "second(s)" << endl;
    

    sTime = clock();
    cout << "SVM training... " << endl;
    Ptr<SVM> model = svmInit();
    svmTrain(model, trainMat, HogLabel);
    cout << "SVM trained in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << "second(s)" << endl;
    

    sTime = clock();
    cout << "Testing images..." << endl;
    allDescPerImg.clear();
    allClassPerImg.clear();
    for (int i = 0; i < labels_cnt; i++) {
        readDetectComputeimage(TEST_DATASET_PATH, labels[i], i, allDescPerImg, allClassPerImg, 20);
    }
    deskewedCells.clear();

    dataHog.clear();
    CreateDataHog(allDescPerImg, allClassPerImg, dataHog, HogLabel);

    Mat testMat(dataHog.size(), dataHog[0].size(), CV_32FC1);
    ConvertVectortoMatrix(dataHog, testMat);

    Mat testResponse;
    svmPredict(model, testResponse, testMat);

    float count = 0;
    float accuracy = 0;
    SVMevaluate(testResponse, count, accuracy, allClassPerImg);
    
    cout << "Test completed in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << "Second(s)" << endl;
    cout << "Accuracy: " << accuracy << "%" << endl;


    PrintParam(model);
    return(0);
}

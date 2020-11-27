#ifdef __cplusplus
#endif
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <io.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/core/utility.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <opencv2/video/background_segm.hpp>


#include "HCNetSDK.h"
#include "plaympeg4.h"
#include <cstdio>
#include <cstring>
#include "PlayM4.h"
#include <thread>
#include <future>
#include <ctime>
#include <time.h>
#include <windows.h>
#include <thread>
#include <mutex>
#include <sstream>
#include <stdio.h>
#include "opencv2/core/hal/interface.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/features2d.hpp"
#include "vlc/plugins/vlc_reader.h"
#include "opencv2/objdetect/objdetect.hpp"

#include "omp.h"

std::mutex mtx;
#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl
#define USECOLOR 1
#define MAX_QUEUE 50；

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace cv::xfeatures2d;
#pragma warning(disable:4996)

bool stop(false);
//--------------------------------------------
int iPicNum = 0;//Set channel NO.
HWND hWnd = NULL;

// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = false;
bool try_gpu = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = 0.15;
float conf_thresh = 0.8;
string features_type = "orb";
string matcher_type = "affine";
string estimator_type = "affine";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = true;
std::string save_graph_to;
string warp_type = "spherical";
//int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
string seam_find_type = "gc_color";
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
bool timelapse = false;
int range_width = 9;

#define device_num 4
#define level_num 0

static long nPort[device_num];


void CALLBACK g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void* pUser)
{
	char tempbuf[256] = { 0 };
	switch (dwType)
	{
	case EXCEPTION_RECONNECT:    //预览时重连
		printf("----------reconnect--------%d\n", time(NULL));
		break;
	default:
		break;
	}
}

vector<Mat> before_;
int seam_detect(cv::Mat before_, cv::Mat frame_)
{
	int flag = 0;
	IplImage* tempimg_src;
	tempimg_src = &cvIplImage(before_);

	CvSize size1 = cvSize((*tempimg_src).width, (*tempimg_src).height);
	IplImage* tempimg_dst = cvCreateImage(size1, 8, 3);
	cvSmooth(tempimg_src, tempimg_dst, CV_BLUR, 3, 3, 0, 0);//平滑函数

	//下一帧平滑
	IplImage* frameimg_src;
	frameimg_src = &cvIplImage(frame_);
	CvSize size2 = cvSize((*frameimg_src).width, (*frameimg_src).height);
	IplImage* frameimg_dst = cvCreateImage(size2, 8, 3);
	cvSmooth(frameimg_src, frameimg_dst, CV_BLUR, 3, 3, 0, 0);//平滑函数


	cv::Mat temp1, frame1;
	temp1 = cv::cvarrToMat(tempimg_dst);
	frame1 = cv::cvarrToMat(frameimg_dst); //转回为Mat
	//将background和frame转为灰度图  
	cv::Mat gray1, gray2;
	cvtColor(temp1, gray1, CV_BGR2GRAY);
	cvtColor(frame1, gray2, CV_BGR2GRAY);
	//将background和frame做差  
	cv::Mat diff;
	absdiff(gray1, gray2, diff);
	imshow("帧差图", diff);
	waitKey(1);
	
	
	int sum = 0;
	sum = countNonZero(diff);
	cout << sum << endl;
	if (sum > 40000)
		flag = 1;
	temp1.release();
	frame1.release();
	gray1.release();
	gray2.release();
	diff.release();
	cvReleaseData(tempimg_dst);
	cvReleaseData(frameimg_dst);
	return flag;//返回result  
}

//马赛克
void drawMosaicRef(const cv::Mat& mat, const cv::Rect& rect, int cellSz)
{
	cv::Rect mat_rect(0, 0, mat.cols, mat.rows);
	auto intersection = mat_rect & rect;

	cv::Mat msc_roi = mat(intersection);

	bool has_crop_x = false;
	bool has_crop_y = false;

	int cols = msc_roi.cols;
	int rows = msc_roi.rows;

	if (msc_roi.cols % cellSz != 0)
	{
		has_crop_x = true;
		cols -= msc_roi.cols % cellSz;
	}

	if (msc_roi.rows % cellSz != 0)
	{
		has_crop_y = true;
		rows -= msc_roi.rows % cellSz;
	}

	cv::Mat cell_roi;
	for (int i = 0; i < rows; i += cellSz)
	{
		for (int j = 0; j < cols; j += cellSz)
		{
			cell_roi = msc_roi(cv::Rect(j, i, cellSz, cellSz));
			cell_roi = cv::mean(cell_roi);
		}
		if (has_crop_x)
		{
			cell_roi = msc_roi(cv::Rect(cols, i, msc_roi.cols - cols, cellSz));
			cell_roi = cv::mean(cell_roi);
		}
	}

	if (has_crop_y)
	{
		for (int j = 0; j < cols; j += cellSz)
		{
			cell_roi = msc_roi(cv::Rect(j, rows, cellSz, msc_roi.rows - rows));
			cell_roi = cv::mean(cell_roi);
		}
		if (has_crop_x)
		{
			cell_roi = msc_roi(cv::Rect(cols, rows, msc_roi.cols - cols, msc_roi.rows - rows));
			cell_roi = cv::mean(cell_roi);
		}
	}
}

//人脸识别
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip)
{
	double t = 0;
	vector<Rect> faces, faces2;
	const static Scalar colors[] =
	{
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)
	};
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
	equalizeHist(smallImg, smallImg);

	t = (double)getTickCount();
	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| CASCADE_SCALE_IMAGE,
		Size(30, 30));
	if (tryflip)
	{
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			| CASCADE_SCALE_IMAGE,
			Size(30, 30));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)getTickCount() - t;
	printf("detection time = %g ms\n", t * 1000 / getTickFrequency());
	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
		int radius;

		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width * 0.5) * scale);
			center.y = cvRound((r.y + r.height * 0.5) * scale);
			radius = cvRound((r.width + r.height) * 0.25 * scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(img, Point(cvRound(r.x * scale), cvRound(r.y * scale)),
				Point(cvRound((r.x + r.width - 1) * scale), cvRound((r.y + r.height - 1) * scale)),
				color, 3, 8, 0);
		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(r);
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			//|CASCADE_DO_CANNY_PRUNING
			| CASCADE_SCALE_IMAGE,
			Size(30, 30));
		for (size_t j = 0; j < nestedObjects.size(); j++)
		{
			Rect nr = nestedObjects[j];
			center.x = cvRound((r.x + nr.x + nr.width * 0.5) * scale);
			center.y = cvRound((r.y + nr.y + nr.height * 0.5) * scale);
			radius = cvRound((nr.width + nr.height) * 0.25 * scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
	}
	imshow("result", img);
}

cv::CascadeClassifier cascade, nestedCascade;
string cascadeName = "D:\\opencv-build\\test\\Project1\\face\\haarcascade_fullbody.xml";
string nestedCascadeName = "D:\\opencv-build\\test\\Project1\\face\\haarcascade_frontalface_alt.xml";
double scale = 1;
bool tryflip = false;
cv::Mat before;

cv::Mat MoveDetect(cv::Mat before, cv::Mat frame) {
	cascade.load(samples::findFile(cascadeName));
	nestedCascade.load(samples::findFileOrKeep(nestedCascadeName));

	cv::Mat result = frame.clone();
	detectAndDraw(result, cascade, nestedCascade, scale, tryflip);
	//帧差法
	cout << before.size() << endl;
	cout << frame.size() << endl;
	//上一帧平滑
	IplImage* tempimg_src;
	tempimg_src = &cvIplImage(before);

	CvSize size1 = cvSize((*tempimg_src).width, (*tempimg_src).height);
	IplImage* tempimg_dst = cvCreateImage(size1, 8, 3);
	cvSmooth(tempimg_src, tempimg_dst, CV_BLUR, 3, 3, 0, 0);//平滑函数

	//下一帧平滑
	IplImage* frameimg_src;
	frameimg_src = &cvIplImage(frame);
	CvSize size2 = cvSize((*frameimg_src).width, (*frameimg_src).height);
	IplImage* frameimg_dst = cvCreateImage(size2, 8, 3);
	cvSmooth(frameimg_src, frameimg_dst, CV_BLUR, 3, 3, 0, 0);//平滑函数


	cv::Mat temp1, frame1;
	temp1 = cv::cvarrToMat(tempimg_dst);
	frame1 = cv::cvarrToMat(frameimg_dst); //转回为Mat
	//将background和frame转为灰度图  
	cv::Mat gray1, gray2;
	cvtColor(temp1, gray1, CV_BGR2GRAY);
	cvtColor(frame1, gray2, CV_BGR2GRAY);
	//将background和frame做差  
	cv::Mat diff;
	absdiff(gray1, gray2, diff);
	//imshow("帧差图", diff);

	//对差值图diff_thresh进行阈值化处理  二值化
	cv::Mat diff_thresh;
	cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));//函数会返回指定形状和尺寸的结构元素。MORPH_RECT矩形，MORPH_CROSS交叉形，MORPH_ELLIPSE椭圆形															
	//调用之后，调用膨胀与腐蚀函数的时候，第三个参数值保存了getStructuringElement返回值的Mat类型变量，也就是element变量。
	cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(40, 40));
	//imshow("erode", kernel_erode);
	//imshow("dilate", kernel_dilate);
	//进行二值化处理，选择50，255为阈值
	threshold(diff, diff_thresh, 20, 255, CV_THRESH_BINARY);
	//imshow("二值化处理后", diff_thresh);

	//腐蚀  
	erode(diff_thresh, diff_thresh, kernel_erode);
	//imshow("腐蚀处理后", diff_thresh);
	//膨胀  
	dilate(diff_thresh, diff_thresh, kernel_dilate);
	//imshow("膨胀处理后", diff_thresh);

	//查找轮廓并绘制轮廓  
	vector<vector<cv::Point> > contours;
	findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//找轮廓函数
	//drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓  
	 //查找正外接矩形  
	vector<cv::Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		//rectangle(result, boundRect[i], cv::Scalar(0, 255, 0), 2);//在result上绘制正外接矩形  
		
		//drawMosaicRef(result, boundRect[i], 10);
		detectAndDraw(result, cascade, nestedCascade, scale, tryflip);
	}
	

	temp1.release();
	frame1.release();
	gray1.release();
	gray2.release();
	diff.release();
	diff_thresh.release();
	kernel_dilate.release();
	kernel_dilate.release();
	contours.clear();
	boundRect.clear();
	//cvReleaseData(tempimg_src);
	cvReleaseData(tempimg_dst);
	//cvReleaseData(frameimg_src);
	cvReleaseData(frameimg_dst);
	return result;//返回result  

}




struct s
{
	LONG lRealPlayHandle;
};
struct s s1, s2, s3, s4;
struct s ss[3];

void device_init(const char* IPaddress, const char* administrator, const char* password, struct s* s_input)
{
	LONG lUserID;
	NET_DVR_DEVICEINFO_V30 struDeviceInfo;
	lUserID = NET_DVR_Login_V30(const_cast<char*>(IPaddress), 8000, const_cast<char*>(administrator), const_cast<char*>(password), &struDeviceInfo);
	//std::cout << lUserID << endl;
	if (lUserID < 0)
	{
		printf("Login error, %d\n", NET_DVR_GetLastError());
		NET_DVR_Cleanup();
	}
	//设置异常消息回调函数
	NET_DVR_SetExceptionCallBack_V30(0, NULL, g_ExceptionCallBack, NULL);
	//启动预览并设置回调数据流
	char windowname[100] = "\0";
	strcpy_s(windowname, IPaddress);
	cv::namedWindow(windowname, WINDOW_NORMAL);
	LONG lRealPlayHandle;
	HWND  h1 = (HWND)cvGetWindowHandle(const_cast<char*>(windowname));
	NET_DVR_PREVIEWINFO struPlayInfo = { 0 };
	struPlayInfo.hPlayWnd = h1;         //需要SDK解码时句柄设为有效值，仅取流不解码时可设为空
	struPlayInfo.lChannel = 1;           //预览通道号
	struPlayInfo.dwStreamType = 1;       //0-主码流，1-子码流，2-码流3，3-码流4，以此类推
	struPlayInfo.dwLinkMode = 0;         //0- TCP方式，1- UDP方式，2- 多播方式，3- RTP方式，4-RTP/RTSP，5-RSTP/HTTP
	s_input->lRealPlayHandle = NET_DVR_RealPlay_V40(lUserID, &struPlayInfo, NULL, NULL);
	cv::waitKey(1);

	//return s1;
}


vector<cv::Point> corners(2);//图像左角点//indices.size()
vector<CameraParams> cameras;
vector<cv::UMat> masks_warped(2); //indices.size()
vector<cv::Mat> picArray;
vector<int> indices;
float warped_image_scale;
cv::Ptr<cv::WarperCreator> warper_creator; //投影
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;//曝光补偿
cv::Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
vector<cv::Size> sizes(2);

void read_file(string path, vector<Mat> picArray)
{
	vector<string> files;
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	Mat src;
	string format = ".jpg";
	if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1)
	{
		int i = 0;
		do
		{
			// 保存文件的全路径
			files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			cout << files[i] << endl;
			src = cv::imread(files[i]);
			cv::resize(src, src, Size(640, 384), 0, 0, CV_INTER_AREA);
			picArray.push_back(src);
			i++;
		} while (_findnext(hFile, &fileinfo) == 0); //寻找下一个，成功返回0，否则-1
		_findclose(hFile);
	}
}

void restruct(vector<cv::Mat> picArray)
{
	if (warp_type == "plane")
		warper_creator = makePtr<cv::PlaneWarper>();
	else if (warp_type == "affine")
		warper_creator = makePtr<cv::AffineWarper>();
	else if (warp_type == "cylindrical")
		warper_creator = makePtr<cv::CylindricalWarper>();
	else if (warp_type == "spherical")
		warper_creator = makePtr<cv::SphericalWarper>();
	else if (warp_type == "fisheye")
		warper_creator = makePtr<cv::FisheyeWarper>();
	else if (warp_type == "stereographic")
		warper_creator = makePtr<cv::StereographicWarper>();
	else if (warp_type == "compressedPlaneA2B1")
		warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
	else if (warp_type == "compressedPlaneA1.5B1")
		warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
	else if (warp_type == "compressedPlanePortraitA2B1")
		warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
	else if (warp_type == "compressedPlanePortraitA1.5B1")
		warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
	else if (warp_type == "paniniA2B1")
		warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
	else if (warp_type == "paniniA1.5B1")
		warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
	else if (warp_type == "paniniPortraitA2B1")
		warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
	else if (warp_type == "paniniPortraitA1.5B1")
		warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
	else if (warp_type == "mercator")
		warper_creator = makePtr<cv::MercatorWarper>();
	else if (warp_type == "transverseMercator")
		warper_creator = makePtr<cv::CylindricalWarper>();

	warper_creator = cv::makePtr<cv::AffineWarper>();
	

	int num_images = picArray.size();
	cout << num_images << endl;
	vector<ImageFeatures> features(num_images); //存储图像特征点

	cv::Ptr<cv::xfeatures2d::SURF> finder;
	finder = cv::xfeatures2d::SURF::create(2000); //值越大特征点越少

	Mat full_img, img;
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;

	for (int i = 0; i < num_images; ++i)
	{
		computeImageFeatures(finder, picArray[i], features[i]);
	}

	/*drawKeypoints(picArray[0], features[0].getKeypoints(), picArray[0]);
	imshow("Mywindow1", picArray[0]);
	waitKey(10);*/

	vector<MatchesInfo> pairwise_matches;//表示特征匹配信息变量
	MatchesInfo pairwise_matches_single;
	cv::Ptr<FeaturesMatcher> matcher;
	float match_conf = 0.36f;  //越大匹配越难
	matcher = cv::makePtr<AffineBestOf2NearestMatcher>(false, 0, match_conf); //定义特征匹配器，2NN方法
	// AffineBestOf2NearestMatcher BestOf2NearestMatcher BestOf2NearestRangeMatcher

	matcher->operator()(features, pairwise_matches);  // == (*matcher)(features, pairwise_matches); //进行特征匹配
	matcher->collectGarbage();  //释放已被分配但还没有被使用的内存

	//保留图像
	float conf_thresh = 0.1f;
	//vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	//indices = { 3,4 };
	cout << "indices.size:" << indices.size() << endl;


	/*Mat match2;
	drawMatches(picArray[0], features[0].getKeypoints(), picArray[1], features[1].getKeypoints(), pairwise_matches[1].getMatches(), match2);
	cv::imshow("KNN", match2);
	waitKey(10);*/

	for (int i = 0; i < indices.size() * indices.size(); i++)
	{
		cout << "\n" << endl;
		cout << "原始特征点" << pairwise_matches[i].getMatches().size() << endl;
	}

	vector<Point2f> pointq[9];
	vector<Point2f> pointt[9];
	vector<uchar> m[25];
	for (int i = 0; i < indices.size() * indices.size(); i++)
	{
		if (i % (indices.size() + 1) == 0)
		{

		}
		else
		{
			for (int j = 0; j < pairwise_matches[i].getMatches().size(); j++)
			{
				pointq[i].push_back(features[i / indices.size()].keypoints[pairwise_matches[i].matches[j].queryIdx].pt);

				pointt[i].push_back(features[i / indices.size()].keypoints[pairwise_matches[i].matches[j].trainIdx].pt);
			}
			cv::findHomography(pointq[i], pointt[i], cv::RANSAC, 3, m[i], 2000, 0.9);\
			cout << i << endl;
		}
	}
	cout << "indices.size:" << indices.size() << endl;
	vector<DMatch> inliers[100];
	for (int i = 0; i < indices.size() * indices.size(); i++)
	{
		if (i % (indices.size() + 1) == 0)
		{
			//cout << "out" << endl;
		}
		else
		{
			for (int j = 0; j < pairwise_matches[i].getMatches().size(); j++)
			{
				if (m[i][j])
					inliers[i].push_back(pairwise_matches[i].matches[j]);
			}
			pairwise_matches[i].matches.swap(inliers[i]);
			//inliers.clear();
		}
	}

	//for (int i = 0; i < indices.size() * indices.size(); i++)
	//{
	//	cout << "\n" << endl;
	//	cout << "RANSAC特征点：" << pairwise_matches[i].getMatches().size() << endl;
	//}


	/*Mat match;
	drawMatches(picArray[0], features[0].getKeypoints(), picArray[1], features[1].getKeypoints(), pairwise_matches[1].getMatches(), match);
	cv::imshow("RANSAC", match);
	waitKey(10);*/

	//预估相机参数
	cv::Ptr<Estimator> estimator;
	estimator = cv::makePtr<AffineBasedEstimator>(); //HomographyBasedEstimator //AffineBasedEstimator

	//vector<CameraParams> cameras; //相机参数素组
	if (!(*estimator)(features, pairwise_matches, cameras)) //得到相机参数
	{
		cout << "affine estimation failed.\n";
	}
	for (size_t i = 0; i < indices.size(); ++i)
	{
		cv::Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		cout << "Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R << endl;
	}

	//光束平差，精确相机参数
	cv::Ptr<BundleAdjusterBase> adjuster;
	adjuster = cv::makePtr<NoBundleAdjuster>(); //detail::BundleAdjusterReproj  detail::BundleAdjusterRay  detail::BundleAdjusterAffinePartial NoBundleAdjuster
	adjuster->setConfThresh(conf_thresh);

	string ba_refine_mask = "xxxxx";
	cv::Mat_<uchar> refine_mask = cv::Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);
	if (!(*adjuster)(features, pairwise_matches, cameras))
	{
		cout << "Camera parameters adjusting failed.\n";
	}

	//求中值焦距
	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		cout << "Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R << endl;
		focals.push_back(cameras[i].focal);
	}
	sort(focals.begin(), focals.end());
	//float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	//vector<cv::Point> corners(indices.size());//图像左角点
	//vector<cv::UMat> masks_warped(indices.size());
	vector<cv::UMat> images_warped(indices.size());
	//vector<cv::Size> sizes(indices.size());
	vector<cv::UMat> masks(indices.size());

	//准备图像掩膜
	for (int i = 0; i < indices.size(); ++i)
	{
		masks[i].create(picArray[i].size(), CV_8U);
		masks[i].setTo(cv::Scalar::all(255));
	}

	cv::Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(static_cast<float>(warped_image_scale * seam_work_aspect)));//static_cast<float>(cameras[0].focal);//static_cast<float>(warped_image_scale * seam_work_aspect)

	for (int i = 0; i < indices.size(); ++i)
	{
		cv::Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);

		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;
		corners[i] = warper->warp(picArray[i], K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, images_warped[i]);

		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped[i]);
	}
	for (int i = 0; i < indices.size(); ++i)
	{
		//cout << "Image #" << i + 1 << "  corner: " << corners[i] << "  " << "size: " << sizes[i] << endl;
	}
	vector<cv::UMat> images_warped_f(indices.size());
	for (int i = 0; i < indices.size(); ++i)
	{
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
		/*imshow("WARP", images_warped[i]);
		waitKey(0);*/
	}

	int expos_comp_nr_feeds = 1;
	int expos_comp_nr_filtering = 2;
	int expos_comp_block_size = 32;
	//int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;//曝光补偿

	//cv::Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	if (dynamic_cast<GainCompensator*>(compensator.get()))
	{
		GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
		gcompensator->setNrFeeds(expos_comp_nr_feeds);
	}
	if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
	{
		ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
		ccompensator->setNrFeeds(expos_comp_nr_feeds);
	}
	if (dynamic_cast<BlocksCompensator*>(compensator.get()))
	{
		BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
		bcompensator->setNrFeeds(expos_comp_nr_feeds);
		bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
		bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
	}

	compensator->feed(corners, images_warped, masks_warped);


	cv::Ptr<SeamFinder> seam_finder;
	//seam_finder = makePtr<detail::NoSeamFinder>();
	seam_finder = new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	if (!seam_finder)
	{
		cout << "Can't create the following seam finder '" << "'\n";
	}
	seam_finder->find(images_warped_f, corners, masks_warped);


	//images_warped.clear();
	images_warped_f.clear();
	masks.clear();
	picArray.clear();
	before_.clear();
	
}

int main(int argc, char* argv[])
{
	freopen("err.log", "w", stderr);
	NET_DVR_Init();
	//设置连接时间与重连时间
	NET_DVR_SetConnectTime(2000, 1);
	NET_DVR_SetReconnect(10000, true);
	//---------------------------------------



	for (int i = 0; i < device_num; i++) {
		nPort[i] = -1;
	}

	device_init("192.168.1.2", "admin", "HIKVISION610", &s1);
	device_init("192.168.1.3", "admin", "HIKVISION610", &s2);
	device_init("192.168.1.4", "admin", "HIKVISION610", &s3);
	device_init("192.168.1.5", "admin", "HIKVISION610", &s4);
	Sleep(1000);
	//waitKey(0);

	vlc_reader vr1("rtsp://admin:HIKVISION610@192.168.1.3");
	vlc_reader vr2("rtsp://admin:HIKVISION610@192.168.1.2");
	vlc_reader vr3("rtsp://admin:HIKVISION610@192.168.1.4");
	vlc_reader vr4("rtsp://admin:HIKVISION610@192.168.1.5");
	int width = 640, height = 384;
	vr1.start(width, height);
	vr2.start(width, height);
	vr3.start(width, height);
	vr4.start(width, height);
	vector<vlc_reader*> vr = { &vr1 ,&vr2,&vr3,&vr4 };


	//全局
	vector<Mat> fframe;
	Mat ffframe[device_num];

	//vector<int> indices;
	//float warped_image_scale;
	//cv::Ptr<cv::WarperCreator> warper_creator; //投影
	//if (warp_type == "plane")
	//	warper_creator = makePtr<cv::PlaneWarper>();
	//else if (warp_type == "affine")
	//	warper_creator = makePtr<cv::AffineWarper>();
	//else if (warp_type == "cylindrical")
	//	warper_creator = makePtr<cv::CylindricalWarper>();
	//else if (warp_type == "spherical")
	//	warper_creator = makePtr<cv::SphericalWarper>();
	//else if (warp_type == "fisheye")
	//	warper_creator = makePtr<cv::FisheyeWarper>();
	//else if (warp_type == "stereographic")
	//	warper_creator = makePtr<cv::StereographicWarper>();
	//else if (warp_type == "compressedPlaneA2B1")
	//	warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
	//else if (warp_type == "compressedPlaneA1.5B1")
	//	warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
	//else if (warp_type == "compressedPlanePortraitA2B1")
	//	warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
	//else if (warp_type == "compressedPlanePortraitA1.5B1")
	//	warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
	//else if (warp_type == "paniniA2B1")
	//	warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
	//else if (warp_type == "paniniA1.5B1")
	//	warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
	//else if (warp_type == "paniniPortraitA2B1")
	//	warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
	//else if (warp_type == "paniniPortraitA1.5B1")
	//	warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
	//else if (warp_type == "mercator")
	//	warper_creator = makePtr<cv::MercatorWarper>();
	//else if (warp_type == "transverseMercator")
	//	warper_creator = makePtr<cv::CylindricalWarper>();

	//warper_creator = cv::makePtr<cv::AffineWarper>();
	//int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;//曝光补偿
	//cv::Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);

	//背景
	bool stopp(false);

	while (!stopp)
	{
		stopp = TRUE;
		#pragma omp parallel for
		for (int i = 0; i < device_num; i++)
		{
			ffframe[i] = (*vr[i]).frame();
			cv::waitKey(30);
		}
		for (int i = 0; i < device_num; i++)
		{
			if (ffframe[i].rows == 0)
				stopp = FALSE;
		}
		if (stopp == FALSE)
		{
			cout << "blackground failure" << endl;
		}
	}
	if (stopp)
	{
		std::cout << "blackground sucess" << endl;

		for (int i = 0; i < 2; i++)
		{
			cv::resize(ffframe[i], ffframe[i], Size(640, 384), 0, 0, CV_INTER_AREA);
			picArray.push_back(ffframe[i]);
		}
		/*Mat src;
		src = cv::imread("F:\\20201121\\yaopinjie\\1\\14.jpg");
		cv::resize(src, src, Size(640, 384), 0, 0, CV_INTER_AREA);
		picArray.push_back(src);

		src = cv::imread("F:\\20201121\\yaopinjie\\2\\14.jpg");
		cv::resize(src, src, Size(640, 384), 0, 0, CV_INTER_AREA);
		picArray.push_back(src);*/

		restruct(picArray);
		picArray.clear();
		/*imwrite("s2.bmp", ffframe[0]);*/
	
		////到此
		////int num_images = picArray.size();
		////cout << num_images << endl;
		////vector<ImageFeatures> features(num_images); //存储图像特征点
		//////cv::Ptr<cv::Feature2D> finder;
		//////finder = cv::ORB::create();
		////cv::Ptr<cv::xfeatures2d::SURF> finder;
		////finder = cv::xfeatures2d::SURF::create(2000); //值越大特征点越少
		////Mat full_img, img;
		////vector<Mat> images(num_images);
		////vector<Size> full_img_sizes(num_images);
		////double seam_work_aspect = 1;
		////for (int i = 0; i < num_images; ++i)
		////{
		////	computeImageFeatures(finder, picArray[i], features[i]);
		////}
		/////*drawKeypoints(picArray[0], features[0].getKeypoints(), picArray[0]);
		////imshow("Mywindow1", picArray[0]);
		////waitKey(10);
		////drawKeypoints(picArray[1], features[1].getKeypoints(), picArray[1]);
		////imshow("pic1", picArray[1]);
		////waitKey(10);*/
		////vector<MatchesInfo> pairwise_matches;//表示特征匹配信息变量
		////MatchesInfo pairwise_matches_single;
		////cv::Ptr<FeaturesMatcher> matcher;
		////float match_conf = 0.36f;  //越大匹配越难
		////matcher = cv::makePtr<AffineBestOf2NearestMatcher>(false, 0, match_conf); //定义特征匹配器，2NN方法
		////// AffineBestOf2NearestMatcher BestOf2NearestMatcher BestOf2NearestRangeMatcher
		////matcher->operator()(features, pairwise_matches);  // == (*matcher)(features, pairwise_matches); //进行特征匹配
		////matcher->collectGarbage();  //释放已被分配但还没有被使用的内存
		//////保留图像
		////float conf_thresh = 0.1f;
		//////vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
		////indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
		//////indices = { 3,4 };
		////cout << "indices.size:" << indices.size() << endl;
		/////*Mat match2;
		////drawMatches(picArray[0], features[0].getKeypoints(), picArray[1], features[1].getKeypoints(), pairwise_matches[1].getMatches(), match2);
		////cv::imshow("KNN", match2);
		////waitKey(10);*/
		/////*for (int i = 0; i < indices.size() * indices.size(); i++)
		////{
		////	cout << "\n" << endl;
		////	cout << "原始特征点" << pairwise_matches[i].getMatches().size() << endl;
		////}*/
		////vector<Point2f> pointq[9];
		////vector<Point2f> pointt[9];
		////vector<uchar> m[25];
		////for (int i = 0; i < indices.size() * indices.size(); i++)
		////{
		////	if (i % (indices.size() + 1) == 0)
		////	{
		////	}
		////	else
		////	{
		////		for (int j = 0; j < pairwise_matches[i].getMatches().size(); j++)
		////		{
		////			pointq[i].push_back(features[i / indices.size()].keypoints[pairwise_matches[i].matches[j].queryIdx].pt);
		////			pointt[i].push_back(features[i / indices.size()].keypoints[pairwise_matches[i].matches[j].trainIdx].pt);
		////		}
		////		cv::findHomography(pointq[i], pointt[i], cv::RANSAC, 3, m[i], 2000, 0.9);
		////	}
		////}
		////vector<DMatch> inliers[100];
		////for (int i = 0; i < indices.size() * indices.size(); i++)
		////{
		////	if (i % (indices.size() + 1) == 0)
		////	{
		////		//cout << "out" << endl;
		////	}
		////	else
		////	{
		////		for (int j = 0; j < pairwise_matches[i].getMatches().size(); j++)
		////		{
		////			if (m[i][j])
		////				inliers[i].push_back(pairwise_matches[i].matches[j]);
		////		}
		////		pairwise_matches[i].matches.swap(inliers[i]);
		////		//inliers.clear();
		////	}
		////}
		////for (int i = 0; i < indices.size() * indices.size(); i++)
		////{
		////	cout << "\n" << endl;
		////	cout << "RANSAC特征点：" << pairwise_matches[i].getMatches().size() << endl;
		////}
		/////*Mat match;
		////drawMatches(picArray[0], features[0].getKeypoints(), picArray[1], features[1].getKeypoints(), pairwise_matches[1].getMatches(), match);
		////cv::imshow("RANSAC", match);
		////waitKey(10);*/
		//////预估相机参数
		////cv::Ptr<Estimator> estimator;
		////estimator = cv::makePtr<AffineBasedEstimator>(); //HomographyBasedEstimator //AffineBasedEstimator
		//////vector<CameraParams> cameras; //相机参数素组
		////if (!(*estimator)(features, pairwise_matches, cameras)) //得到相机参数
		////{
		////	cout << "affine estimation failed.\n";
		////}
		////for (size_t i = 0; i < indices.size(); ++i)
		////{
		////	cv::Mat R;
		////	cameras[i].R.convertTo(R, CV_32F);
		////	cameras[i].R = R;
		////	cout << "Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R << endl;
		////}
		//////光束平差，精确相机参数
		////cv::Ptr<BundleAdjusterBase> adjuster;
		////adjuster = cv::makePtr<NoBundleAdjuster>(); //detail::BundleAdjusterReproj  detail::BundleAdjusterRay  detail::BundleAdjusterAffinePartial NoBundleAdjuster
		////adjuster->setConfThresh(conf_thresh);
		////string ba_refine_mask = "xxxxx";
		////cv::Mat_<uchar> refine_mask = cv::Mat::zeros(3, 3, CV_8U);
		////if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
		////if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
		////if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
		////if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
		////if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
		////adjuster->setRefinementMask(refine_mask);
		////if (!(*adjuster)(features, pairwise_matches, cameras))
		////{
		////	cout << "Camera parameters adjusting failed.\n";
		////}
		//////求中值焦距
		////vector<double> focals;
		////for (size_t i = 0; i < cameras.size(); ++i)
		////{
		////	cout << "Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R << endl;
		////	focals.push_back(cameras[i].focal);
		////}
		////sort(focals.begin(), focals.end());
		//////float warped_image_scale;
		////if (focals.size() % 2 == 1)
		////	warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
		////else
		////	warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
		//////vector<cv::Point> corners(indices.size());//图像左角点
		//////vector<cv::UMat> masks_warped(indices.size());
		////vector<cv::UMat> images_warped(indices.size());
		////vector<cv::Size> sizes(indices.size());
		////vector<cv::UMat> masks(indices.size());
		//////准备图像掩膜
		////for (int i = 0; i < indices.size(); ++i)
		////{
		////	masks[i].create(picArray[i].size(), CV_8U);
		////	masks[i].setTo(cv::Scalar::all(255));
		////}
		////cv::Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(static_cast<float>(warped_image_scale * seam_work_aspect)));//static_cast<float>(cameras[0].focal);//static_cast<float>(warped_image_scale * seam_work_aspect)
		////for (int i = 0; i < indices.size(); ++i)
		////{
		////	cv::Mat_<float> K;
		////	cameras[i].K().convertTo(K, CV_32F);
		////	float swa = (float)seam_work_aspect;
		////	K(0, 0) *= swa; K(0, 2) *= swa;
		////	K(1, 1) *= swa; K(1, 2) *= swa;
		////	corners[i] = warper->warp(picArray[i], K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, images_warped[i]);
		////	sizes[i] = images_warped[i].size();
		////	warper->warp(masks[i], K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped[i]);
		////}
		////for (int i = 0; i < indices.size(); ++i)
		////{
		////	cout << "Image #" << i + 1 << "  corner: " << corners[i] << "  " << "size: " << sizes[i] << endl;
		////}
		////vector<cv::UMat> images_warped_f(indices.size());
		////for (int i = 0; i < indices.size(); ++i)
		////{
		////	images_warped[i].convertTo(images_warped_f[i], CV_32F);
		////	/*imshow("WARP", images_warped[i]);
		////	waitKey(0);*/
		////}
		////int expos_comp_nr_feeds = 1;
		////int expos_comp_nr_filtering = 2;
		////int expos_comp_block_size = 32;
		//////int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;//曝光补偿
		//////cv::Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
		////if (dynamic_cast<GainCompensator*>(compensator.get()))
		////{
		////	GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
		////	gcompensator->setNrFeeds(expos_comp_nr_feeds);
		////}
		////if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
		////{
		////	ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
		////	ccompensator->setNrFeeds(expos_comp_nr_feeds);
		////}
		////if (dynamic_cast<BlocksCompensator*>(compensator.get()))
		////{
		////	BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
		////	bcompensator->setNrFeeds(expos_comp_nr_feeds);
		////	bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
		////	bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
		////}
		////compensator->feed(corners, images_warped, masks_warped);
		////cv::Ptr<SeamFinder> seam_finder;
		//////seam_finder = makePtr<detail::NoSeamFinder>();
		////seam_finder = new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
		////if (!seam_finder)
		////{
		////	cout << "Can't create the following seam finder '" << "'\n";
		////}
		////seam_finder->find(images_warped_f, corners, masks_warped);
		//////images_warped.clear();
		////images_warped_f.clear();
		////masks.clear();
		////picArray.clear();

		//实时
		stopp = false;
		int flag = 0;
		int index = 0;
		while (!stop)
		{
			
			while (!stopp)
			{
				stopp = TRUE;
				#pragma omp parallel for
				for (int i = 0; i < device_num; i++)
				{
					ffframe[i] = (*vr[i]).frame();
					cv::waitKey(30);
				}
				for (int i = 0; i < device_num; i++)
				{
					if (ffframe[i].rows == 0)
						stopp = FALSE;
				}
				if (stopp == FALSE)
				{
					cout << "failure2" << endl;
				}
			}
			if (stopp)
			{
				std::cout << "取流成功2" << endl;

				for (int i = 0; i < 2; i++)
				{

					cv::resize(ffframe[i], ffframe[i], Size(640, 384), 0, 0, CV_INTER_AREA);
					picArray.push_back(ffframe[i]);
				}
				/*string p;
				string format = ".jpg";
				p.assign("F:\\1120\\已矫正\\1.8\\").append(to_string(index + 1) + format).c_str();
				src = cv::imread(p);
				cv::resize(src, src, Size(640, 384), 0, 0, CV_INTER_AREA);
				picArray.push_back(src); 
				p.assign("F:\\1120\\已矫正\\1.9\\").append(to_string(index + 1) + format).c_str();
				src = cv::imread(p);
				cv::resize(src, src, Size(640, 384), 0, 0, CV_INTER_AREA);
				picArray.push_back(src);*/
				
				/*imwrite("s2.bmp", ffframe[0]);*/

				/*Mat src;
				src = cv::imread("D:\\opencv-build\\test\\Project1\\img\\test1.jpg");
				cv::resize(src, src, Size(640, 384), 0, 0, CV_INTER_AREA);
				picArray.push_back(src);*/

				stopp = FALSE;

				cv::Mat img_warped, img_warped_s;
				cv::Mat dilated_mask, seam_mask, mask, mask_warped;
				cv::Ptr<Blender> blender;
				cv::Ptr<Timelapser> timelapser;
				double compose_work_aspect = 1;

				//bool timelapse = false;
				int blend_type = Blender::MULTI_BAND; //Blender::FEATHER;// Blender::MULTI_BAND;
				//float blend_strength = 5;
				////int timelapse_type = Timelapser::AS_IS;
				for (int img_idx = 0; img_idx < 2; ++img_idx)
				{
					cout << "Compositing image #" << indices[img_idx] + 1 << endl;
					// Read image and resize it if necessary
					cv::Mat K;
					cout << picArray.size() << endl;
					warped_image_scale *= static_cast<float>(compose_work_aspect);
					cv::Ptr<RotationWarper> warper = warper_creator->create(warped_image_scale);

					cameras[img_idx].K().convertTo(K, CV_32F);
					warper->warp(picArray[img_idx], K, cameras[img_idx].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, img_warped);
					cv::Size img_size = picArray[img_idx].size();
					mask.create(img_size, CV_8U);
					mask.setTo(cv::Scalar::all(255));
					warper->warp(mask, K, cameras[img_idx].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped);
					compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
					img_warped.convertTo(img_warped_s, CV_16S);

					//img_warped.release();
					mask.release();

					cv::dilate(masks_warped[img_idx], dilated_mask, cv::Mat());
					cv::resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, cv::INTER_LINEAR_EXACT);
					mask_warped = seam_mask & mask_warped;

					if (!blender && !timelapse)
					{
						blender = Blender::createDefault(blend_type, 0);
						cv::Size dst_sz = resultRoi(corners, sizes).size();
						float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
						if (blend_width < 1.f)
							blender = Blender::createDefault(Blender::NO, 0);
						else if (blend_type == Blender::MULTI_BAND)
						{
							MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
							mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
							cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
						}
						else if (blend_type == Blender::FEATHER)
						{
							FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
							fb->setSharpness(1.f / blend_width);
							cout << "Feather blender, sharpness: " << fb->sharpness() << endl;
						}
						blender->prepare(corners, sizes); //确定图像大小
					}
					else if (!timelapser && timelapse)
					{
						timelapser = Timelapser::createDefault(timelapse_type);
						timelapser->initialize(corners, sizes);
					}
					//img_warped_s.convertTo(img_warped_s, CV_8U);
					blender->feed(img_warped_s, mask_warped, corners[img_idx]);

				}
	
				//融合
				Mat result, result_mask;
				if (!timelapse)
				{
					
					blender->blend(result, result_mask);
					result.convertTo(result, CV_8U);
					////运动物体检测							
					Mat resultt;
					if (before.rows == 0||flag==1) {
						before = result.clone();
						cout << "re" << endl;
					}
					resultt = MoveDetect(before, result);
					cv::imshow("stitch", resultt);
					before = result.clone();
					index++;
					//imwrite("F:\\20201121\\yaopinjie\\3\\"+ to_string(index) + ".jpg", result);
				}
				flag = 0;
				Mat stitch_seam;


				Mat edge;
				Canny(mask_warped, edge, 3, 9, 3);
				//cv::imshow("edge", edge);
				Mat dilate_img;
				Mat element = getStructuringElement(MORPH_RECT, Size(30, 30));
				dilate(edge, dilate_img, element);
				cv::imshow("dilate_img", dilate_img);


				img_warped.copyTo(stitch_seam, dilate_img);
				//stitch_seam2.setTo(0, dilate_img);
				cv::imshow("stitch_seam2", stitch_seam);
				if (before_.size() == 0)
				{
					cout << "ggggg" << endl;
					before_.push_back(stitch_seam); //= stitch_seam2.clone();
				}
				/*cout << before_[0].size() << endl;
				cout << stitch_seam2.size() << endl;*/

				flag = seam_detect(before_[0], stitch_seam);
				before_[0] = stitch_seam.clone();
				cout << flag << endl;
				waitKey(1);
				//重新计算背景
				if (flag == 1)
				{
					restruct(picArray);
					cout << "restruct" << endl;
				}
				
				picArray.clear();
				result_mask.release();
				blender.release();
				result.release();
				//resultt.release();
			}
			else
			{
				std::cout << "----------------------" << endl;
				std::cout << "waitting..." << endl;
				fframe.clear();
			}

		}
	}
	return 0;


}
#else
int main()
{
	std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
	return 0;
}
#endif


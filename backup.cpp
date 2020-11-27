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

#include "omp.h"


using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace cv::xfeatures2d;

#define device_num 4
#define level_num 0
static long nPort[device_num];
static queue<cv::Mat> video_queue[device_num];
static long port_num[device_num];

int my_index[2] = { 0,0 };
cv::Mat my_picArray[2];
void yv12toYUV(char* outYuv, char* inYv12, int width, int height, int widthStep)
{
	int col, row;
	unsigned int Y, U, V;
	int tmp;
	int idx;

	//printf("widthStep=%d.\n",widthStep);

	for (row = 0; row < height; row++)
	{
		idx = row * widthStep;
		int rowptr = row * width;

		for (col = 0; col < width; col++)
		{
			//int colhalf=col>>1;
			tmp = (row / 2) * (width / 2) + (col / 2);
			//         if((row==1)&&( col>=1400 &&col<=1600))
			//         { 
			//          printf("col=%d,row=%d,width=%d,tmp=%d.\n",col,row,width,tmp);
			//          printf("row*width+col=%d,width*height+width*height/4+tmp=%d,width*height+tmp=%d.\n",row*width+col,width*height+width*height/4+tmp,width*height+tmp);
			//         } 
			Y = (unsigned int)inYv12[row * width + col];
			U = (unsigned int)inYv12[width * height + width * height / 4 + tmp];
			V = (unsigned int)inYv12[width * height + tmp];
			//         if ((col==200))
			//         { 
			//         printf("col=%d,row=%d,width=%d,tmp=%d.\n",col,row,width,tmp);
			//         printf("width*height+width*height/4+tmp=%d.\n",width*height+width*height/4+tmp);
			//         return ;
			//         }
			if ((idx + col * 3 + 2) > (1200 * widthStep))
			{
				//printf("row * widthStep=%d,idx+col*3+2=%d.\n",1200 * widthStep,idx+col*3+2);
			}
			outYuv[idx + col * 3] = Y;
			outYuv[idx + col * 3 + 1] = U;
			outYuv[idx + col * 3 + 2] = V;
		}
	}
	//printf("col=%d,row=%d.\n",col,row);
}


//解码回调 视频为YUV数据(YV12)，音频为PCM数据
void CALLBACK DecCBFun(long Port, char* pBuf, long nSize, FRAME_INFO* pFrameInfo, long nReserved1, long nReserved2)
{

	long lFrameType = pFrameInfo->nType;
	int imagename = 0;
	if (lFrameType == T_YV12)
	{
	#if USECOLOR
		int start = clock();
		IplImage* pImgYCrCb = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 3);//得到图像的Y分量  
		yv12toYUV(pImgYCrCb->imageData, pBuf, pFrameInfo->nWidth, pFrameInfo->nHeight, pImgYCrCb->widthStep);//得到所有RGB图像
		IplImage* pImg = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 3);
		cvCvtColor(pImgYCrCb, pImg, CV_YCrCb2RGB);
		int end = clock();
		time_t now = time(0);
		//printf("%d,%d\n",Port, now);
	#else
		IplImage* pImg = cvCreateImage(cvSize(pFrameInfo->nWidth, pFrameInfo->nHeight), 8, 1);
		memcpy(pImg->imageData, pBuf, pFrameInfo->nWidth * pFrameInfo->nHeight);
#endif
		int i = 0;
		int index = 0;
		for (i = 0; i < device_num; i++) //
		{
			if (port_num[i] == Port)
				index = 1; 
		}
		if (index == 0)
		{
			for (i = 0; i < device_num; i++)
			{
				if (port_num[i] == 0)
				{
					port_num[i] = Port;
					break;
				}
			}
		}
		
		cv::Mat img1 = cv::cvarrToMat(pImg, true);


		//修改代码
		for (i = 0; i < device_num; i++)
		{
			if (port_num[i] == Port)   //定位当前设备
			{
				my_index[i]++; 
				//printf("%d,%d\n", Port, my_index[i]);
				cv::resize(img1, img1, Size(640, 384), 0, 0, CV_INTER_AREA);
				my_picArray[i] = img1;//图片队列
				break;
			}
		}
#if USECOLOR
		
		cvReleaseImage(&pImgYCrCb);
		cvReleaseImage(&pImg);
#else
		cvReleaseImage(&pImg);
#endif
		//此时是YV12格式的视频数据，保存在pBuf中，能够fwrite(pBuf,nSize,1,Videofile);
		//fwrite(pBuf,nSize,1,fp);
	}
	/***************
	else if (lFrameType ==T_AUDIO16)
	{
		//此时是音频数据，数据保存在pBuf中。能够fwrite(pBuf,nSize,1,Audiofile);

	}
	else
	{
	}
	*******************/

}

///实时流回调
void CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE* pBuffer, DWORD dwBufSize, void* pUser)
{
	DWORD dRet;
	//dwDataType = NET_DVR_SYSHEAD;
	HWND hWnd = GetConsoleWindow();
	DWORD CameraIndex = 0;
	CameraIndex = lRealHandle;
	//printf("lRealHandle = %ld\n", CameraIndex);
	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD:    //系统头
		if (!PlayM4_GetPort(&nPort[CameraIndex])) //获取播放库未使用的通道号
		{
			break;
		}
		if (dwBufSize > 0)
		{
			if (!PlayM4_OpenStream(nPort[CameraIndex], pBuffer, dwBufSize, 640 * 384))
			{
				dRet = PlayM4_GetLastError(nPort[CameraIndex]);
				break;
			}

			//设置解码回调函数 只解码不显示
			if (!PlayM4_SetDecCallBack(nPort[CameraIndex], DecCBFun))
			{
				dRet = PlayM4_GetLastError(nPort[CameraIndex]);
				break;
			}

			//设置解码回调函数 解码且显示
			/*if (!PlayM4_SetDecCallBackEx(nPort,DecCBFun,NULL,NULL))
			{
				dRet=PlayM4_GetLastError(nPort);
				break;
			}*/

			//打开视频解码
			if (!PlayM4_Play(nPort[CameraIndex], hWnd))
			{
				dRet = PlayM4_GetLastError(nPort[CameraIndex]);
				break;
			}

			//打开音频解码, 需要码流是复合流
			if (!PlayM4_PlaySound(nPort[CameraIndex]))
			{
				dRet = PlayM4_GetLastError(nPort[CameraIndex]);
				break;
			}
		}
		break;

	case NET_DVR_STREAMDATA:   //码流数据
		if (dwBufSize > 0 && nPort[CameraIndex] != -1)
		{
			BOOL inData = PlayM4_InputData(nPort[CameraIndex], pBuffer, dwBufSize);
			while (!inData)
			{
				Sleep(10);
				inData = PlayM4_InputData(nPort[CameraIndex], pBuffer, dwBufSize);
				//OutputDebugString("PlayM4_InputData failed \n");
			}
		}
		break;
	}
}


int getmax()
{
	Py_Initialize();              //初始化，创建一个Python虚拟环境
	if (!Py_IsInitialized())
	{
		printf("Python环境初始化失败...\n");
	}
	else
	{
		PyObject* pModule = NULL;
		PyObject* pFunc = NULL;
		pModule = PyImport_ImportModule("readmax.py");  //参数为Python脚本的文件名
		if (!pModule)
		{
			printf("导入Python模块失败...\n");
		}
		else
		{
			pFunc = PyObject_GetAttrString(pModule, "getmax");   //获取函数
			PyObject* pReturn = PyEval_CallObject(pFunc, NULL);
			int nResult;
			PyArg_Parse(pReturn, "i", &nResult);//执行函数
			std::cout << nResult << std::endl;
			return nResult;
		}
	}

	Py_Finalize();
	return 0;

}

//void read_file(string path,vector<Mat> picArray)
//{
//	//string path = "D:\\实习\\stitch_opencv4.2_0915\\img" ;
//	vector<string> files;
//	intptr_t hFile = 0;
//	struct _finddata_t fileinfo;
//	string p;
//	Mat src;
//	string format = ".jpg";
//	if ((hFile = _findfirst(p.assign(path).append("\\*"+ format).c_str(), &fileinfo)) != -1) 
//	{
//		int i = 0;
//		do 
//		{
//			// 保存文件的全路径
//			files.push_back(p.assign(path).append("\\").append(fileinfo.name));
//			cout<<files[i]<<endl;
//			src = cv::imread(files[i]);
//			cv::resize(src, src, Size(640, 384), 0, 0, CV_INTER_AREA);
//			picArray.push_back(src);
//			i++;
//		} 
//		while (_findnext(hFile, &fileinfo) == 0); //寻找下一个，成功返回0，否则-1
//		_findclose(hFile);
//	}
//}

//局部特征点
void local_match(vector<Mat> picArray)
{
	vector<cv::Mat> image;
	Rect rect1(440, 0, 200, 384);
	Mat mask1;
	mask1 = Mat::zeros(picArray[0].size(), CV_8UC1);
	mask1(rect1).setTo(255);
	Mat image_t;
	picArray[0].copyTo(image_t, mask1);
	image.push_back(image_t);
	//picArray[0] = image[0];

	Rect rect2(0, 0, 200, 384);
	Mat mask2;
	mask2 = Mat::zeros(picArray[0].size(), CV_8UC1);
	mask2(rect2).setTo(255);
	Mat image_g;
	picArray[1].copyTo(image_g, mask2);
	image.push_back(image_g);
}

//高斯建模
void gaussian_background()
{
	
	Mat mask1,mask2,mask3;
	Mat ground1,ground2,ground3;
	cv::Ptr<cv::BackgroundSubtractorMOG2> subtractor1, subtractor2, subtractor3;
	subtractor1 = createBackgroundSubtractorMOG2(20, 50);
	subtractor2 = createBackgroundSubtractorMOG2(20, 50);
	subtractor3 = createBackgroundSubtractorMOG2(20, 50);

	subtractor1->apply(my_picArray[0], mask1, 0);
	subtractor1->getBackgroundImage(ground1);
	imshow("ground1", ground1);
	subtractor2->apply(my_picArray[1], mask2, 0);
	subtractor2->getBackgroundImage(ground2);
	imshow("ground2", ground2);
	subtractor3->apply(my_picArray[2], mask3, 0);
	subtractor3->getBackgroundImage(ground3);
	imshow("ground3", ground3); 
}



#endif
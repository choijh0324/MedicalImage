#pragma once
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <list>

using namespace std;
using namespace cv;


namespace geo
{
	typedef struct geometric
	{
		string name;
		int top_x;
		int top_y;
		int bottom_x;
		int bottom_y;

		int label;
		int x;
		int y;
		float ratio;
		int area;
	}geometric;


	/**
	@class Geometry
	@data 2018/04/17
	*/
	class Geometry
	{
	private:
		vector<geometric> test;
		vector<geometric> extractor;
		vector<int> drawing_histogram(Mat image, int state);
		void build_geometry(string img_dir, string roi_text, int label, vector <geometric> &list);
		Mat convertBinary(Mat input);
		geometric getGeometric(string image, Point leftTop, Point rightDown);
		vector<std::string> split(const string& s, char delimiter);
		typedef struct score
		{
			string name;
			int label;
			double distance;
			int index;
		}score;
	public:
		Geometry();
		void loadImages(char** images, char** bbox, int length);///< 전체 이미지를 불러오고 피쳐 어레이를 만든다
		char** get_top_ten_location(const char * test_img, int left, int right, int top, int down, int label[], int index[]);///< 해당 테스트 이미지의 위치가 제일 가까운 열개의 이미지를 불러온다
		char**  get_top_ten_area(const char * test_img, int left, int right, int top, int down, int label[], int index[]);///< 해당 테스트 이미지의 넓이가 제일 가까운 열개의 이미지를 불러온다
		char**  get_top_ten_ratio(const char * test_img, int left, int right, int top, int down, int label[], int index[]);///< 해당 테스트 이미지의 비율이 제일 가까운 열개의 이미지를 불러온다
		void showImage(int index, int rank);
	};
};
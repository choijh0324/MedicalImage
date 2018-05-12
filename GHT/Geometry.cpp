#include "Geometry.h"

namespace geo
{
	vector<int> Geometry::drawing_histogram(Mat image, int state)
	{
		vector<int> histogram;
		if (state == 0)
		{
			for (int i = 0; i < image.rows; i++)
			{
				int count = 0;
				for (int j = 0; j < image.cols; j++) {
					uchar pix = image.at<uchar>(Point(j, i));
					if (pix == 255)
					{
						count++;
					}
				}
				histogram.push_back(count);
			}
		}
		else
		{
			for (int i = 0; i < image.cols; i++)
			{
				int count = 0;
				for (int j = 0; j < image.rows; j++) {
					if (image.at<uchar>(Point(i, j)) == 255)
					{
						count++;
					}
				}
				histogram.push_back(count);
			}
		}
		return histogram;
	}
	void Geometry::build_geometry(string img_dir, string roi_text, int label, vector<geometric>& list)
	{
		typedef struct roi
		{
			int start_x;
			int start_y;
			int end_x;
			int end_y;
		}roi;
		vector<string> dir_ = split(img_dir, '/');
		string dir = dir_[dir_.size() - 1];
		int ROI_num = 0;
		int beg_X, beg_Y;
		int end_X, end_Y;
		int b = 255;
		float trans_thresh[256];
		Mat img, modified, target;

		vector<roi> rois;

		img = imread(img_dir);
		if (!img.data)
			return;
		int width = img.size().width;
		int height = img.size().height;

		if (width == 0 || height == 0)
			return;
		float w_ratio = 600.0 / (float)width;
		float h_ratio = 600.0 / (float)height;



		//find corner color
		Scalar avg = 0;
		resize(img, modified, Size(), w_ratio, h_ratio, CV_INTER_NN);
		cvtColor(modified, modified, CV_BGR2GRAY);


		//roi
		ifstream fileReader;
		fileReader.open(roi_text);
		int num_Roi;
		fileReader >> num_Roi;
		for (int i = 0; i < num_Roi; i++)
		{
			int start_x, start_y, end_x, end_y;
			fileReader >> start_x >> start_y >> end_x >> end_y;
			roi temp;
			temp.start_x = (int)((float)start_x * w_ratio);
			temp.start_y = (int)((float)start_y * h_ratio);
			temp.end_x = (int)((float)end_x * w_ratio);
			temp.end_y = (int)((float)end_y * h_ratio);
			rois.push_back(temp);
		}
		fileReader.close();
		//cout << w_ratio << " " << h_ratio << " ";

		resize(img, img, Size(), w_ratio, h_ratio, CV_INTER_NN);

		//imshow("converted", bin);


		int point1_x = 0, point1_y = 0;
		int point2_x = 0, point2_y = 0;

		int point3_x = 300, point3_y = 0;
		int point4_x = 300, point4_y = 0;

		Mat bin = convertBinary(modified);

		vector<int>hist1 = drawing_histogram(bin(Rect(0, 0, 300, 600)), 0);
		vector<int>hist2 = drawing_histogram(bin(Rect(0, 0, 300, 600)), 1);
		vector<int>hist3 = drawing_histogram(bin(Rect(300, 0, 300, 600)), 0);
		vector<int>hist4 = drawing_histogram(bin(Rect(300, 0, 300, 600)), 1);

		vector<int>peaks;
		for (int i = 20; i < bin.rows - 15; i += 1)
		{
			int intensity = hist1[i];
			if (intensity >= hist1[i - 1] && intensity > hist1[i - 2] && intensity >= hist1[i + 1] && intensity > hist1[i + 2])
			{
				peaks.push_back(i);
			}
		}
		//cout << "hist 1:" << peaks.size() << endl;
		point1_y = peaks[0];
		point2_y = peaks[peaks.size() - 1];
		peaks.clear();
		for (int i = 2; i < bin.cols / 2 - 2; i += 1)
		{
			int intensity = hist2[i];
			if (intensity > 100 && intensity >= hist2[i - 1] && intensity > hist2[i - 2] && intensity >= hist2[i + 1] && intensity > hist2[i + 2])
			{
				peaks.push_back(i);
			}
			if (intensity == 600)
			{
				point1_x == i;
				break;
			}
		}

		point2_x = peaks[0];
		//cout << "hist 2:" << peaks.size() << endl;
		peaks.clear();
		for (int i = 20; i < bin.rows - 15; i += 1)
		{
			int intensity = hist3[i];
			if (intensity >= hist3[i - 1] && intensity > hist3[i - 2] && intensity >= hist3[i + 1] && intensity > hist3[i + 2])
			{
				peaks.push_back(i);
			}
		}
		//cout << "hist 3:" << peaks.size() << endl;

		point3_y = peaks[0];
		point4_y = peaks[peaks.size() - 1];
		peaks.clear();

		for (int i = (bin.cols) - 3; i > (bin.cols) / 2 + 2; i -= 1)
		{
			int intensity = hist4[i - 300];
			if (intensity > 100 && intensity >= hist4[i - 301] && intensity > hist4[i - 302] && intensity > hist4[i - 298] && intensity >= hist4[i - 299])
			{
				peaks.push_back(i);
			}
			if (intensity == 600)
			{
				point3_x == i;
				break;
			}
		}
		//cout << "hist 4:" << peaks.size() << endl;
		point4_x = peaks[0];
		//Mat binary, canny, final_image;


		//namedWindow("Finding Threshold", CV_WINDOW_AUTOSIZE);
		//binary = convert_to_binary_image(target, threshold_value, 255);
		Mat element = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * 5 + 1, 2 * 5 + 1),
			Point(5, 5));

		int min1_y, max2_y;
		if (point1_y < point3_y)
			min1_y = point1_y;
		else
			min1_y = point3_y;

		if (point2_y > point4_y)
			max2_y = point2_y;
		else
			max2_y = point4_y;

		int l_width = point4_x - point2_x;
		int l_height = max2_y - min1_y;

		int translate_x = 300 - ((point4_x + point2_x) / 2);
		int translate_y = 300 - ((max2_y + min1_y) / 2);
		float lw_ratio = 200.0 / (float)l_width;
		float lh_ratio = 200.0 / (float)l_height;
		if (l_width <= 0 || l_height <= 0)
			return;
		//cvtColor(final_img, final_img, CV_GRAY2RGB);
		cvtColor(target, target, CV_GRAY2RGB);
		//rectangle(final_img, Point(point2_x + 3, min1_y), Point(point4_x, max2_y), Scalar(0, 0, 255), 5);

		for (int i = 0; i < rois.size(); i++)
		{
			rois[i].start_x = (int)((float)(rois[i].start_x + translate_x) * lw_ratio);
			rois[i].end_x = (int)((float)(rois[i].end_x + translate_x) * lw_ratio);
			rois[i].start_y = (int)((float)(rois[i].start_y + translate_y) * lh_ratio);
			rois[i].end_y = (int)((float)(rois[i].end_y + translate_y) * lh_ratio);


		}

		Mat aligned = img(Rect(Point(point2_x + 3, min1_y), Point(point4_x, max2_y)));
		//rectangle(img, Point(point2_x + 3, min1_y), Point(point4_x, max2_y), Scalar(0, 0, 255), 5);
		Mat transform_M = (Mat_<double>(2, 3) << 1, 0, translate_x, 0, 1, translate_y);
		//Mat aligned;
		warpAffine(img, aligned, transform_M, img.size());
		//cout << lw_ratio << " " << lh_ratio << endl;
		resize(aligned, aligned, Size(), lw_ratio, lh_ratio, CV_INTER_NN);


		for (int i = 0; i < rois.size(); i++)
		{
			//rectangle(aligned, Point(rois[i].start_x, rois[i].start_y), Point(rois[i].end_x, rois[i].end_y), Scalar(0, 255, 0), 1);
			int area = (rois[i].end_x - rois[i].start_x) * (rois[i].end_y - rois[i].start_y);
			int center_x = (rois[i].start_x + rois[i].end_x) / 2;
			int center_y = (rois[i].start_y + rois[i].end_y) / 2;
			int axis_x = (aligned.size().width / 2) - center_x;
			int axis_y = (aligned.size().height / 2) - center_y;
			//double distance_ = sqrt((double)(axis_x*axis_x + axis_y*axis_y));
			//double angle_ = atan2((double)axis_x, (double)axis_y);
			double ratio_ = (double)(rois[i].end_x - rois[i].start_x) / (double)(rois[i].end_y - rois[i].start_y);
			geometric temp;
			if (area > 15000)
				area = 15000;
			if (ratio_ > 3.0)
				ratio_ = 3.0;
			temp.name = dir.substr(0,dir.size()-4)+"_"+to_string(i)+".png";
			temp.label = label;
			temp.area = area;
			temp.x = axis_x;
			temp.y = axis_y;
			temp.ratio = ratio_;
			temp.top_x = rois[i].end_x / lw_ratio - translate_x;
			temp.top_y = rois[i].end_y / lh_ratio - translate_y;
			temp.bottom_x = rois[i].start_x / lw_ratio - translate_x;
			temp.bottom_y = rois[i].start_y / lh_ratio - translate_y;
			list.push_back(temp);
			//information << img_name << "\t" << z << "\t" << axis_x << "\t" << axis_y << "\t" << area << "\t" << ratio_ << endl;
		}
	}

	Mat Geometry::convertBinary(Mat input)
	{
		Mat target;
		equalizeHist(input, target);
		//target = modified.clone();
		//imshow("original", target);

		Mat mean_m, stddev_m;
		meanStdDev(target, mean_m, stddev_m);
		double mean = mean_m.at<double>(Point(0, 0));
		double stddev = stddev_m.at<double>(Point(0, 0));
		Mat new_target(600, 600, CV_32FC1);
		for (int i = 0; i < 600; i++)
		{
			for (int j = 0; j < 600; j++)
			{
				double value = 0.0;
				int count = 0;
				for (int k = -3; k < 4; k++)
				{
					for (int l = -3; l < 4; l++) {
						if (i + k >= 0 && i + k < 600 && j + l >= 0 && j + l < 600)
						{
							value += ((double)target.at<uchar>(Point(i + k, j + l)) - mean) / (stddev);
							count++;
						}
					}
				}
				new_target.at<float>(Point(i, j)) = (float)value / (float)count;
			}
		}
		//imshow("new", new_target);
		//normalize(new_target, target, 0, 255, NORM_MINMAX, CV_8UC1);
		//imshow("normalize", target);


		Mat bin(600, 600, CV_8UC1);
		for (int i = 0; i < 600; i++)
		{
			for (int j = 0; j < 600; j++)
			{
				if (new_target.at<float>(Point(i, j)) > -0.2)
					bin.at<uchar>(Point(i, j)) = 255;
				else
					bin.at<uchar>(Point(i, j)) = 0;
			}
		}
		return bin;
	}

	geometric Geometry::getGeometric(string image, Point leftTop, Point rightDown)
	{

		typedef struct roi
		{
			int start_x;
			int start_y;
			int end_x;
			int end_y;
		}roi;

		int ROI_num = 0;
		int beg_X, beg_Y;
		int end_X, end_Y;
		int b = 255;
		float trans_thresh[256];
		Mat img, modified, target;

		roi roi;
		geometric null;
		null.name = "-1";
		img = imread(image);
		if (!img.data) {

			return null;
		}
		int width = img.size().width;
		int height = img.size().height;

		if (width == 0 || height == 0)
			return null;
		float w_ratio = 600.0 / (float)width;
		float h_ratio = 600.0 / (float)height;



		//find corner color
		Scalar avg = 0;
		resize(img, modified, Size(), w_ratio, h_ratio, CV_INTER_NN);
		cvtColor(modified, modified, CV_BGR2GRAY);


		//roi
		ifstream fileReader;

		roi.start_x = (int)((float)leftTop.x * w_ratio);
		roi.start_y = (int)((float)leftTop.y * h_ratio);
		roi.end_x = (int)((float)rightDown.x * w_ratio);
		roi.end_y = (int)((float)rightDown.y * h_ratio);

		//cout << w_ratio << " " << h_ratio << " ";

		resize(img, img, Size(), w_ratio, h_ratio, CV_INTER_NN);

		//imshow("converted", bin);


		int point1_x = 0, point1_y = 0;
		int point2_x = 0, point2_y = 0;

		int point3_x = 300, point3_y = 0;
		int point4_x = 300, point4_y = 0;

		Mat bin = convertBinary(modified);

		vector<int>hist1 = drawing_histogram(bin(Rect(0, 0, 300, 600)), 0);
		vector<int>hist2 = drawing_histogram(bin(Rect(0, 0, 300, 600)), 1);
		vector<int>hist3 = drawing_histogram(bin(Rect(300, 0, 300, 600)), 0);
		vector<int>hist4 = drawing_histogram(bin(Rect(300, 0, 300, 600)), 1);

		vector<int>peaks;
		for (int i = 20; i < bin.rows - 15; i += 1)
		{
			int intensity = hist1[i];
			if (intensity >= hist1[i - 1] && intensity > hist1[i + -2] && intensity >= hist1[i + 1] && intensity > hist1[i + 2])
			{
				peaks.push_back(i);
			}
		}
		//cout << "hist 1:" << peaks.size() << endl;
		point1_y = peaks[0];
		point2_y = peaks[peaks.size() - 1];
		peaks.clear();
		for (int i = 2; i < bin.cols / 2 - 2; i += 1)
		{
			int intensity = hist2[i];
			if (intensity > 100 && intensity >= hist2[i - 1] && intensity > hist2[i - 2] && intensity >= hist2[i + 1] && intensity > hist2[i + 2])
			{
				peaks.push_back(i);
			}
			if (intensity == 600)
			{
				point1_x == i;
				break;
			}
		}

		point2_x = peaks[0];
		//cout << "hist 2:" << peaks.size() << endl;
		peaks.clear();
		for (int i = 20; i < bin.rows - 15; i += 1)
		{
			int intensity = hist3[i];
			if (intensity >= hist3[i - 1] && intensity > hist3[i - 2] && intensity >= hist3[i + 1] && intensity > hist3[i + 2])
			{
				peaks.push_back(i);
			}
		}
		//cout << "hist 3:" << peaks.size() << endl;

		point3_y = peaks[0];
		point4_y = peaks[peaks.size() - 1];
		peaks.clear();

		for (int i = (bin.cols) - 3; i > (bin.cols) / 2 + 2; i -= 1)
		{
			int intensity = hist4[i - 300];
			if (intensity > 100 && intensity >= hist4[i - 301] && intensity > hist4[i - 302] && intensity > hist4[i - 298] && intensity >= hist4[i - 299])
			{
				peaks.push_back(i);
			}
			if (intensity == 600)
			{
				point3_x == i;
				break;
			}
		}
		//cout << "hist 4:" << peaks.size() << endl;
		point4_x = peaks[0];
		//Mat binary, canny, final_image;


		//namedWindow("Finding Threshold", CV_WINDOW_AUTOSIZE);
		//binary = convert_to_binary_image(target, threshold_value, 255);
		Mat element = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * 5 + 1, 2 * 5 + 1),
			Point(5, 5));

		int min1_y, max2_y;
		if (point1_y < point3_y)
			min1_y = point1_y;
		else
			min1_y = point3_y;

		if (point2_y > point4_y)
			max2_y = point2_y;
		else
			max2_y = point4_y;

		int l_width = point4_x - point2_x;
		int l_height = max2_y - min1_y;

		int translate_x = 300 - ((point4_x + point2_x) / 2);
		int translate_y = 300 - ((max2_y + min1_y) / 2);
		float lw_ratio = 200.0 / (float)l_width;
		float lh_ratio = 200.0 / (float)l_height;
		if (l_width <= 0 || l_height <= 0)
			return null;
		//cvtColor(final_img, final_img, CV_GRAY2RGB);
		cvtColor(target, target, CV_GRAY2RGB);
		//rectangle(final_img, Point(point2_x + 3, min1_y), Point(point4_x, max2_y), Scalar(0, 0, 255), 5);


		roi.start_x = (int)((float)(roi.start_x + translate_x) * lw_ratio);
		roi.end_x = (int)((float)(roi.end_x + translate_x) * lw_ratio);
		roi.start_y = (int)((float)(roi.start_y + translate_y) * lh_ratio);
		roi.end_y = (int)((float)(roi.end_y + translate_y) * lh_ratio);



		Mat aligned = img(Rect(Point(point2_x + 3, min1_y), Point(point4_x, max2_y)));
		//rectangle(img, Point(point2_x + 3, min1_y), Point(point4_x, max2_y), Scalar(0, 0, 255), 5);
		Mat transform_M = (Mat_<double>(2, 3) << 1, 0, translate_x, 0, 1, translate_y);
		//Mat aligned;
		warpAffine(img, aligned, transform_M, img.size());
		//cout << lw_ratio << " " << lh_ratio << endl;
		resize(aligned, aligned, Size(), lw_ratio, lh_ratio, CV_INTER_NN);


		//rectangle(aligned, Point(rois[i].start_x, rois[i].start_y), Point(rois[i].end_x, rois[i].end_y), Scalar(0, 255, 0), 1);
		int area = (roi.end_x - roi.start_x) * (roi.end_y - roi.start_y);
		int center_x = (roi.start_x + roi.end_x) / 2;
		int center_y = (roi.start_y + roi.end_y) / 2;
		int axis_x = (aligned.size().width / 2) - center_x;
		int axis_y = (aligned.size().height / 2) - center_y;
		//double distance_ = sqrt((double)(axis_x*axis_x + axis_y*axis_y));
		//double angle_ = atan2((double)axis_x, (double)axis_y);
		double ratio_ = (double)(roi.end_x - roi.start_x) / (double)(roi.end_y - roi.start_y);
		geometric temp;
		if (area > 15000)
			area = 15000;
		if (ratio_ > 3.0)
			ratio_ = 3.0;
		temp.name = image;
		temp.label = -1;
		temp.area = area;
		temp.x = axis_x;
		temp.y = axis_y;
		temp.ratio = ratio_;
		temp.top_x = roi.end_x / lw_ratio - translate_x;
		temp.top_y = roi.end_y / lh_ratio - translate_y;
		temp.bottom_x = roi.start_x / lw_ratio - translate_x;
		temp.bottom_y = roi.start_y / lh_ratio - translate_y;
		//information << img_name << "\t" << z << "\t" << axis_x << "\t" << axis_y << "\t" << area << "\t" << ratio_ << endl;
		return temp;
	}

	geo::Geometry::Geometry()
	{
	}



	void Geometry::loadImages(char** images, char** bbox, int length)
	{
		ifstream features("features_pool.txt");
		//ifstream testfeatures("test_pool.txt");

		if (!features.is_open())
		{

			//ofstream testFeature("test_pool.txt");
			cout << "load images...." << endl;


			//test train이 있는지 먼저확인하고 아님 피쳐뽑기
			//test 셔플하기 

			for (int z = 0; z < length; z++)
			{
				/*
				vector <int> shuffler;
				vector <int> selector;

				for (int k = 0; k < numfeatures[z]; k++)
				{
					shuffler.push_back(k);
				}
				random_shuffle(shuffler.begin(), shuffler.end());
				for (int k = 0; k < numfeatures[z] / 10; k++)
				{
					selector.push_back(shuffler[k]);
				}
				sort(selector.begin(), selector.end());
				int index = 0;
				*/

				string image_dir = images[z];
				string roi_text = bbox[z];
				ifstream file_name("query.txt");
				int count = 0;
				string prev;
				while (!file_name.eof()) {
					string file;
					file_name >> file;
					vector<string> token = split(file, '_');
					file = token[0]+"_"+token[1];
					cout << file<<endl;
					if (file == prev)
						continue;
					int label = atoi(token[1].substr(0, 2).c_str());
					switch (label)
					{
					case 1:
						label = 0;
						break;
					case 3:
						label = 1;
						break;
					case 4:
						label = 2;
						break;
					case 9:
						label = 3;
						break;
					case 10:
						label = 4;
						break;
					case 11:
						label = 5;
						break;
					default:
						break;
					}
					/*
					if (count == selector[index])
					{
						build_geometry(image_dir + file + ".png", roi_text + file + ".txt", z, test);
						index++;
					}*/
					//else
					build_geometry(image_dir + file + ".png", roi_text + file + ".txt", label, extractor);
					count++;
					prev = file;
				}
				file_name.close();
				cout << "done: " << z << " number:" << count << endl;
			}
			cout << "done!!" << endl;;
			ofstream featurePool("features_pool.txt");
			for (int i = 0; i < extractor.size(); i++)
			{
				featurePool << extractor[i].name << "\t" << extractor[i].label << "\t" << extractor[i].x << "\t" << extractor[i].y << "\t" << extractor[i].top_x << "\t" << extractor[i].top_y
					<< "\t" << extractor[i].bottom_x << "\t" << extractor[i].bottom_y << "\t" << extractor[i].area << "\t" << extractor[i].ratio << endl;
			}
			/*
			for (int i = 0; i < test.size(); i++)
			{
				testFeature << test[i].name << "\t" << test[i].label << "\t" << test[i].x << "\t" << test[i].y << "\t" << test[i].top_x << "\t" << test[i].top_y << "\t" <<
					test[i].bottom_x << "\t" << test[i].bottom_y << "\t" << test[i].area << "\t" << test[i].ratio << endl;
			}
			*/
		}
		else
		{
			while (!features.eof())
			{
				geometric temp;
				features >> temp.name >> temp.label >> temp.x >> temp.y >> temp.top_x >> temp.top_y >> temp.bottom_x >> temp.bottom_y >> temp.area >> temp.ratio;
				extractor.push_back(temp);
			}
			/*
			while (!testfeatures.eof())
			{
				geometric temp;
				testfeatures >> temp.name >> temp.label >> temp.x >> temp.y >> temp.top_x >> temp.top_y >> temp.bottom_x >> temp.bottom_y >> temp.area >> temp.ratio;
				test.push_back(temp);
			}
			*/
		}

	}

	char** Geometry::get_top_ten_location(const char * test_img, int left, int right, int top, int down, int label[], int index[])
	{
		geometric test = getGeometric(test_img, Point(left,top), Point(right,down));
		//top 10
		list<score> top_ten;
		char* top_list[10];
		score temp;
		double distance = sqrt(pow((double)abs(test.x - extractor[0].x), 2) + pow((double)abs(test.y - extractor[0].y), 2));
		temp.name = extractor[0].name;
		temp.label = extractor[0].label;
		temp.distance = distance;
		temp.index = 0;
		top_ten.insert(top_ten.begin(), temp);
		for (int i = 1; i < extractor.size(); i++)
		{
			double distance = sqrt(pow((double)abs(test.x - extractor[i].x), 2) + pow((double)abs(test.y - extractor[i].y), 2));
			bool insert = false;
			for (list<score>::iterator iterPos = top_ten.begin(); iterPos != top_ten.end(); iterPos++)
			{
				if (iterPos->distance > distance)
				{
					score temp;
					temp.name = extractor[i].name;
					temp.label = extractor[i].label;
					temp.distance = distance;
					temp.index = i;
					top_ten.insert(iterPos, temp);
					insert = true;
					break;
				}
			}
			if (top_ten.size() == 51)
				top_ten.pop_back();
			if (top_ten.size() < 50 && insert == false)
			{
				score temp;
				temp.name = extractor[i].name;
				temp.label = extractor[i].label;
				temp.distance = distance;
				temp.index = i;
				top_ten.insert(top_ten.end(), temp);
			}
		}
		int i = 0;
		for (list<score>::iterator iterPos = top_ten.begin(); iterPos != top_ten.end(); iterPos++)
		{
			top_list[i] = (char *)malloc(strlen(iterPos->name.c_str())+1);
			strcpy(top_list[i], iterPos->name.c_str());
			label[i] = iterPos->label;
			index[i] = iterPos->index;
			i++;
			if (i == 10)
				break;
		}
		return top_list;
	}

	char** Geometry::get_top_ten_area(const char * test_img, int left, int right, int top, int down, int label[], int index[])
	{
		geometric test = getGeometric(test_img, Point(left, top), Point(right, down));
		//top 10
		char* top_list[10]; 
		list<score> top_ten;
		score temp;
		double distance = abs((double)(test.area - extractor[0].area) / 5.0);
		temp.name = extractor[0].name;
		temp.label = extractor[0].label;
		temp.distance = distance;
		temp.index = 0;
		top_ten.insert(top_ten.begin(), temp);
		for (int i = 1; i < extractor.size(); i++)
		{
			double distance = abs((double)(test.area - extractor[i].area) / 5.0);
			bool insert = false;
			for (list<score>::iterator iterPos = top_ten.begin(); iterPos != top_ten.end(); iterPos++)
			{
				if (iterPos->distance > distance)
				{
					score temp;
					temp.name = extractor[i].name;
					temp.label = extractor[i].label;
					temp.distance = distance;
					temp.index = i;
					top_ten.insert(iterPos, temp);
					insert = true;
					break;
				}
			}
			if (top_ten.size() == 51)
				top_ten.pop_back();
			if (top_ten.size() < 50 && insert == false)
			{
				score temp;
				temp.name = extractor[i].name;
				temp.label = extractor[i].label;
				temp.distance = distance;
				temp.index = i;
				top_ten.insert(top_ten.end(), temp);
			}
		}
		int i = 0;
		for (list<score>::iterator iterPos = top_ten.begin(); iterPos != top_ten.end(); iterPos++)
		{
			top_list[i] = (char *)malloc(strlen(iterPos->name.c_str()) + 1);
			strcpy(top_list[i], iterPos->name.c_str());
			label[i] = iterPos->label;
			index[i] = iterPos->index;
			i++;
			if (i == 10)
				break;
		}
		return top_list;
	}

	char** Geometry::get_top_ten_ratio(const char * test_img, int left, int right, int top, int down, int label[], int index[])
	{
		geometric test = getGeometric(test_img, Point(left, top), Point(right, down));
		//top 10
		char* top_list[10];
		list<score> top_ten;
		score temp;
		double distance = (double)abs((test.ratio - extractor[0].ratio)*100.0);
		temp.name = extractor[0].name;
		temp.label = extractor[0].label;
		temp.distance = distance;
		temp.index = 0;
		top_ten.insert(top_ten.begin(), temp);
		for (int i = 1; i < extractor.size(); i++)
		{
			double distance = (double)abs((test.ratio - extractor[i].ratio)*100.0);
			bool insert = false;
			for (list<score>::iterator iterPos = top_ten.begin(); iterPos != top_ten.end(); iterPos++)
			{
				if (iterPos->distance > distance)
				{
					score temp;
					temp.name = extractor[i].name;
					temp.label = extractor[i].label;
					temp.distance = distance;
					temp.index = i;
					top_ten.insert(iterPos, temp);
					insert = true;
					break;
				}
			}
			if (top_ten.size() == 51)
				top_ten.pop_back();
			if (top_ten.size() < 50 && insert == false)
			{
				score temp;
				temp.name = extractor[i].name;
				temp.label = extractor[i].label;
				temp.distance = distance;
				temp.index = i;
				top_ten.insert(top_ten.end(), temp);
			}
		}
		int i = 0;
		for (list<score>::iterator iterPos = top_ten.begin(); iterPos != top_ten.end(); iterPos++)
		{
			top_list[i] = (char *)malloc(strlen(iterPos->name.c_str()) + 1);
			strcpy(top_list[i], iterPos->name.c_str());
			label[i] = iterPos->label;
			index[i] = iterPos->index;
			i++;
			if (i == 10)
				break;
		}
		return top_list;
	}

	void Geometry::showImage(int index, int rank)
	{
		Mat image = imread(extractor[index].name);
		resize(image, image, Size(600, 600));

		rectangle(image, Point((extractor[index].top_x), (extractor[index].top_y)), Point((extractor[index].bottom_x), (extractor[index].bottom_y)), Scalar(0, 0, 255), 3);
		resize(image, image, Size(300, 300));
		imshow("top" + to_string(rank), image);
	}

	vector<std::string> Geometry:: split(const string& s, char delimiter)
	{
		vector<std::string> tokens;
		string token;
		istringstream tokenStream(s);
		while (getline(tokenStream, token, delimiter))
		{
			tokens.push_back(token);
		}
		return tokens;
	}

}


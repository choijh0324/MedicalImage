#include <iostream>
#include <fstream>
#include <list>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include "GeneralHoughTransform.hpp"

using namespace std;
using namespace cv;
typedef vector<vector<Vec2f>> RTable;
typedef struct score
{
	string name;
	int label;
	double distance;
	int index;
}score;
typedef struct Shape
{
	int label;
	string name;
	Vec2f origin;
	RTable r_table;

}Shape;
vector <string> roi_dir;
vector<string> lesion_text;
vector<Shape> test;
vector<Shape> extractor;
vector <string> lesion;
list<Vec2d> PrevPoints;
list<double> PrevDistance;
list<score> get_top_ten(RTable test, vector<Shape> pool);
list<string>get_fifty_semantic(string semantic, list<int> &label);
vector<std::string> split(const string& s, char delimiter);
void pushDistance(Vec2d prevPoint, double distance);
void clearQueue();
int haveNearest(Vec2d point,int threshold);
int main()
{
	lesion.push_back("Nodule");
	lesion.push_back("Consolidation");
	lesion.push_back("Interstitial Opacity");
	lesion.push_back("Cardiomegaly");
	lesion.push_back("Pleural Effusion");
	lesion.push_back("Pneumothorax");

	vector <int> numfeatures;
	ifstream size("size.txt");
	while (!size.eof())
	{
		int num;
		size >> num;
		numfeatures.push_back(num);
	}
	size.close();
	vector<string> lesion_image;
	Mat label(0, 2, CV_32SC1);
	Mat input_data(0, 1, CV_32SC1);

	ifstream testfeatures("features_pool.txt");
	ifstream features("test_pool.txt");


	if (!features.is_open())
	{
		ofstream featurePool("features_pool.txt");
		ofstream testFeature("test_pool.txt");
		lesion_image.push_back("D:/ChestPA_6Class/1.nodule/MASK/");
		lesion_image.push_back("D:/ChestPA_6Class/3.consolidation/MASK/");
		lesion_image.push_back("D:/ChestPA_6Class/4.interstitial_opacity/MASK/");
		lesion_image.push_back("D:/ChestPA_6Class/9.cardiomegaly/MASK/");
		lesion_image.push_back("D:/ChestPA_6Class/10.pleural_effusion/MASK/");
		lesion_image.push_back("D:/ChestPA_6Class/11.pneumothorax/MASK/");

		lesion_text.push_back("D:/ChestPA_6Class/1.nodule/TXT/");
		lesion_text.push_back("D:/ChestPA_6Class/3.consolidation/TXT/");
		lesion_text.push_back("D:/ChestPA_6Class/4.interstitial_opacity/TXT/");
		lesion_text.push_back("D:/ChestPA_6Class/9.cardiomegaly/TXT/");
		lesion_text.push_back("D:/ChestPA_6Class/10.pleural_effusion/TXT/");
		lesion_text.push_back("D:/ChestPA_6Class/11.pneumothorax/TXT/");



		//cout << "load images...." << endl;


		//test train이 있는지 먼저확인하고 아님 피쳐뽑기
		//test 셔플하기 
		for (int x = 0; x < lesion_image.size(); x++)
		{
			vector <int> shuffler;
			vector <int> selector;

			for (int k = 0; k < numfeatures[x]; k++)
			{
				shuffler.push_back(k);
			}
			random_shuffle(shuffler.begin(), shuffler.end());
			for (int k = 0; k < numfeatures[x] / 10; k++)
			{
				selector.push_back(shuffler[k]);
			}
			sort(selector.begin(), selector.end());
			int index = 0;


			string image_dir = lesion_image[x];
			ifstream file_name(image_dir + "file_list.txt");
			int count = 0;

			while (!file_name.eof()) {
				string file;
				file_name >> file;
				Mat roi = imread(image_dir + file);
				if (roi.rows == 0 || roi.cols == 0)
					continue;

				resize(roi, roi, Size(600, 600));
				GeneralHoughTransform ght(roi);
				RTable rt = ght.getRTable();
				Vec2f origin = ght.getOrigin();
				if (count == selector[index])
				{
					//test
					//test.push_back(ght.getRTable());
					testFeature << x << "\t" <<  file.substr(0, file.length() - 4) + ".png" << "\t";
					int numFeature = 0;
					for (int i = 0; i < rt.size(); i++)
					{
						numFeature += rt[i].size();
					}
					testFeature << numFeature << "\t" << origin[0] << "\t" << origin[1] << endl;
					for (int i = 0; i < rt.size(); i++)
					{
						testFeature << rt[i].size() << "\t";
						for (int j = 0; j < rt[i].size(); j++)
							testFeature << rt[i][j][0] << " " << rt[i][j][1] << "\t";
						testFeature << endl;
					}
					index++;
				}
				else
				{
					int numFeature = 0;
					//train
					//extractor.push_back(ght.getRTable());
					featurePool << x << "\t" <<  file.substr(0, file.length() - 4) + ".png" << "\t";
					for (int i = 0; i < rt.size(); i++)
					{
						numFeature += rt[i].size();
					}
					featurePool << numFeature << "\t" << origin[0] << "\t" << origin[1] << endl;
					for (int i = 0; i < rt.size(); i++)
					{
						featurePool << rt[i].size() << "\t";
						for (int j = 0; j < rt[i].size(); j++)
							featurePool << rt[i][j][0] << " " << rt[i][j][1] << "\t";
						featurePool << endl;
					}
				}
				count++;
			}
			file_name.close();

			//cout << "done: " << x << " number:" << count << endl;
		}
		testFeature.close();
		featurePool.close();
		//cout << "done!!" << endl;;
	}
	else
	{
		features.close();
		testfeatures.close();
	}
	features.open("features_pool.txt");
	testfeatures.open("test_pool.txt");
	ofstream nameRecord("image_pool.txt");
	ofstream queryRecord("query.txt"); 
	while (!features.eof())
	{
		Shape sFeature;
		RTable rTable;
		int numElement;
		features >> sFeature.label >> sFeature.name >> numElement;
		features >> sFeature.origin[0] >> sFeature.origin[1];
		queryRecord << sFeature.name << endl;
		int count = 0;
		
		for (int i = 0; i < 24; i++)
		{
			int num;
			features >> num;
			count += num;
			vector<Vec2f> temp;
			for (int i = 0; i < num; i++)
			{
				Vec2f tempV;
				features >> tempV[0] >> tempV[1];
				temp.push_back(tempV);
			}
			rTable.push_back(temp);
		}
		sFeature.r_table = rTable;
		extractor.push_back(sFeature);
	}
	features.close();


	while (!testfeatures.eof())
	{
		Shape sFeature;
		RTable rTable;
		int numElement;
		testfeatures >> sFeature.label >> sFeature.name >> numElement;
		testfeatures >> sFeature.origin[0] >> sFeature.origin[1];
		nameRecord << sFeature.name << endl;
		int count = 0;
		for (int i = 0; i < 24; i++)
		{
			int num;
			testfeatures >> num;
			count += num;
			vector<Vec2f> temp;
			for (int i = 0; i < num; i++)
			{
				Vec2f tempV;
				testfeatures >> tempV[0] >> tempV[1];
				temp.push_back(tempV);
			}
			rTable.push_back(temp);
		}
		sFeature.r_table = rTable;
		test.push_back(sFeature);
	}
	testfeatures.close();
	random_shuffle(test.begin(), test.end());
	ofstream writer("top_twenty_accuracy.txt");


	double average = 0.0;
	int count=0;
	int prev = 0;
	long long spentTime = 0;
	for (int i = 0; i < test.size()-1; i++)
	{
		//cout << test[i].name << endl;
		auto started = chrono::high_resolution_clock::now();
		vector<string> name_ = split(test[i].name, '_');
		string name = "D:/ChestPA_6Class/wholePNG/" + name_[0] + "_" + name_[1] + ".png";
		Mat test_img = imread(name);
		
		resize(test_img, test_img, Size(600, 600));
		RTable table = test[i].r_table;
		Vec2f origin = test[i].origin;
		
		for (int j = 0; j < table.size(); j++)
		{
			for (int k = 0; k < table[j].size(); k++)
			{
				circle(test_img, Point(origin[0] - table[j][k][0], origin[1] - table[j][k][1]), 1, Scalar(255, 0, 0));
			}
		}
		resize(test_img, test_img, Size(300, 300));
		cout << "test: " << lesion[test[i].label] << endl;
		Mat top[10];
		imshow("test", test_img);
		list <score> top_ten = get_top_ten(test[i].r_table, extractor);
		list <int>labels;
		list<string> semantic_fifty = get_fifty_semantic(test[i].name, labels);
		int k = 0;
		int total_score = 0;
		/*
		if (prev != test[i].label)
		{
			spentTime /= (long long)count;
			average /= (double)count;
			writer << " label : " << prev << " number of sample:" << count << " average: " << average << " spent time: " << spentTime <<" milli second"<< endl;
			average = 0.0;
			count = 0;
			spentTime = 0;
			prev++;
		}*/
		/*
		list<score>resorted_list;
		for (list<score>::iterator iterPos = top_ten.begin(); iterPos != top_ten.end(); iterPos++)
		{
			string name = iterPos->name;
			int count = 0;
			list<int>::iterator iterPos3 = labels.begin();
			for (list<string>::iterator iterPos2 = semantic_fifty.begin(); iterPos2 != semantic_fifty.end();iterPos2++,iterPos3++)
			{
				if (iterPos->name == *iterPos2)
				{
					resorted_list.push_back(*iterPos);
					semantic_fifty.erase(iterPos2);
					labels.erase(iterPos3);
					break;
				}
				if (resorted_list.size() == 20)
					break;
				count++;
			}
		}
		for (int j = resorted_list.size(); j < 20; j++)
		{
			score temp;
			temp.index = -1;
			temp.distance = -1;
			temp.label = *labels.begin();
			temp.name = *semantic_fifty.begin();
			labels.pop_front();
			semantic_fifty.pop_front();
			resorted_list.push_back(temp);
		}
		if (test[i].name == "091000044_04InterstitialOpacity_0.png")
		{
			ofstream list_("modified.csv");
			for (list<score>::iterator iterPos = resorted_list.begin(); iterPos != resorted_list.end(); iterPos++) 
			{
				list_ << iterPos->name << "\n";
			}
			list_.close();
		}*/
		//for (list<score>::iterator iterPos = resorted_list.begin(); iterPos != resorted_list.end(); iterPos++)
		for (list<score>::iterator iterPos = top_ten.begin(); iterPos != top_ten.end(); iterPos++)
		{
			
			if (test[i].label == iterPos->label)
				total_score++;
			
			
			if (k < 10) {
				vector<string> name_ = split(iterPos->name, '_');
				string name = "D:/ChestPA_6Class/wholePNG/" + name_[0] + "_" + name_[1] + ".png";
				top[k] = imread(name);
				resize(top[k], top[k], Size(600, 600));
				int index = iterPos->index;
				RTable table = extractor[index].r_table;
				Vec2f extractOrigin = extractor[index].origin;
				for (int j = 0; j < table.size(); j++)
				{
					for (int l = 0; l < table[j].size(); l++)
					{
						circle(top[k], Point(extractOrigin[0] - table[j][l][0], extractOrigin[1] - table[j][l][1]), 1, Scalar(255, 0, 0));
					}
				}
				resize(top[k], top[k], Size(300, 300));
				imshow("top" + to_string(k + 1), top[k]);
				k++;
				cout << "top" << k << ": " << lesion[iterPos->label] << " ";
			}
			

		}
		//cout << endl;
		//cout << "score : " << total_score << "/10 " << endl;
		average += (double)total_score;
		count++;
		char ch = waitKey();
		if (ch == 27) break;
		auto done = std::chrono::high_resolution_clock::now();

		spentTime += std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
	}
	average /= (double)count;
	spentTime /= (long long)count;
	writer << " label : " << prev << " number of sample:" << count << " average: " << average << " spent time: " << spentTime << " milli second" << endl;
	return 0;
}
list<string>get_fifty_semantic(string semantic, list<int> &labels)
{
	ifstream test("semantic/"+semantic+".csv");
	string line; 
	list<string>images;
	for (int i = 0; i < 50; i++)
	{
		getline(test, line);
		vector<string> token = split(line, ',');
		string name = token[1];
		int label = atoi(token[2].c_str());
		images.push_back(name);
		labels.push_back(label);
	}
	return images;
}

list<score> get_top_ten(RTable test, vector<Shape> pool)
{
	list<score> top_ten;
	score temp;
	temp.name = "";
	temp.label = -1;
	temp.index = -1;
	temp.distance = INFINITY;
	top_ten.insert(top_ten.begin(), temp);
	for (int i = 0; i < pool.size(); i++)
	{
		double distance = 0.0;
		//double min = INFINITY;
		RTable table = pool[i].r_table;
		RTable *longer, *smaller;
		bool insert = false;
		for (int j = 0; j < table.size(); j++)
		{
			double sum = 0.0;
			if (table[j].size() > test[j].size())
				(longer = &table, smaller = &test);
			else
				(longer = &test, smaller = &table);

			//&& prevX - (threshold * 5) <= (*smaller)[j][k][0]
			int threshold = (*smaller)[j].size() / 20;
			for (int k = 0; k < (*smaller)[j].size(); k++)
			{
				int index = haveNearest((*smaller)[j][k], threshold);
				if (index>=0)
				{
					list<double>::iterator it = PrevDistance.begin();
					advance(it, index); 
					double distance = (*it);
					sum += distance;
					continue;
				}
				
				double min = INFINITY;
				for (int l = 0; l < (*longer)[j].size(); l++)
				{
					double distance = sqrt(pow((*longer)[j][l][0] - (*smaller)[j][k][0], 2) + pow((*longer)[j][l][1] - (*smaller)[j][k][1], 2));
					if (min > distance)
						min = distance;
				}
				pushDistance((*smaller)[j][k], min);
				sum += min;
			}

			//clearQueue();
			distance += sum + 200.0 * ((*longer)[j].size() - (*smaller)[j].size());

		}


		for (list<score>::iterator iterPos = top_ten.begin(); iterPos != top_ten.end(); iterPos++)
		{
			if (iterPos->distance > distance)
			{
				score temp;
				temp.name = pool[i].name;
				temp.label = pool[i].label;
				temp.distance = distance;
				temp.index = i;
				top_ten.insert(iterPos, temp);
				insert = true;
				break;
			}
		}
		if (top_ten.size() == 11)
			top_ten.pop_back();
		if (top_ten.size() < 10 && insert == false)
		{
			score temp;
			temp.name = pool[i].name;
			temp.label = pool[i].label;
			temp.distance = distance;
			temp.index = i;
			top_ten.insert(top_ten.end(), temp);
		}
		clearQueue();
	}
	return top_ten;
}

void pushDistance(Vec2d prevPoint, double distance)
{
	PrevPoints.push_front(prevPoint);
	PrevDistance.push_front(distance);
	if (PrevPoints.size() == 8)
	{
		PrevPoints.pop_back();
		PrevDistance.pop_back();
	}
}

void clearQueue()
{
	PrevPoints.clear();
	PrevDistance.clear();
}

int haveNearest(Vec2d point, int threshold)
{
	//prevY - threshold <= (*smaller)[j][k][1] && prevX - threshold < (*smaller)[j][k][0]
	int count = 0;
	for (list<Vec2d>::iterator iterPos = PrevPoints.begin(); iterPos != PrevPoints.end(); iterPos++)
	{
		if ((*iterPos)[0] - (threshold) <= point[0] && (*iterPos)[1] - threshold <= point[1])
			return count;
		count++;
	}
	return -1;
}

vector<std::string> split(const string& s, char delimiter)
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
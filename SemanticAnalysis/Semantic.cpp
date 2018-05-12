#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
int hamming_distance(unsigned long x, unsigned long y);
vector<std::string> split(const string& s, char delimiter);
int modify(string & s);
int main()
{
	ifstream test("091000044_04InterstitialOpacity_0.png.csv");
	string line;
	string location = "D:/ChestPA_6Class/wholePNG/";
	string bbox = "D:/ChestPA_6Class/wholeTEXT/";
	Mat testImage;
	testImage = imread(location + "091000044_04InterstitialOpacity.png");
	int top, bottom, left, right;
	ifstream box(bbox + "091000044_04InterstitialOpacity.txt");
	int number;
	box >> number >> left >> top >> right >> bottom;
	rectangle(testImage, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 5);
	resize(testImage, testImage, Size(300, 300));
	imshow("test", testImage);
	Mat tops[10];
	for (int i = 0; i < 10; i++)
	{
		getline(test, line);
		vector<string> token = split(line, ',');
		string name = token[1];
		int label = atoi(token[2].c_str());
		int num = modify(name);
		tops[i] = imread(location + name);
		string file= bbox + name.substr(0, name.size() - 3) + "txt";
		ifstream box(file);
		box >> number;
		for(int j=0;j<=num;j++)
			box >> left >> top >> right >> bottom;
		rectangle(tops[i], Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 5);
		resize(tops[i], tops[i], Size(300, 300));
		imshow("top "+ to_string(i+1), tops[i]); 
		cout << name << " " << label << endl;
	}
	char ch = waitKey();
	return 0;
}

int hamming_distance(unsigned long x, unsigned long y)
{
	int dist = 0;
	unsigned  val = x ^ y;

	// Count the number of bits set
	while (val != 0)
	{
		// A bit is set, so increment the count and clear the bit
		dist++;
		val &= val - 1;
	}

	// Return the number of differing bits
	return dist;
}
int modify(string & s)
{
	int count = 0;
	int number = 0;
	int i = 0;
	for (; i < s.size(); i++)
	{
		if (s[i] == '_')
			count++;
		if (count == 2)
		{
			string label_ = "";
			for (int j = i; j < s.size(); j++)
			{
				if (s[j] == '.')
				{
					number = atoi(label_.c_str());
					break;
				}
				label_ += s[j];
			}
			break;
		}
	}
	s = s.substr(0, i) + ".png";
	return number;
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
//ghc genotype.c -c && ghc -c -O2 Safe.hs && ghc --make -no-hs-main -optc-O -I/usr/local/Cellar/opencv/2.4.10.1/include -L/usr/local/Cellar/opencv/2.4.10.1/lib -lopencv_objdetect -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -L/usr/lib -lc++ -lc++abi -lm -lc -I/usr/include -lopencv_photo -lopencv_contrib -I/Library/Frameworks/Python.framework/Versions/2.7/include/ -lpython2.7 -L/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config -I/Applications/ghc-7.10.1.app/Contents/lib/ghc-7.10.1/include stable_version_gesture_recognition.cpp Safe -c && ghc --make -no-hs-main stable_version_gesture_recognition.o genotype.o Safe -o hand -I/usr/local/Cellar/opencv/2.4.10.1/include -L/usr/local/Cellar/opencv/2.4.10.1/lib -lopencv_objdetect -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc -L/usr/lib -lc++ -lc++abi -lm -lc -I/usr/include -lopencv_photo -lopencv_contrib -I/Library/Frameworks/Python.framework/Versions/2.7/include/ -lpython2.7 -L/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config -I/Applications/ghc-7.10.1.app/Contents/lib/ghc-7.10.1/include -lopencv_flann -lopencv_ml


//ghc genotype.c -c
//ghc -c -O2 Safe.hs
//ghc --make -no-hs-main -optc-O -L/usr/lib -lc++ -lc++abi -lm -lc -I/usr/include stable_version_gesture_recognition.cpp Safe -c -lm -lstdc++ `pkg-config opencv --cflags --libs`
//ghc --make -no-hs-main stable_version_gesture_recognition.o genotype.o Safe -o hand -L/usr/lib -I/usr/include -lstdc++ `pkg-config opencv --cflags --libs`

//ghc genotype.c -c && ghc -c -O2 Safe.hs && ghc --make -no-hs-main -optc-O -L/usr/lib -lc++ -lc++abi -lm -lc -I/usr/include stable_version_gesture_recognition.cpp Safe -c -lm -lstdc++ `pkg-config opencv --cflags --libs` && ghc --make -no-hs-main stable_version_gesture_recognition.o genotype.o Safe -o hand -L/usr/lib -I/usr/include -lstdc++ `pkg-config opencv --cflags --libs` -lpython2.7

#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring> 
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#include <HsFFI.h>
#ifdef __GLASGOW_HASKELL__
#include "Safe_stub.h"
#include <limits>
extern "C" {
	void __stginit_Safe(void);
}
#endif

extern "C" {
#include "graph.h"
#include "genotype.h"
}

#if (defined _WIN32 || defined __WIN64)
    #include <windows.h>
#elif defined __APPLE__    
    #include <spawn.h>
    #include <sys/types.h>
    #include <unistd.h>
#else
	#include <spawn.h>
#endif

		#include <stdio.h>  /* defines FILENAME_MAX */
		#ifdef WINDOWS
		    #include <direct.h>
		    #define GetCurrentDir _getcwd
		#else
		    #include <unistd.h>
		    #define GetCurrentDir getcwd
		 #endif





#include <sys/stat.h>

// various paths based on OS
#if (defined _WIN32 || defined __WIN64)
    static char *GENOTYPE_FILE_NAME = "genes";

    static char *GENOTYPE_FITNESS_FILE_NAME = "genes_fitness";

    static char *BEST_GENOTYPE_FILE_NAME = "best_solution";

    static char *PYTHON_PATH = "C:\\Windows\\py.exe ";

    static char *DETECTOR_FILENAME = "detector";

    static char *NN_FILENAME = "classifier\\net";
#elif defined __APPLE__
    static char *GENOTYPE_FILE_NAME = "genes";

    static char *GENOTYPE_FITNESS_FILE_NAME = "genes_fitness";

    static char *BEST_GENOTYPE_FILE_NAME = "best_solution";

    static char *DETECTOR_FILENAME = "detector";

    static char *NN_FILENAME = "classifier/net";
#else
    static char *GENOTYPE_FILE_NAME = "genes";

    static char *GENOTYPE_FITNESS_FILE_NAME = "genes_fitness";

    static char *BEST_GENOTYPE_FILE_NAME = "best_solution";

    static char *DETECTOR_FILENAME = "detector";

    static char *NN_FILENAME = "classifier/net";
#endif

//For peas
extern char **environ;

// detector evolution script, peas framework
#if (defined _WIN32 || defined __WIN64)
    const char *python_prog = "nice peas\\test\\nao_feature_detector_gestures";
#elif defined __APPLE__
    const char *python_prog = "nice peas/test/nao_feature_detector_gestures";
#else
    const char *python_prog = "nice peas/test/nao_feature_detector_gestures";
#endif

using namespace std;
using namespace cv;

namespace patch
{
	template < typename T > std::string to_string( const T& n )
	{
		std::ostringstream stm;
		stm << n;
		return stm.str();
	}
}


struct myclass {
    bool operator() (cv::Point pt1, cv::Point pt2) { return (pt1.x < pt2.x);}
} myobject;



    Rect main_selected_head; 
    Rect main_selected_body; 
    Rect prev_head;
    Rect prev_body;


cv::Point face_median;



int point_distance(cv::Point p0, cv::Point p1){
    return sqrt((p0.x - p1.x)*(p0.x - p1.x) + (p0.y - p1.y)*(p0.y - p1.y));
}


bool rect_contains(Rect rect, double x, double y)
{
	double pointX = x;
	double pointY = y;
        // Just had to change around the math
	if (pointX < (rect.x + (.5*rect.width)) && pointX > (rect.x - (.5*rect.width)) &&
           pointY < (rect.y + (.5*rect.height)) && pointY > (rect.y - (.5*rect.height)))
		return true;
	else
		return false;
}

bool point_in_rect(Rect rect, double x, double y)
{
	double pointX = x;
	double pointY = y;
        // Just had to change around the math
	if (pointX < (rect.x + (.5*rect.width)) && pointX > (rect.x - (.5*rect.width)) &&
           pointY < (rect.y + (.5*rect.height)) && pointY > (rect.y - (.5*rect.height)))
		return true;
	else
		return false;
}


// Returns whether a point lies on the contour
int lies_on_contour(vector<vector<Point> > contours, Point point)
{
	for(size_t n = 0; n < contours.size(); ++n)
	{
	    if(contours[n].end() != find(contours[n].begin(), contours[n].end(), point))
			return 1;
	}	
	
	return 0;
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<std::pair<cv::Mat,int> > loadBinary(const std::string &datapath, const std::string &labelpath){
    std::vector<std::pair<cv::Mat,int> > dataset;
    std::ifstream datas(datapath.c_str(),std::ios::binary);
    std::ifstream labels(labelpath.c_str(),std::ios::binary);

    if (!datas.is_open() || !labels.is_open())
        cout << "binary files could not be loaded" << endl;
        //throw std::runtime_error("binary files could not be loaded");

    int magic_number=0; int number_of_images=0;int r; int c;
    int n_rows=0; int n_cols=0; unsigned char temp=0;

    // parse data header
    datas.read((char*)&magic_number,sizeof(magic_number));
    magic_number=reverseInt(magic_number);
    datas.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images=reverseInt(number_of_images);
    datas.read((char*)&n_rows,sizeof(n_rows));
    n_rows=reverseInt(n_rows);
    datas.read((char*)&n_cols,sizeof(n_cols));
    n_cols=reverseInt(n_cols);

    // parse label header - ignore
    int dummy;
    labels.read((char*)&dummy,sizeof(dummy));
    labels.read((char*)&dummy,sizeof(dummy));

    for(int i=0;i<number_of_images;++i){
        cv::Mat img(n_rows,n_cols,CV_32FC1);

        for(r=0;r<n_rows;++r){
            for(c=0;c<n_cols;++c){
                datas.read((char*)&temp,sizeof(temp));
                img.at<float>(r,c) = ((float)temp); // 0.255 values
            }
        }
        labels.read((char*)&temp,sizeof(temp));
        dataset.push_back(std::make_pair(img,(int)temp));
    }
    return dataset;
}

//create the cascade classifier object used for the face detection
CascadeClassifier face_cascade;
CascadeClassifier upper_body_cascade;

//setup image files used in the capture process
Mat grayscaleFrame;

int EXTRACT_FEATURES = 17; // # of joint angles
int EXTRACT_ROWS = 40; // # of time instances
int GESTURES = 6;
int PEOPLE = 6;
int DETECTORS = 50;
int GENERATIONS = 100000;
	
// structure to hold detected limbs
typedef struct Limbs
{
	cv::Point  start;
	cv::Point  end;
	RotatedRect limb_bounding_rectangle;
	cv::Point break_point;
	vector<Point> details;
} Limb;

// A utility function to print the adjacenncy list representation of graph
void printGraph(struct Graph* graph)
{
    int v;
    for (v = 0; v < graph->V; ++v)
    {
		if(graph->array[v].head == NULL) continue;
        struct AdjListNode* pCrawl = graph->array[v].head;
        printf("\n Adjacency list of vertex %d\n head ", v);
        while (pCrawl)
        {
            printf("-> %d (%s) (@ %d, %d) ", pCrawl->dest, pCrawl->weight, pCrawl->position.x, pCrawl->position.y);
            pCrawl = pCrawl->next;
        }
        printf("\n");
    }
}
		

// Returns the square of the euclidean distance between 2 points.
double dist(Point x,Point y)
{
	return (x.x-y.x)*(x.x-y.x)+(x.y-y.y)*(x.y-y.y);
}

cv::Mat getHeatMap(cv::Mat input) // input is of type CV_8UC1, return is of type CV_8UC3
{
    cv::Mat result(input.rows, input.cols, CV_8UC3);
    for (int yy = 0; yy < input.rows; ++yy)
    {
        for (int xx = 0; xx < input.cols; ++xx)
        {
            int pixelValue = input.at<uchar>(yy, xx);
            if (pixelValue < 128) {
                result.at<cv::Vec3b>(yy, xx) = cv::Vec3b(0, 0 + 2*pixelValue, 255 - 2 * pixelValue);
            } else {
                result.at<cv::Vec3b>(yy, xx) = cv::Vec3b(0 + 2*pixelValue, 255 - 2 * pixelValue, 0);
            }
        }
    }
    return result;
}

void save_heatmap(Mat mat, const char* str){ //CV_32F
	//mat = mat.mul(100000.0);
	//cv::Mat matrix_norm = cv::Mat::zeros ( mat.cols, mat.rows, CV_32F );
	//cv::normalize ( mat, matrix_norm, 0, 255, NORM_MINMAX, CV_32F, Mat() );
	//cout << mat << endl;
	double min;
	double max;
	cv::minMaxIdx(mat, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(mat, adjMap, 255 / max);
	
	imwrite( str, adjMap );
}

// Checks whether a file is empty
int isEmpty(FILE *file)
{
    long savedOffset = ftell(file);
    fseek(file, 0, SEEK_END);
    
    if (ftell(file) == 0)
    {
        return 1;
    }
    
    fseek(file, savedOffset, SEEK_SET);
    return 0;
}

// checks whether file exists
int file_exist(char *filename)
{
    struct stat   buffer;
    return (stat (filename, &buffer) == 0);
}


//This function returns the radius and the center of the circle given 3 points
//If a circle cannot be formed , it returns a zero radius circle centered at (0,0)
pair<Point,double> circleFromPoints(Point p1, Point p2, Point p3)
{
	double offset = pow(p2.x,2) +pow(p2.y,2);
	double bc =   ( pow(p1.x,2) + pow(p1.y,2) - offset )/2.0;
	double cd =   (offset - pow(p3.x, 2) - pow(p3.y, 2))/2.0;
	double det =  (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x)* (p1.y - p2.y); 
	double TOL = 0.0000001;
	if (abs(det) < TOL) { return make_pair(Point(0,0),0); }

	double idet = 1/det;
	double centerx =  (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * idet;
	double centery =  (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * idet;
	double radius = sqrt( pow(p2.x - centerx,2) + pow(p2.y-centery,2));

	return make_pair(Point(centerx,centery),radius);
}

// Load face classification data
void initFaces(){
	face_cascade.load("head_cascade_11.xml");
	upper_body_cascade.load("body_cascade_4.xml");
}

//function detects faces on a frame and returns a vector with rectangles that correspond to the detected faces
Rect detectFaces(Mat frame){
	//convert captured image to gray scale and equalize
	cvtColor(frame, grayscaleFrame, CV_BGR2GRAY);
	//equalizeHist(grayscaleFrame, grayscaleFrame);
        std::vector<Rect> all_faces;
        for (int j=1;j<=10;j++){
            for (int i=0;i<=10;i++){
	        std::vector<Rect> faces;
	        face_cascade.detectMultiScale(grayscaleFrame, faces, 1+(0.1*j), i, CV_HAAR_DO_ROUGH_SEARCH|CV_HAAR_SCALE_IMAGE, Size(10,10));

                all_faces.reserve(all_faces.size() + distance(faces.begin(),faces.end()));
                all_faces.insert(all_faces.end(),faces.begin(),faces.end());
            }
        }

int face_border_x = 80;
int face_border_y = 20;
int face_border_x1 = 230;
int face_border_y1 = 120;

    Rect main_selected_head11; 
    main_selected_head11.x = face_border_x;
    main_selected_head11.y = face_border_y;
    main_selected_head11.width = face_border_x1;
    main_selected_head11.height = face_border_y1;

rectangle(frame, main_selected_head11, Scalar(0,0,255));


        std::vector<Rect> selected_head;

        for(int i = 0; i < all_faces.size(); i++){
            if(((all_faces[i] & main_selected_head11).area() == all_faces[i].width*all_faces[i].height)){
                    selected_head.push_back(all_faces[i]);
                    //rectangle(frame, all_faces[i], Scalar(0,0,0));
            }
        }


	//cout << "faces: " << selected_head.size() << endl;


    std::vector<Point> circle_coordinates;
    for(int i = 0; i<selected_head.size(); i++){
        Mat roi = grayscaleFrame.colRange(selected_head[i].x,selected_head[i].x+selected_head[i].width).rowRange(selected_head[i].y,selected_head[i].y+selected_head[i].height);
        vector<Vec3f> circles;
        HoughCircles(roi, circles, CV_HOUGH_GRADIENT, 1, selected_head[i].width/5, 100, 10, 0, 4 );
        for( size_t j = 0; j < circles.size(); j++ ) {
            Point center(cvRound(circles[j][0])+selected_head[i].x, cvRound(circles[j][1])+selected_head[i].y);
            circle_coordinates.push_back(center);
            //cout << "circle at: " << cvRound(circles[j][0])+selected_head[i].x << " and " << cvRound(circles[j][1])+selected_head[i].y << endl;
            circle(frame,center,circles[j][2],Scalar(0,0,255),0);
        }
    }


face_median.x=-1;
face_median.y=-1;
if(circle_coordinates.size() > 0){
std::sort(circle_coordinates.begin(), circle_coordinates.end(), myobject);
face_median = circle_coordinates[(int)(circle_coordinates.size()/2)];
face_median.y=0;
circle(frame,face_median,10,Scalar(0,0,255),0);
}

if(main_selected_head.x != 999 && main_selected_head.y != 999)
	prev_head = main_selected_head;
//main_selected_head.x = 999;
//main_selected_head.y = 999;
//main_selected_head.width = 999;
//main_selected_head.height = 999;

int found_max = -1;

    for(int i = 0; i < selected_head.size(); i++){
        int found = -1;
        for(int j = 0; j < circle_coordinates.size(); j++){
cout << "is " << circle_coordinates[j].x << ", " << circle_coordinates[j].y << " inside: " << selected_head[i] << endl;
            if(circle_coordinates[j].inside(selected_head[i])){
                found++;
            }
        }
cout << "found: " << found << endl;
            if((found > found_max)/* && ((selected_head[i].width*selected_head[i].height) < (main_selected_head.width*main_selected_head.height))*/){
                found_max = found;
                found = -1;
                        main_selected_head.x = selected_head[i].x;
                        main_selected_head.y = selected_head[i].y;
                        main_selected_head.width = selected_head[i].width;
                        main_selected_head.height = selected_head[i].height;
            }
    }

    //rectangle(frame, main_selected_head, Scalar(0,0,255));

    //cout << main_selected_head.x << " " << main_selected_head.y << endl;


    Rect face(-1,-1,-1,-1);
    if(main_selected_head.x < 999 and main_selected_head.y < 999 and main_selected_head.width < 999 and main_selected_head.height < 999){
        int center_x = main_selected_head.x + 0.5 * main_selected_head.width;
        int center_y = main_selected_head.y + 0.5 * main_selected_head.height;
        int center_x1 = prev_head.x + 0.5 * prev_head.width;
        int center_y1 = prev_head.y + 0.5 * prev_head.height;
        cv::Point A(center_x, center_y);
        cv::Point B(center_x1, center_y1);
        if(point_distance(A,B) > 10){
            if(main_selected_head.x != 999 and main_selected_head.y != 999 and main_selected_head.width != 999 and main_selected_head.height != 999 and
               prev_head.x != 999 and prev_head.y != 999 and prev_head.width != 999 and prev_head.height != 999){
                main_selected_head = prev_head;
            }
        }
        if(main_selected_head.x != 999 and main_selected_head.y != 999 and main_selected_head.width != 999 and main_selected_head.height != 999){
            //Rect face;
            face.x = main_selected_head.x;
           face.y = main_selected_head.y;
            face.width = main_selected_head.width;
            face.height = main_selected_head.height;
            //rectangle(frame, face, Scalar(0,0,0));
        }
    }

if(face.x == -1 && face.y == -1){
//cout << "face none" << endl;
face = prev_head;
//cout << "face " << face << endl;
}


//rectangle(frame, face, Scalar(0,0,0));
//cout << face.x << " " << face.y << " " << face.width << " " << face.height << endl;

    return face;
}

// function detects upper bodies on a frame and recurns a vector with rectangles that correspond to the detected upper bodies
Rect detectUpperBodies(Mat frame){
	//convert captured image to gray scale and equalize
	cvtColor(frame, grayscaleFrame, CV_BGR2GRAY);
	//equalizeHist(grayscaleFrame, grayscaleFrame);

        std::vector<Rect> all_bodies;
        for (int j=1;j<=10;j++){
            for (int i=0;i<=10;i++){
	        std::vector<Rect> bodies;
	        upper_body_cascade.detectMultiScale(grayscaleFrame, bodies, 1+(0.1*j), i, CV_HAAR_DO_ROUGH_SEARCH|CV_HAAR_SCALE_IMAGE, Size(10,10));

                all_bodies.reserve(all_bodies.size() + distance(bodies.begin(),bodies.end()));
                all_bodies.insert(all_bodies.end(),bodies.begin(),bodies.end());
            }
        }
	
int body_border_x = 60;
int body_border_y = 120;
int body_border_x1 = 250;
int body_border_y1 = 100;

    Rect main_selected_body11; 
    main_selected_body11.x = body_border_x;
    main_selected_body11.y = body_border_y;
    main_selected_body11.width = body_border_x1;
    main_selected_body11.height = body_border_y1;

rectangle(frame, main_selected_body11, Scalar(0,0,255));

        std::vector<Rect> selected_body;

        for(int i = 0; i < all_bodies.size(); i++){
            if(((all_bodies[i] & main_selected_body11).area() == all_bodies[i].width*all_bodies[i].height)){
                    selected_body.push_back(all_bodies[i]);
                    //rectangle(frame, all_bodies[i], Scalar(0,0,0));
            }
        }///finding all rectangles within a big ROI


    std::vector<Point> circle_coordinates;
    for(int i = 0; i<selected_body.size(); i++){
        Mat roi = grayscaleFrame.colRange(selected_body[i].x,selected_body[i].x+selected_body[i].width).rowRange(selected_body[i].y,selected_body[i].y+selected_body[i].height);
        vector<Vec3f> circles;
        HoughCircles(roi, circles, CV_HOUGH_GRADIENT, 1, selected_body[i].width/5, 200, 10, 0, 10 );
        for( size_t j = 0; j < circles.size(); j++ ) {
            if(face_median.x != -1 && face_median.y != -1 && abs(face_median.x - (cvRound(circles[j][0])+selected_body[i].x+selected_body[i].width/2)) < 10){
                Point center(cvRound(circles[j][0])+selected_body[i].x, cvRound(circles[j][1])+selected_body[i].y);
                circle_coordinates.push_back(center);
                //cout << "circle at: " << cvRound(circles[j][0])+selected_body[i].x << " and " << cvRound(circles[j][1])+selected_body[i].y << endl;
                circle(frame,center,circles[j][2],Scalar(0,0,255),0);
            }
        }
    }///has to be right, because it is showing circles in right places

cout << "before" << endl;
cout << prev_body.x << " " << prev_body.y << " " << prev_body.width << " " << prev_body.height << endl;
cout << main_selected_body.x << " " << main_selected_body.y << " " << main_selected_body.width << " " << main_selected_body.height << endl;

if(main_selected_body.x != 999 && main_selected_body.y != 999 && main_selected_body.x != -1 && main_selected_body.y != -1){
//cout << "prev_body: " << prev_body.x << " " << prev_body.y << endl;
	prev_body = main_selected_body;
//cout << "prev_body: " << prev_body.x << " " << prev_body.y << endl;
}

//main_selected_body.x = 999;
//main_selected_body.y = 999;
//main_selected_body.width = 999;
//main_selected_body.height = 999;

int found_max = -1;


    for(int i = 0; i < selected_body.size(); i++){
        int found = -1;
        //find how many circles are in rectangle
        for(int j = 0; j < circle_coordinates.size(); j++){
            if(circle_coordinates[j].inside(selected_body[i])){
                found++;
            }
        }
        if((found > found_max)){
               	found_max = found;
                main_selected_body.x = selected_body[i].x;
                main_selected_body.y = selected_body[i].y;
                main_selected_body.width = selected_body[i].width;
                main_selected_body.height = selected_body[i].height;
         }
         found = -1;
    }


    rectangle(frame, main_selected_body, Scalar(0,0,255));

    //cout << main_selected_body.x << " " << main_selected_body.y << endl;


    Rect body(-1,-1,-1,-1);
    if(main_selected_body.x < 999 && main_selected_body.y < 999 && main_selected_body.width < 999 && main_selected_body.height < 999){
        int center_x = main_selected_body.x + 0.5 * main_selected_body.width;
        int center_y = main_selected_body.y + 0.5 * main_selected_body.height;
        int center_x1 = prev_body.x + 0.5 * prev_body.width;
        int center_y1 = prev_body.y + 0.5 * prev_body.height;
        cv::Point A(center_x, center_y);
        cv::Point B(center_x1, center_y1);
        if(point_distance(A,B) > 10){
            if(main_selected_body.x != 999 && main_selected_body.y != 999 && main_selected_body.width != 999 && main_selected_body.height != 999 &&
               prev_body.x != 999 && prev_body.y != 999 && prev_body.width != 999 && prev_body.height != 999){
                //main_selected_body = prev_body;
            }
        }
        if(main_selected_body.x != 999 && main_selected_body.y != 999 && main_selected_body.width != 999 && main_selected_body.height != 999){
            //Rect body;
            body.x = main_selected_body.x;
           body.y = main_selected_body.y;
            body.width = main_selected_body.width;
            body.height = main_selected_body.height;
            //rectangle(frame, body, Scalar(0,0,0));
        }
    }
 
if(body.x == -1 && body.y == -1 && main_selected_body.x != 999 && main_selected_body.y != 999){
body = main_selected_body;
}


//rectangle(frame, body, Scalar(0,0,0));
cout << "after" << endl;
cout << body.x << " " << body.y << " " << body.width << " " << body.height << endl;
cout << main_selected_body.x << " " << main_selected_body.y << " " << main_selected_body.width << " " << main_selected_body.height << endl;


	//cout << "bodies: " << selected_body.size() << endl;
	return body;
}

// function takes in two rectangles (a and b) and returns a rectangle from the intersection between rectangles a and b
CvRect rect_intersect(CvRect a, CvRect b) 
{ 
    CvRect r; 
    r.x = (a.x > b.x) ? a.x : b.x;
    r.y = (a.y > b.y) ? a.y : b.y;
    r.width = (a.x + a.width < b.x + b.width) ? 
        a.x + a.width - r.x : b.x + b.width - r.x; 
    r.height = (a.y + a.height < b.y + b.height) ? 
        a.y + a.height - r.y : b.y + b.height - r.y; 
    if(r.width <= 0 || r.height <= 0) 
        r = cvRect(0, 0, 0, 0); 

    return r; 
}

// function reduces illumination by normalising the intensity values of the frame brightness (Y - lumma)
void reduceIllumination(Mat frame){
	cv::cvtColor(frame, frame, CV_BGR2YUV);
	std::vector<cv::Mat> channels;
	cv::split(frame, channels);
	cv::equalizeHist(channels[0], channels[0]);
	cv::merge(channels, frame);
	cv::cvtColor(frame, frame, CV_YUV2BGR);
}

// Adds an edge to an undirected graph
void addEdge(int src_id, int dst_id, struct Graph* graph, int src, cv::Point src_point, int dest, cv::Point dest_point, int weight)
{
    // Add an edge from src to dest.  A new node is added to the adjacency
    // list of src.  The node is added at the begining
    struct AdjListNode* newNode = newAdjListNode(src_id, dest_point, dest, weight);
    newNode->next = graph->array[src].head;
    graph->array[src].head = newNode;
 
    // Since graph is undirected, add an edge from dest to src also
    newNode = newAdjListNode(dst_id, src_point, src, weight);
    newNode->next = graph->array[dest].head;
    graph->array[dest].head = newNode;

    //keeps number of vertices added to the graph
	graph->vertices_added++;
}

int is_connected(struct Graph* graph, int node_i, int node_j){
	//iterate both strings of feature graphs
    	int v;
	//iterate both feature graphs adjacency lists
    	for (v = 0; v < graph->V; ++v)
    	{
    		struct AdjListNode* pCrawl = graph->array[v].head;
    		while (pCrawl)
    		{
			if((v == node_i && pCrawl->dest == node_j) || (v == node_j && pCrawl->dest == node_i)) return 1;
			pCrawl = pCrawl->next;
		}
	}
	return 0;
}


/* This procedure captures stream/pre-recorded video from caera/file, extracts features and creates a graph for every frame
   Then graphs are concattenated together to produce a string of feature graphs.
   Function returns a number of frames in a pre-recorded video
*/




/** * Determines the angle of a straight line drawn between point one and two.
 The number returned, which is a float in degrees, 
 tells us how much we have to rotate a horizontal line clockwise for it to match the line between the two points. * 
 If you prefer to deal with angles using radians instead of degrees, 
 just change the last line to: "return atan2(yDiff, xDiff);" */ 
float GetAngleOfLineBetweenTwoPoints(cv::Point p1, cv::Point p2) 
{ 
	float xDiff = p2.x - p1.x;
	float yDiff = p2.y - p1.y; 
	
	return atan2(yDiff, xDiff) * (180 / M_PI); 
}

double bitstring_to_double(const char* p)
{
    unsigned long long x = 0;
    for (; *p; ++p)
    {
        x = (x << 1) + (*p - '0');
    }
    double d;
    memcpy(&d, &x, 8);
    return d;
}


Mat match_strings(vector<Graph> test, vector<Graph> query, int max_length){

vector <string> added;

int graph_c = -1;


cout << test.size()*5 << " " << test.size()*5 << endl;

	Mat M = Mat(test.size()*5,test.size()*5, CV_32F, cvScalar(0.));

	//iterate both strings of feature graphs
	for(std::vector<Graph>::iterator test_it = test.begin(); test_it != test.end(); ++test_it) {
                (&(*test_it))->id=0;
		for(std::vector<Graph>::iterator query_it = query.begin(); query_it != query.end(); ++query_it) {
                        (&(*query_it))->id=1;
			if((&(*test_it))->vertices_added == (&(*query_it))->vertices_added){
		    	int v;
				

				//iterate both feature graphs adjacency lists
		    	for (v = 0; v < (&(*test_it))->V; ++v)
		    	{
				graph_c++;
				int v2;
				for (v2 = 0; v2 < (&(*query_it))->V; ++v2)
				{

		    	    		struct AdjListNode* pCrawl = (&(*test_it))->array[v].head;
		    	    		while (pCrawl)
		    	    		{
					        struct AdjListNode* pCrawl2 = (&(*query_it))->array[v2].head;
					        while (pCrawl2)
					        {
								// DIAGONAL - similarity between feature end-point distances
								if((pCrawl->dest) == (pCrawl2->dest) && (v==v2)){
									float dn = sqrt(dist(pCrawl->position, pCrawl2->position))/max_length;
//cout << "accessing: " << graph_c << " " << graph_c << endl;
//									// maximum allowed deviation is 1% of the frame diagonal
//									if(dn <= max_length*0.01/max_length){
//										if(max_length > 0.0)
//											M.at<float>(graph_c, graph_c) = max_length*0.01/max_length - dn;
//										else
//											M.at<float>(graph_c, graph_c) = 0.0;
//									}else{
//										M.at<float>(graph_c, graph_c) = 0.0;
//									}
								}else{

//if((&(*test_it))->sfg_index == (&(*query_it))->sfg_index && (&(*test_it))->id == (&(*query_it))->id)
//cout << "angle between " << get_name(pCrawl->position) << "( " << (&(*test_it))->id << " | " << (&(*test_it))->sfg_index << " ) and " << get_name(pCrawl2->position) << "( " << (&(*query_it))->id << " | " << (&(*query_it))->sfg_index << " )" << endl;

float de1 = 0.0;
float de2 = 0.0;

for(std::vector<Graph>::iterator test_it_dest = test.begin(); test_it_dest != test.end(); ++test_it_dest) {
	int v_dest;
    for (v_dest = 0; v_dest < (&(*test_it_dest))->V; ++v_dest)
    {
        struct AdjListNode* pCrawl_test_dest = (&(*test_it_dest))->array[v_dest].head;
        while (pCrawl_test_dest)
        {
			if((pCrawl_test_dest->dest != pCrawl->dest) && ((&(*test_it))->id == (&(*test_it_dest))->id) && ((&(*test_it))->sfg_index == (&(*test_it_dest))->sfg_index)){

				de1 = GetAngleOfLineBetweenTwoPoints(pCrawl->position, pCrawl_test_dest->position) / 360.0;


					/// compare to edges of the graphs in the query
					for(std::vector<Graph>::iterator query_it_dest = query.begin(); query_it_dest != query.end(); ++query_it_dest) {
						int v2_dest;
					    for (v2_dest = 0; v2_dest < (&(*query_it_dest))->V; ++v2_dest)
					    {
						struct AdjListNode* pCrawl_query_dest = (&(*query_it_dest))->array[v2_dest].head;
						while (pCrawl_query_dest)
						{

std::stringstream compared;
compared << (&(*test_it))->id << (&(*test_it))->sfg_index << (&(*test_it_dest))->id << (&(*test_it_dest))->sfg_index << (&(*query_it))->id << (&(*query_it))->sfg_index << (&(*query_it_dest))->id << (&(*query_it_dest))->sfg_index << pCrawl_query_dest->dest << pCrawl2->dest;
//cout << compared.str() << endl;

int found = 0;
//cout << "is connected: " << 
if (is_connected((&(*test_it)), pCrawl->dest, pCrawl_test_dest->dest) && is_connected((&(*query_it)), pCrawl2->dest, pCrawl_query_dest->dest)) found++;


								if((pCrawl_query_dest->dest != pCrawl2->dest) && ((&(*query_it_dest))->id == (&(*query_it))->id) && ((&(*query_it_dest))->sfg_index == (&(*query_it))->sfg_index) /*&& (std::find(added.begin(), added.end(), compared.str()) == added.end())*/ && (found) /*(is there a connection between 'pCrawl->position' and 'pCrawl2->position')*/){
added.push_back(compared.str());

//				cout << "angle between " << get_name(pCrawl->position) << "(" << (&(*test_it))->id << " | " << (&(*test_it))->sfg_index << ")" << " and " << get_name(pCrawl_test_dest->position) << "(" << (&(*test_it_dest))->id << " | " << (&(*test_it_dest))->sfg_index << ")"  << endl;
//									cout << "compare to angle between " << get_name(pCrawl2->position) << "(" << (&(*query_it))->id << " | " << (&(*query_it))->sfg_index << ")" << " and " << get_name(pCrawl_query_dest->position) << "(" << (&(*query_it_dest))->id << " | " << (&(*query_it_dest))->sfg_index << ")"  << endl;
									de2 = GetAngleOfLineBetweenTwoPoints(pCrawl2->position, pCrawl_query_dest->position) / 360.0;
								}
								pCrawl_query_dest = pCrawl_query_dest->next; 
							}
						}
					}




			}
			pCrawl_test_dest = pCrawl_test_dest->next;
		}
	}
}




float de_diff = abs(de1-de2);

// 3,6% maximum deviation of inclination between line edge connecting two feature end-points 
//M.at<float>(std::distance(test.begin(), test_it)*6+pCrawl->dest, std::distance(query.begin(), query_it)*6+pCrawl2->dest) = 1.0;
//cout << "update " << std::distance(test.begin(), test_it)*4+pCrawl->dest << " and " << std::distance(query.begin(), query_it)*4+pCrawl2->dest << endl;

cout << "accessing2: " << std::distance(test.begin(), test_it)*5+pCrawl->dest << " " << std::distance(query.begin(), query_it)*5+pCrawl2->dest << endl;

if(de_diff/360.0 <= 360.0*0.01/360.0){
	M.at<float>(std::distance(test.begin(), test_it)*5+pCrawl->dest, std::distance(query.begin(), query_it)*5+pCrawl2->dest) = 360.0*0.01/360.0 - de_diff/360.0;
	//M.at<float>(std::distance(query.begin(), query_it)*4+pCrawl2->dest, std::distance(test.begin(), test_it)*4+pCrawl->dest) = 360.0*0.01/360.0 - de_diff/360.0; //symmetry
}else{
	M.at<float>(std::distance(test.begin(), test_it)*5+pCrawl->dest, std::distance(query.begin(),query_it)*5+pCrawl2->dest) = 0.0;
	//M.at<float>(std::distance(query.begin(),query_it)*4+pCrawl2->dest, std::distance(test.begin(),test_it)*4+pCrawl->dest) = 0.0;
}
								}
					
					            pCrawl2 = pCrawl2->next;
					        }
							pCrawl = pCrawl->next;
					    }
		    	    }
		    	}
			}
		}
	}
	
	return M;
}


void my_sleep(unsigned msec) {
    struct timespec req, rem;
    int err;
    req.tv_sec = msec / 1000;
    req.tv_nsec = (msec % 1000) * 1000000;
    while ((req.tv_sec != 0) || (req.tv_nsec != 0)) {
        if (nanosleep(&req, &rem) == 0)
            break;
        err = 999;
        // Interrupted; continue
        if (err == 999) {
            req.tv_sec = rem.tv_sec;
            req.tv_nsec = rem.tv_nsec;
        }
        // Unhandleable error (EFAULT (bad pointer), EINVAL (bad timeval in tv_nsec), or ENOSYS (function not supported))
        break;
    }
}

// function reads genotype from fifo-queue. Currently function wastes CPU by continuously trying to read the file 
// After genotype is read, fifo-queue is removed
void read_ann(Genotype *genotype, char *NEW_GENOTYPE_FILE_NAME, char *NEW_GENOTYPE_FITNESS_FILE_NAME, int with_removal, int node_types_specified, int bypass_lines) {	
//printf("\n %s \n", NEW_GENOTYPE_FILE_NAME);

    int times = 0;
    int redo = 1;
    while(redo){
//cout << "1" << endl;
printf("\n %s \n", NEW_GENOTYPE_FILE_NAME);
        FILE *fp = fopen(NEW_GENOTYPE_FILE_NAME, "rb");
        if ( fp == NULL ){
//cout << "2" << endl;
		if(fp != NULL && ftell(fp) >= 0){
            		fclose(fp);
//cout << "3" << endl;
		}
//cout << "4" << endl;
            sleep(1);
            continue;

            read_ann(genotype, NEW_GENOTYPE_FILE_NAME, NEW_GENOTYPE_FITNESS_FILE_NAME, with_removal, node_types_specified, bypass_lines);
        }
        while ( !feof (fp) )
        {
//cout << "5" << endl;

int lines_counter = 0;
//cout << "5.1" << endl;
std::ifstream inFile(NEW_GENOTYPE_FILE_NAME); 
//cout << "5.2" << endl;
std::string unused;
//cout << "5.3" << endl;
while ( std::getline(inFile, unused) ){
//cout << "5.4" << endl;
   ++lines_counter;
//cout << "6" << endl;
}
//cout << "6.1" << endl;
inFile.close();
//cout << "6.2" << endl;
if(!bypass_lines)
if (lines_counter < 2){ /*cout << "7" << endl;*/ sleep(5); continue; }

//cout << "7.5" << endl;
            redo = 0;
            if(isEmpty(fp) && (fp != NULL && ftell(fp) >= 0)){
//cout << "8" << endl;
            fclose(fp);
//cout << "8.1" << endl;
                continue;
            }else{
//cout << "9" << endl;
                times++;
                
                if(times <= 100){
//cout << "10" << endl;
                    int result = genotype_fread(genotype, fp, node_types_specified);
//cout << "10.1" << endl;
			if(fp != NULL && ftell(fp) >= 0){ //cout << "11" << endl;
                    		fclose(fp);}
                    //removing file
//cout << "11.1" << endl;
					if(with_removal)
                    	remove(NEW_GENOTYPE_FILE_NAME);

//cout << "12" << endl;
                    break;
                }else{
//cout << "13" << endl;
                    break;
                }
            }
            
        }
    }
    
    if((with_removal && file_exist(NEW_GENOTYPE_FILE_NAME)) || (with_removal && file_exist(NEW_GENOTYPE_FITNESS_FILE_NAME))){
///cout << "14" << endl;
        read_ann(genotype, NEW_GENOTYPE_FILE_NAME, NEW_GENOTYPE_FITNESS_FILE_NAME, with_removal, node_types_specified, bypass_lines);
    }
}

// function converts float to double, leaving 7 decimal digits
double round_to_decimal(float f) {
    char buf[42];
    sprintf(buf, "%.7g", f);
    return atof(buf);
}

bool reallyIsNan(float x)
{
    //Assumes sizeof(float) == sizeof(int)
    int intIzedX = *(reinterpret_cast<int *>(&x));
    int clearAllNonNanBits = intIzedX & 0x7F800000;
    return clearAllNonNanBits == 0x7F800000;
}

template <bool> struct static_assert;
template <> struct static_assert<true> { };

template<typename T>
inline bool is_NaN(T const& x) {
    static_cast<void>(sizeof(static_assert<std::numeric_limits<T>::has_quiet_NaN>));
    return std::numeric_limits<T>::has_quiet_NaN and (x != x);
}

template <typename T>
inline bool is_inf(T const& x) {
    static_cast<void>(sizeof(static_assert<std::numeric_limits<T>::has_infinity>));
    return x == std::numeric_limits<T>::infinity() or x == -std::numeric_limits<T>::infinity();
}


// funtion feeds input to a neural network by using hnn (Haskell) package
double feed(Genotype *genotype, Mat matrix, int type, int the_class = -1, cv::Mat confusion_matrix = cv::Mat::zeros(GESTURES,GESTURES, CV_32F)) {
	//reconstruct matrices from available data to a suitable format

//    for (int i=0; i<genotype_get_size(); i++)
//printf("lets see %f \n", (*genotype)->genes[i]);
//print_genotype(&(*genotype));   NOT

	double* matrixzvector = (double *)malloc(sizeof(double)*(matrix.rows * matrix.cols));
	int mm=0;
	for (int nr=0; nr<matrix.rows; nr++){ 
	    for (int nc=0; nc<matrix.cols; nc++){ 
	        matrixzvector[mm]=round_to_decimal(matrix.at<float>(nr,nc));
	        mm=mm+1;
	    }
	}

	double* node_types = (double *)malloc(sizeof(double)*(int)sqrt(genotype_get_size()));
	double* flat_matrix = (double *)malloc(sizeof(double)*(genotype_get_size()));

    for (int i=0; i<(int)sqrt(genotype_get_size()); i++){
        node_types[i] = (*genotype)->node_types[0];
//printf(" %f ", (*genotype)->node_types[i]);
    }
//printf("\n");

    for (int i=0; i<genotype_get_size(); i++){
        flat_matrix[i] = (*genotype)->genes[i];
//printf(" %f ", flat_matrix[i]);
    }
//printf("\n");

	//result array
	double *res;
	
	int asize = genotype_get_size();
	int bsize = matrix.rows * matrix.cols;
	int csize = (int)sqrt(genotype_get_size());
	
	/*
		type - feedworward (0) or recurrent (1)
		asize - number of weights
		bsize - number of inputs
		csize - number of nodes
		flat_matrix - weights
		matrixzvector - inputs
		node_types - node types
		res - result array
	*/
	process_network_input(&type, &asize, &bsize, &csize, &(*flat_matrix), &(*matrixzvector), &(*node_types), &res);
	
	free(matrixzvector);
	free(node_types);
	free(flat_matrix);
	double result;
	if(type == 0){
		result = *(res+((int)*res));  //the last element of the result array from recurrent 
	}else{
		result = 0;
		double val = *(res+1);
//cout << val;
if(!isnan(val))
confusion_matrix.at<float>((int)the_class, 0) += val;

		for(int i=2;i<=(int)*res;i++){
//cout << setw(20) << *(res+i);
if(!isnan(*(res+i)))
confusion_matrix.at<float>((int)the_class, i-1) += *(res+i);

			if(val < *(res+i)){
				result = i-1;
				val = *(res+i);
			}
//cout << setw(20) << val;
		}
	}
	

//cout << endl;


//cout << confusion_matrix << endl;
	return result;
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

int main(int argc, char *argv[])
{	


initFaces();

    main_selected_head.x = 999;
    main_selected_head.y = 999;
    main_selected_head.width = 999;
    main_selected_head.height = 999;

	prev_head.x = 999;
	prev_head.y = 999;
	prev_head.width = 999;
	prev_head.height = 999;

    main_selected_body.x = 999;
    main_selected_body.y = 999;
    main_selected_body.width = 999;
    main_selected_body.height = 999;

	prev_body.x = 999;
	prev_body.y = 999;
	prev_body.width = 999;
	prev_body.height = 999;

face_median.x = -1;
face_median.y = -1;

int generation_counter = 1;

	//initialisation of haskell insterface
	int i;
	    hs_init(&argc, &argv);
	#ifdef __GLASGOW_HASKELL__
	    hs_add_root(__stginit_Safe);
	#endif
	
	
	int a=1;
//	printf("1 for training detectors, 2 for getting output from evolved detectors, 3 for training classifying ANN, 4 for testing evolved detectors together with trained classifier, 9 for data preparation: ");
	//scanf("%d", &a);

	
	//BOF EVOLVING DETECTORS
	if(a == 1){

		
		char cCurrentPath[FILENAME_MAX];

		 if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath)))
		     {
		     return 999;
		     }

		cCurrentPath[sizeof(cCurrentPath) - 1] = '\0'; /* not really required */

		printf ("The current working directory is %s", cCurrentPath);
		
		//bof start hyperneat algorithms
		//int generations = 50000;   //ignore - had been used to make sure that peas and gesture recognition algorithm will end simultaneously
		//int detectors = 2; //specify number of detectors that want to be evolved (NOTE: in current version, the same amount of script files must exist, numbered 1..N)
		for(int detector_id=1;detector_id<=DETECTORS;detector_id++){
			
			//bof specify names of files
			char id[1*sizeof(int)+1000];
			sprintf(id, "_%d", 1);//detector_id);

			char counter[1*sizeof(int)+1000];
			sprintf(counter, "_%d", generation_counter);//detector_id);
			
			char * NEW_GENOTYPE_FILE_NAME = new char[std::strlen(GENOTYPE_FILE_NAME)+std::strlen(id)+16];
		    std::strcpy(NEW_GENOTYPE_FILE_NAME,GENOTYPE_FILE_NAME);
		    std::strcat(NEW_GENOTYPE_FILE_NAME,id);
                    std::strcat(NEW_GENOTYPE_FILE_NAME,"gen");
                    std::strcat(NEW_GENOTYPE_FILE_NAME,counter);

			char * NEW_GENOTYPE_FITNESS_FILE_NAME = new char[std::strlen(GENOTYPE_FITNESS_FILE_NAME)+std::strlen(id)+16];
		    std::strcpy(NEW_GENOTYPE_FITNESS_FILE_NAME,GENOTYPE_FITNESS_FILE_NAME);
		    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,id);
                    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,"gen");
                    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,counter);
			//eof specify names
			
					cout << "looking for: " << NEW_GENOTYPE_FITNESS_FILE_NAME << endl;

std::string genotype_file_names = patch::to_string(detector_id);
			
			// specify name of the detector's evolutionary script
			int parentID = getpid();  // should child kill when evolution is done?
	    	char str[1*sizeof(double)];
	    	sprintf(str, "%d", parentID);
	    	char* name_with_extension = (char*)malloc(2+strlen(python_prog)+3*sizeof(char)+1*sizeof(int)+1000+sizeof(GENOTYPE_FILE_NAME)+6);
	    	strcpy(name_with_extension, python_prog);
	    	//strcat(name_with_extension, id);	
	    	strcat(name_with_extension, ".py");
	    	strcat(name_with_extension, " ");
	    	strcat(name_with_extension, str);
	    	strcat(name_with_extension, " ");
	    	strcat(name_with_extension, genotype_file_names.c_str());
	    	strcat(name_with_extension, " ");
	    	strcat(name_with_extension, GENOTYPE_FILE_NAME);
        	        	
	    	pid_t pid;
	    	char *argvv[] = {"sh", "-c", name_with_extension, NULL};
	    	int status;
        	
			//different process spawning methods used on different platforms
	    	#if (defined _WIN32 || defined __WIN64)
        	
	    	    STARTUPINFO si;
	    	    PROCESS_INFORMATION pi;
        	
	    	    ZeroMemory( &si, sizeof(si) );
	    	    si.cb = sizeof(si);
	    	    ZeroMemory( &pi, sizeof(pi) );
        	
	    	    char str2[strlen(PYTHON_PATH)+strlen(name_with_extension)];
	    	    strcpy(str2, PYTHON_PATH);
	    	    strcat(str2, name_with_extension);uestionnaires or n
        	
	    	    //0 instead of CREATE_NEW_CONSOLE shows outoput in parent console (i.e. Webots if started from Webots environment)
	    	    if (!CreateProcess(NULL, str2, NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &si, &pi))
	    	    {
	    	        printf( "CreateProcess failed (%d).\n", GetLastError() );
	    	    }
        	
	    	    CloseHandle(pi.hThread);
        	
	    	#elif defined __APPLE__
	    	    status = posix_spawn(&pid, "/bin/sh", NULL, NULL, argvv, environ);
	    	#else
status = posix_spawn(&pid, "/bin/sh", NULL, NULL, argvv, environ);
        	//cout << "I am here 99 " << endl;
	    	#endif
	
			free(name_with_extension);
		}
		//eof start hyperneat algorithms

        		//cout << "I am here 2" << endl;

		// for every generation, use every detector on an a set of images
		for(int j=0;j<GENERATIONS;j++){			
			        		//cout << "I am here 3" << endl;
			// 10 detectors, 4 gestures
			Mat detectors_outputs = Mat(DETECTORS, 2, CV_32F, cvScalar(0.));
			
			// STEP_1: read ANN, create affinity matrix, feed affinity matrix to ANN and collect outputs for every detector in a matrix (rows - detectors; columns - gesture classes)
        		//cout << "I am here 4" << endl;
			for(int detector_id=1;detector_id<=DETECTORS;detector_id++){
				        		//cout << "I am here 5" << endl;
				char id[1*sizeof(int)+1000];
				sprintf(id, "_%d", detector_id);
				
			char counter[1*sizeof(int)+1000];
			sprintf(counter, "_%d", generation_counter);
			
			char * NEW_GENOTYPE_FILE_NAME = new char[std::strlen(GENOTYPE_FILE_NAME)+std::strlen(id)+30];
		    std::strcpy(NEW_GENOTYPE_FILE_NAME,GENOTYPE_FILE_NAME);
		    std::strcat(NEW_GENOTYPE_FILE_NAME,id);
                    std::strcat(NEW_GENOTYPE_FILE_NAME,"gen");
                    std::strcat(NEW_GENOTYPE_FILE_NAME,counter);
        	
				char * NEW_GENOTYPE_FITNESS_FILE_NAME = new char[std::strlen(GENOTYPE_FITNESS_FILE_NAME)+std::strlen(id)+30];
			    std::strcpy(NEW_GENOTYPE_FITNESS_FILE_NAME,GENOTYPE_FITNESS_FILE_NAME);
			    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,id);
                    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,"gen");
                    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,counter);
				//eof specify names
				
					cout << "looking for: " << NEW_GENOTYPE_FITNESS_FILE_NAME << endl;
				
				//genotype size 442x442 maximum connections - for the recurrent network (although it can still act as a feedforward network)
				genotype_set_size(101*101);
        		Genotype ann = genotype_create();
//cout << "I am here 6" << endl;

				//read ANN that had been generated from peas framework
				read_ann(&ann, NEW_GENOTYPE_FILE_NAME, NEW_GENOTYPE_FITNESS_FILE_NAME, 1, 1, 0);
        		//cout << "I am here 1" << endl;
				for(int sfg=1;sfg<=30;sfg++){ //7
					//for(int person=1;person<=PEOPLE;person++){
//if(person == 3) continue;


						
						string currentPath(cCurrentPath);
						std::string folder_name = currentPath+"/A0"+patch::to_string(sfg)+"/S0"+patch::to_string(0)+"/";
						
						vector<string> files = vector<string>();
					    //getdir(folder_name,files,true);
						//cout << "files in " << folder_name << " : " << files.size() << endl;
						
						//for(int trial=1; trial <= 20; trial++){
						
							int mode = 0; // 0 - pre-recorder videos; 1 - camera video stream
							vector<Graph> string_feature_graph_demo;  // string of feature graphs (string=vector | feature graphs=adjacentcy-list graph structure)
							float diagonal_size = 0.0;
							
							//in order to skip viewing pre-recorded videos every time, their corresponding affinity matrix is stored in a separate file
							std::stringstream ss;
							std::stringstream ss2;
							std::stringstream ss3;
if(sfg < 10)
ss << "00";
if(sfg>=10 && sfg<100)
ss << "0";
							ss << sfg;
							//ss2 << person;
							//ss3 << trial;
							std::string full_filename;
							string filename = "data/test_data/stored/";
							full_filename.append(filename);
							full_filename.append(ss.str());
							//full_filename.append("_");
							//full_filename.append(ss2.str());
							//full_filename.append("_");
							//full_filename.append(ss3.str());
							full_filename.append(".txt");
cout << "looking for: "<<full_filename<<endl;
							Mat resized_M;
							//cout << "file name " << full_filename << endl;
							
							//if saved affinity matrix exists, read it instead of viewing pre-recorded videos
							if(!std::ifstream(full_filename.c_str())){
cout << "here" << endl;
								int frames = 0;//start_capture(mode, string_feature_graph_demo, -1, diagonal_size, 80, sfg);
								//printf("finished\n");
	                    		//cout << "here2" << endl;
								//make affinity matrix
								Mat M = match_strings(string_feature_graph_demo, string_feature_graph_demo, diagonal_size);
								//cout << M.rows << " | " << M.cols << endl;
	                    		//cout << "here3" << endl;
								//resize the affinity matrix to match 21x21 detector's input
								Size size(10,10);
								resized_M = Mat(10,10, CV_32F, cvScalar(0.));
								resize(M,resized_M,size);
								
								FileStorage fs(full_filename, FileStorage::WRITE);
								fs << "mat" << resized_M;
								fs.release();
								string filename2 = "data/test_data/stored/saved_heatmap_sfg_";
								std::string full_filename2;
								full_filename2.append(filename);
								full_filename2.append(ss.str());
								full_filename2.append("_");
								full_filename2.append(ss2.str());
								full_filename2.append("_");
								full_filename2.append(ss3.str());
								full_filename2.append(".png");
								save_heatmap(resized_M, full_filename2.c_str());
							}else{
cout << "here2" << endl;
								FileStorage fs(full_filename, FileStorage::READ);
								fs["mat"] >> resized_M;
								fs.release();

Size size(10,10);
								if (resized_M.cols != 0)
									resize(resized_M,resized_M,size);
							}

if (resized_M.cols != 0){
normalize(resized_M, resized_M, 0.0, 1.0, NORM_MINMAX, -1);

for(int row=0;row<resized_M.rows;row++)
for(int column=0;column<resized_M.cols;column++)
if(isnan(resized_M.at<float>(row, column)))
resized_M.at<float>(row, column) = 0.0;

cout << "here3" << endl;
	                    	
							//feed affinity matrix into HyperNEAT generated ANN
							int type=0;
							double ann_output = feed(&ann, resized_M, type);
cout << "here4" << endl;
							//cout << "ANN output: " << ann_output << endl;
							// round the output. NOTE: 0.5 means that the output neuron did not fire (sigmoid(0)==0.5)
							// there is something wrong with using simply 0, therefore value 0.001 is used instead
							//cout << "gesture " << sfg << " person " << person << " detector " << detector_id << endl;
//if (sfg > 0) return 0;
                        	//if(!isnan(ann_output)){
				//cout << "not nan" << endl;			detectors_outputs.at<float>(detector_id-1, floor((sfg-1)/5)) += max(0.000001, round(abs(ann_output)));
                                //}else{
				//cout << "nan" << endl;			detectors_outputs.at<float>(detector_id-1, floor((sfg-1)/5)) += 0.000001;
				//}				
                        	if(!isnan(ann_output))
							detectors_outputs.at<float>(detector_id-1, floor((sfg-1)/15)) += max(0.000001, round(abs(ann_output)));
                                else
							detectors_outputs.at<float>(detector_id-1, floor((sfg-1)/15)) += 0.000001;
							//cout << "detectors: " << endl << detectors_outputs << endl;

}else{
	detectors_outputs.at<float>(detector_id-1, floor((sfg-1)/15)) += 0.000001;
}

						//}
					//}
				}
			}
			
			//STEP_2: calculate fitness function
			for(int detector_id=1;detector_id<=DETECTORS;detector_id++){
				
				//bof specify names
				char id[1*sizeof(int)+1000];
				sprintf(id, "_%d", detector_id);
				
			char counter[1*sizeof(int)+1000];
			sprintf(counter, "_%d", generation_counter);//detector_id);
			
			char * NEW_GENOTYPE_FILE_NAME = new char[std::strlen(GENOTYPE_FILE_NAME)+std::strlen(id)+16];
		    std::strcpy(NEW_GENOTYPE_FILE_NAME,GENOTYPE_FILE_NAME);
		    std::strcat(NEW_GENOTYPE_FILE_NAME,id);
                    std::strcat(NEW_GENOTYPE_FILE_NAME,"gen");
                    std::strcat(NEW_GENOTYPE_FILE_NAME,counter);
        	
				char * NEW_GENOTYPE_FITNESS_FILE_NAME = new char[std::strlen(GENOTYPE_FITNESS_FILE_NAME)+std::strlen(id)+16];
			    std::strcpy(NEW_GENOTYPE_FITNESS_FILE_NAME,GENOTYPE_FITNESS_FILE_NAME);
			    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,id);
                    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,"gen");
                    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,counter);

				//eof specify names
				
				// matrix to hold all rounded outputs from detectors_x_gestures matrix (apart from the currently reviewed detector).
				// this matrix will be used to calculate nearest neighbours Manhattan distance from this detector to all others. 
				CvMat* pointsForSearch = cvCreateMat(DETECTORS-1, 2, CV_32F);
				float* pointsForSearch_ = (float *)malloc((sizeof(float)*DETECTORS*2)-(2*sizeof(float)));
				int deviation = 0;
				for(int detector_id_query=1;detector_id_query<=DETECTORS;detector_id_query++){
					if(detector_id_query != detector_id){
						pointsForSearch_[(detector_id_query-1-deviation)*2+0] = detectors_outputs.at<float>(detector_id_query-1,0);
						pointsForSearch_[(detector_id_query-1-deviation)*2+1] = detectors_outputs.at<float>(detector_id_query-1,1);
					}else{
						deviation++;
					}
				}

				cvSetData(pointsForSearch, pointsForSearch_, pointsForSearch->step); ///detectorN -> detector1(row1): gesture1(col1); detector2(row2): gesture2(col2); ...
				
				//calculate manhattan distance to every other detector and accumulate Manhattan distances to all detectors into a single variable
				double total_distance = 0.0;
				for(int detector_id_query2=1;detector_id_query2<=DETECTORS-1;detector_id_query2++){
					CvMat* a_row = cvCreateMat(1, 2, CV_32FC1);

					cvSet2D(a_row, 0, 0, cvGet2D(pointsForSearch, detector_id_query2-1, 0));
					cvSet2D(a_row, 0, 1, cvGet2D(pointsForSearch, detector_id_query2-1, 1));
					
					CvMat* a_detector = cvCreateMat(1, 2, CV_32FC1);
					float a_detector_data[2] = {detectors_outputs.at<float>(detector_id-1,0), detectors_outputs.at<float>(detector_id-1,1)};
					cvSetData(a_detector, a_detector_data, a_detector->step);
					
					double detectors_manhattan_distance = cvCalcEMD2(a_row, a_detector, CV_DIST_L1, 0, 0, 0, 0, 0) / (float)DETECTORS;
					//printf("manhattan distance from this detector to others %f\n", detectors_manhattan_distance);
					
					total_distance += detectors_manhattan_distance;
					
					cvReleaseMat(&a_row);
					cvReleaseMat(&a_detector);
				}
				
				// take average Manhattan distance
				double fitness = total_distance/(float)DETECTORS;
				//cout << "fitness: " << fitness << endl;
				
				//write fitness to fifo-queue
				char output[50];
	    		snprintf(output,50,"%f",fitness);
        		
	    		FILE *file;
	    		file = fopen(NEW_GENOTYPE_FITNESS_FILE_NAME,"w");
	    		fprintf(file, output);
	    		fclose(file);
	
				free(pointsForSearch_);
				cvReleaseMat(&pointsForSearch);
			}

generation_counter += 1;
	//EOF EVOLVING DETECTORS
	

		}
	//EOF EVOLVING DETECTORS
	
	//BOF COLLECTING EVOLVED DETECTORS OUTPUTS FROM A SET OF TRAINING VIDEOS
	}else if(a == 2){
		
		//holds detectors genotypes
		vector<Genotype> detectors;
		
		//for every stored training video
		//for(int sfg=0;sfg<GESTURES;sfg++){
		//	for(int person=0;person<PEOPLE;person++){
                  for(int sfg=1;sfg<=720;sfg++){
				//if(person == 3 || person == 4 || person == 1) continue;
				//collect detectors outputs
				vector<double> detector_output_per_video;
				//for(int trial=0; trial < 20; trial++){	
				//for 6 evolved detectors (should be 10, but not always all 10 detectors are evolved - some may not reach minimum fitness)
				for(int i=1; i<=DETECTORS; i++){
					char id[1*sizeof(int)+1000];
					sprintf(id, "_%d", i);
            	
					char * NEW_DETECTOR_FILE_NAME = new char[std::strlen(DETECTOR_FILENAME)+std::strlen(id)+1];
				    std::strcpy(NEW_DETECTOR_FILE_NAME,DETECTOR_FILENAME);
				    std::strcat(NEW_DETECTOR_FILE_NAME,id);
            	
					genotype_set_size(101*101);
					Genotype ann = genotype_create();
            	
            	
					char * NEW_GENOTYPE_FITNESS_FILE_NAME = new char[std::strlen(GENOTYPE_FITNESS_FILE_NAME)+std::strlen(id)+1];
				    std::strcpy(NEW_GENOTYPE_FITNESS_FILE_NAME,GENOTYPE_FITNESS_FILE_NAME);
				    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,id);
            	
					//for every detector, read its ANN from file
					read_ann(&ann, NEW_DETECTOR_FILE_NAME, NEW_GENOTYPE_FITNESS_FILE_NAME, 0, 1, 1);
					
					int mode = 0; //using pre-recorded videos
					
					//string of feature graphs
					vector<Graph> string_feature_graph_demo;
					
					float diagonal_size = 0.0;
					
							//in order to skip viewing pre-recorded videos every time, their corresponding affinity matrix is stored in a separate file
							std::stringstream ss;
							std::stringstream ss2;
							std::stringstream ss3;

if(sfg < 10)
ss << "00";
if(sfg>=10 && sfg<100)
ss << "0";

							ss << sfg;
							//ss2 << person;
							//ss3 << trial;

							std::string full_filename;
							string filename = "data/test_data/stored/";
							full_filename.append(filename);
							full_filename.append(ss.str());
							//full_filename.append("_");
							//full_filename.append(ss2.str());
							//full_filename.append("_");
							//full_filename.append(ss3.str());
							full_filename.append(".txt");
							Mat resized_M;
							//cout << "file name " << full_filename << endl;
							
					if(!std::ifstream(full_filename.c_str())){
						int frames = 0;//start_capture(mode, string_feature_graph_demo, -1/*-1*/, diagonal_size, 80, sfg);
            	    	//printf("finished\n");
						//make affinity matrix
						Mat M = match_strings(string_feature_graph_demo, string_feature_graph_demo, diagonal_size);
            	
						Size size(10,10);
						resized_M = Mat(10,10, CV_32F, cvScalar(0.));
cout << "here-2" << endl;
						resize(M,resized_M,size, 0,0, INTER_AREA);
cout << "here-1" << endl;
						
						FileStorage fs(full_filename, FileStorage::WRITE);
cout << "here0" << endl;
						fs << "mat" << resized_M;

cout << "here" << endl;
						fs.release();
cout << "here2" << endl;


								string filename3 = "data/test_data/stored/saved_heatmap_sfg_";
								std::string full_filename3;
								full_filename3.append(filename);
								full_filename3.append(ss.str());
								full_filename3.append("_");
								full_filename3.append(ss2.str());
								full_filename3.append("_");
								full_filename3.append(ss3.str());
								full_filename3.append(".png");
								save_heatmap(resized_M, full_filename3.c_str());
					}else{
						FileStorage fs(full_filename, FileStorage::READ);
						fs["mat"] >> resized_M;
						fs.release();

Size size(10,10);

								if (resized_M.cols != 0)
									resize(resized_M,resized_M,size);
					}
            	
								if (resized_M.cols != 0){
					//collect detectors outputs
					int type=0;
					double ann_output = feed(&ann, resized_M, type);
					//cout << ann_output << endl;
					detector_output_per_video.push_back(ann_output);
}else{
cout << "smth wrong" << endl;
}
				}
				


//std::cout << "Numeric.LinearAlgebra.fromList [";
for (std::vector<double>::const_iterator i = detector_output_per_video.begin(); i != detector_output_per_video.end(); ++i)
if(i - detector_output_per_video.begin() == 49)
    std::cout << *i;
else
    std::cout << *i << " ";
if((sfg-1)/120 == 0)
std::cout << ",1" << endl;
if((sfg-1)/120 == 1)
std::cout << ",2" << endl;
if((sfg-1)/120 == 2)
std::cout << ",3" << endl;
if((sfg-1)/120 == 3)
std::cout << ",4" << endl;
if((sfg-1)/120 == 4)
std::cout << ",5" << endl;
if((sfg-1)/120 == 5)
std::cout << ",6" << endl;
if((sfg-1)/120 == 6)
std::cout << ",7" << endl;
if((sfg-1)/120 == 7)
std::cout << ",8" << endl;
if((sfg-1)/120 == 8)
std::cout << ",9" << endl;
if((sfg-1)/120 == 9)
std::cout << ",10" << endl;

detector_output_per_video.clear();
//} /* for trial */
		//	}  /*for*/
		//} /*for*/
                } /*for*/
	//EOF COLLECTING EVOLVED DETECTORS OUTPUTS FROM A SET OF TRAINING VIDEOS
	
	}else if(a == 3){
	//BOF TRAINING CLASSIFICATION ANN
		int dummy=0; //should be ignored and eventually removed
		train(&dummy);
	//EOF TRAINING CLASSIFICATION ANN
	}else if(a == 4){
int matches = 0;

cv::Mat confusion = cv::Mat::zeros(GESTURES,GESTURES, CV_32F);

		//BOF USING BOTH DETECTORS AND CLASSIFIER ANN TO PROCESS GESTURES FROM VIDEO/CAMERA
		
		//get detectors from stored files
		vector<Genotype> detectors;
		for(int i=1; i<=DETECTORS; i++){
			
			char id[1*sizeof(int)+1000];
			sprintf(id, "_%d", i);

			char * NEW_DETECTOR_FILE_NAME = new char[std::strlen(DETECTOR_FILENAME)+std::strlen(id)+1];
		    std::strcpy(NEW_DETECTOR_FILE_NAME,DETECTOR_FILENAME);
		    std::strcat(NEW_DETECTOR_FILE_NAME,id);

			genotype_set_size(145*145);
			Genotype ann = genotype_create();
			
			char * NEW_GENOTYPE_FITNESS_FILE_NAME = new char[std::strlen(GENOTYPE_FITNESS_FILE_NAME)+std::strlen(id)+1];
		    std::strcpy(NEW_GENOTYPE_FITNESS_FILE_NAME,GENOTYPE_FITNESS_FILE_NAME);
		    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,id);

			//for every detector,
			read_ann(&ann, NEW_DETECTOR_FILE_NAME, NEW_GENOTYPE_FITNESS_FILE_NAME, 0, 1, 0);
			
			detectors.push_back(ann);
		}

cout << endl;
		int type=0; //recurrent network
for(int trial=0; trial < 20; trial++){
cout << "trial " << trial << endl << endl;
		//for every stored training video
		for(int sfg=0;sfg<GESTURES;sfg++){
//cout << "digit " << sfg << endl;
			for(int person=0;person<PEOPLE;person++){
//cout << "person " << person << endl;
				if(person == 1 || person == 4 || person == 3) continue;
				
								//collect detectors outputs
				vector<double> detector_output_per_video;
					
				//for 6 evolved detectors (should be 10, but not always all 10 detectors are evolved - some may not reach minimum fitness)
				for(int i=1; i<=DETECTORS; i++){
					char id[1*sizeof(int)+1000];
					sprintf(id, "_%d", i);
            	
					char * NEW_DETECTOR_FILE_NAME = new char[std::strlen(DETECTOR_FILENAME)+std::strlen(id)+1];
				    std::strcpy(NEW_DETECTOR_FILE_NAME,DETECTOR_FILENAME);
				    std::strcat(NEW_DETECTOR_FILE_NAME,id);
            	
					genotype_set_size(145*145);
					Genotype ann = genotype_create();
            	
            	
					char * NEW_GENOTYPE_FITNESS_FILE_NAME = new char[std::strlen(GENOTYPE_FITNESS_FILE_NAME)+std::strlen(id)+1];
				    std::strcpy(NEW_GENOTYPE_FITNESS_FILE_NAME,GENOTYPE_FITNESS_FILE_NAME);
				    std::strcat(NEW_GENOTYPE_FITNESS_FILE_NAME,id);
            	
					//for every detector, read its ANN from file
					read_ann(&ann, NEW_DETECTOR_FILE_NAME, NEW_GENOTYPE_FITNESS_FILE_NAME, 0, 1, 0);
					
					int mode = 0; //using pre-recorded videos
					
					//string of feature graphs
					vector<Graph> string_feature_graph_demo;
					
					float diagonal_size = 0.0;
					
					//use pre-recorded raw video files or previously constructed and saved affinity matrices
							std::stringstream ss;
							std::stringstream ss2;
							std::stringstream ss3;
							ss << sfg;
							ss2 << person;
							ss3 << trial;
							std::string full_filename;
							string filename = "data/test_data/stored/";
							full_filename.append(filename);
							full_filename.append(ss.str());
							full_filename.append("_");
							full_filename.append(ss2.str());
							full_filename.append("_");
							full_filename.append(ss3.str());
							full_filename.append(".yaml");
							Mat resized_M;
					if(!std::ifstream(full_filename.c_str())){
						int frames = 0;//start_capture(mode, string_feature_graph_demo, -1/*-1*/, diagonal_size, 80, sfg);
            	    	printf("finished\n");
						//make affinity matrix
						Mat M = match_strings(string_feature_graph_demo, string_feature_graph_demo, diagonal_size);
            	
						Size size(8,8);
						resized_M = Mat(8,8, CV_32F, cvScalar(0.));
						resize(M,resized_M,size);
						
						FileStorage fs(full_filename, FileStorage::WRITE);
						fs << "mat" << resized_M;
						fs.release();
					}else{
						FileStorage fs(full_filename, FileStorage::READ);
						fs["mat"] >> resized_M;
						fs.release();

					}
            	
					//collect detectors outputs
					int type=0;
					double ann_output = feed(&ann, resized_M, type);
					
					detector_output_per_video.push_back(ann_output);
				}



			genotype_set_size(145*145);
			
			//feed classifier neural network together with detectors outputs
			type=1; //feed-forward network
			cv::Mat inputs(detector_output_per_video, true);
			double ann_output = feed(&detectors[0], inputs, type, sfg, confusion);
			cout << "digit " << sfg << " - output " << ann_output << endl;

if(sfg == ann_output) matches++;


detector_output_per_video.clear();
}

	//EOF COLLECTING EVOLVED DETECTORS OUTPUTS FROM A SET OF TRAINING VIDEOS
			}
		}

printf("matches: %d \n", matches);

confusion = confusion / 20;
cout << "confusion matrix: " << endl << confusion << endl;


		//EOF USING BOTH DETECTORS AND CLASSIFIER ANN TO PROCESS GESTURES FROM VIDEO/CAMERA
	}else if(a == 9){
		//prepare data
		std::string directory;
		cout << "Enter top level folder to scan: ";
		cin >> directory;
		cout << "dir: " << directory << endl;
		
        	vector<string> files = vector<string>();
        	//call the recursive directory traversal method
        	//getdir(directory,files,true);
        	//list all files within the list.
		for(unsigned int i = 0;i<files.size();i++){
		    //cout << files[i] << endl;
				std::vector<std::string> x = split(files[i], '.');
				if ((x.size() > 0) && (x[1] == "csv")){
			
					string full_filename = directory;
					full_filename.append("/");
					full_filename.append(files[i]);
					cout << "file: " << full_filename.c_str() << endl;
			
					ifstream file ( full_filename.c_str() );
					string value;
					int counter = 0;
			
					std::vector<std::string> features;
			
					std::vector<std::string> file_name_part_vector = split(files[i].c_str(), '.');
					string processed_filename = file_name_part_vector[0];
					processed_filename.append(".txt");
					//cout << processed_filename << endl;
			
					string full_processed_filename = directory;
					full_processed_filename.append("/");
					full_processed_filename.append(processed_filename);
			
					ofstream outputFile(full_processed_filename.c_str());
			
					while ( file.good() )
					{
						if(counter % 7 == 0 && counter != 0){
							//cout << features[4] << " | " << features[5] << " | " << features[6] << endl;
					
							std::ofstream outfile;

						  	outfile.open(full_processed_filename.c_str(), std::ios_base::app);
							outfile << features[4] << " " << features[5] << " " << features[6] << "\n";
				
							features.clear();
						}
				
						getline ( file, value, ';' );
						features.push_back(value);
				
						counter++;
					}
					cout << "num: " << counter << endl;
			
				}
		}
	}
	
	return 0;
}

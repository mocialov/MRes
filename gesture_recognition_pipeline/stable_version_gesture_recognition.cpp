/* To compile:
	ghc genotype.c -c && ghc -c -O2 Safe.hs && ghc --make -no-hs-main -optc-O -L/usr/lib -lc++ -lc++abi -lm -lc -I/usr/include stable_version_gesture_recognition.cpp Safe -c -lm -lstdc++ `pkg-config opencv --cflags --libs` && ghc --make -no-hs-main stable_version_gesture_recognition.o genotype.o Safe -o hand -L/usr/lib -I/usr/include -lstdc++ `pkg-config opencv --cflags --libs` -lpython2.7
*/

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

//convert to string patch
namespace patch
{
	template < typename T > std::string to_string( const T& n )
	{
		std::ostringstream stm;
		stm << n;
		return stm.str();
	}
}

//comparing two points
struct myclass {
    bool operator() (cv::Point pt1, cv::Point pt2) { return (pt1.x < pt2.x);}
} myobject;



Rect main_selected_head; 
Rect main_selected_body; 
Rect prev_head;
Rect prev_body;


cv::Point face_median;


//distance between 2 poitns
int point_distance(cv::Point p0, cv::Point p1){
    return sqrt((p0.x - p1.x)*(p0.x - p1.x) + (p0.y - p1.y)*(p0.y - p1.y));
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
//function is adapted to do some post-processing for detecting NAO robot's head and torso
//Look at the MSc Software Engineering to find a way to do the same for people's faces and upper bodies

//Basically, the NAO detection happens as follows:
// 1) All faces/bodies are found
// 2) the ones below/above the half of the screen are ignored due to the assumptions
// 3) circles are found in every found face/torso using Hough circles function
// 4) the ones that do not have circles are ignored
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

    std::vector<Point> circle_coordinates;
    for(int i = 0; i<selected_head.size(); i++){
        Mat roi = grayscaleFrame.colRange(selected_head[i].x,selected_head[i].x+selected_head[i].width).rowRange(selected_head[i].y,selected_head[i].y+selected_head[i].height);
        vector<Vec3f> circles;
        HoughCircles(roi, circles, CV_HOUGH_GRADIENT, 1, selected_head[i].width/5, 100, 10, 0, 4 );
        for( size_t j = 0; j < circles.size(); j++ ) {
            Point center(cvRound(circles[j][0])+selected_head[i].x, cvRound(circles[j][1])+selected_head[i].y);
            circle_coordinates.push_back(center);

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

	int found_max = -1;

    for(int i = 0; i < selected_head.size(); i++){
        int found = -1;
        for(int j = 0; j < circle_coordinates.size(); j++){
            if(circle_coordinates[j].inside(selected_head[i])){
                found++;
            }
        }

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
		face = prev_head;
	}

	//rectangle(frame, face, Scalar(0,0,0));

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

                circle(frame,center,circles[j][2],Scalar(0,0,255),0);
            }
        }
    }///has to be right, because it is showing circles in right places


	if(main_selected_body.x != 999 && main_selected_body.y != 999 && main_selected_body.x != -1 && main_selected_body.y != -1){
		prev_body = main_selected_body;
	}

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

//detect if two nodes are connected in a graph
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
int start_capture(int mode, vector<Graph> &string_feature_graph, int limit, float &diagonal_size, int duration, int video_nr){
	//Matrices to hold frame, background and foreground
	Mat frame;
	Mat back;
	Mat fore;

	// holds detected palm centres 
	vector<pair<Point,double> > palm_centers;

	VideoCapture cap;

	//construct file name from given parameters for a pre-recorded video file
	std::stringstream ss;
	ss << video_nr;
	std::string full_filename;
	string filename = "data/test_data/train_data_";
	full_filename.append(filename);
	full_filename.append(ss.str());
	full_filename.append(".ogv");


	//Two modes: 0 - start pre-recorded video; 1 - start camera stream
	if(mode == 0){
		cap = VideoCapture(full_filename);
	}
	else if(mode == 1){
		cap = VideoCapture(0);
	}

	if( !cap.isOpened() )
		return -1;

	//cv3
	cv::Ptr<BackgroundSubtractorMOG2> bg = createBackgroundSubtractorMOG2();
	bg->setNMixtures(3);
	bg->setDetectShadows(0);
	//bg->apply(img,mask);

	//cv2
	//Supporting class that does background substraction
	//BackgroundSubtractorMOG2 bg;
	//bg.set("nmixtures",3);
	//bg.set("detectShadows",false);

	//initialise windows to show results
	namedWindow("Frame");
	namedWindow("Background");

	//interval for background update (needed in case if the background changes)
	int backgroundFrame=500;

	//calculate the diagonal size of the frame (needed for as a factor when calculating distance between features in different frames)
	double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	diagonal_size = sqrt(width*width + height*height);

	int frames=0;

	//for a number of frames or until the end of the video
		for(int r =0;r<mode==1 ? duration : INFINITY;r++)
		{		
			//gesture feature hierarchy (0 - no features found)
			int hierarchy = 0;

			//vector that holds potential limbs
			vector<Limb> potential_limbs;

			//contours of the detected movement
			vector<vector<Point> > contours;

			//Get the frame from either a pre-recorded video or a camera stream
			cap >> frame;

			//pre-recorded video ended - return number of frames
			if(frame.empty() || frames==limit){ cout << "here" << (frame.empty()) << " " << (frames==limit) << endl;
	            return frames;}

			frames++;	

			//frame = frame + Scalar(75, 75, 75); //increase the brightness by 75 units
			//reduceIllumination(frame);

			//two vectors that hold detected faces and upper bodies
			Rect approx_faces = detectFaces(frame);

			Rect approx_upperBodies = detectUpperBodies(frame);


			//Update the current background model and extract the foreground
			if(backgroundFrame>0)
			{
				bg->apply(frame,fore);backgroundFrame--;
					//bg.operator ()(frame,fore);backgroundFrame--;
				}else{
					//bg.operator()(frame,fore,0);
				bg->apply(frame,fore,0);
				}

				//Get background image to display it
				//bg.getBackgroundImage(back);
            	
				//Enhance edges in the foreground by applying erosion and dilation
				erode(fore,fore,Mat());
				dilate(fore,fore,Mat());
            	
				//Find the contours in the foreground
				findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_TC89_KCOS);
				vector<vector<Point> > newcontours;

				//find biggest face of all faces detected (only one person is supported in this version)
				cv::Point biggestUpperBodyCenter(-1, -1);
				Rect biggestUpperBody;
				Rect biggestFace;
				cv::Point biggestFaceCenter(-1, -1);
				float biggestFaceArea = 0.0;
				cv::Point possibleElbowStart(-1,-1);
				cv::Point possibleElbowEnd(-1,-1);

				if(approx_faces.x != -1 && approx_faces.y != -1 && approx_faces.width != -1 && approx_faces.height != -1){
				//biggestFace = approx_faces[0];
			    //for (int i = 1; i < approx_faces.size(); i++) {
				//  if((biggestFace.width * biggestFace.height) < (approx_faces[i].width * approx_faces[i].height))
					biggestFace = approx_faces;
			    //}

				rectangle(frame, biggestFace, Scalar(0,0,255));

				biggestFaceArea = biggestFace.width * biggestFace.height;
				biggestFaceCenter.x = biggestFace.x + biggestFace.width/2.0;
				biggestFaceCenter.y = biggestFace.y + biggestFace.height/2.0;

				cv::Point pt1(biggestFace.x + biggestFace.width, biggestFace.y + biggestFace.height);
				cv::Point pt2(biggestFace.x, biggestFace.y);


				//find biggest upper body of all upper bodies deteted (only one person is supported in this version)
				//if(approx_upperBodies.size() > 0){
					//increase features hierarchy level
					hierarchy = 1;

					biggestUpperBody = approx_upperBodies;

				//    for (int i = 1; i < approx_upperBodies.size(); i++) {
				//	  if((biggestUpperBody.width * biggestUpperBody.height) < (approx_upperBodies[i].width * approx_upperBodies[i].height))
				//		biggestUpperBody = approx_upperBodies[i];
				//    }	

					rectangle(frame, biggestUpperBody, Scalar(0,0,255));

				    biggestUpperBodyCenter.x = biggestUpperBody.x + biggestUpperBody.width/2.0;
				    biggestUpperBodyCenter.y = biggestUpperBody.y+biggestUpperBody.height/2.0;//biggestFace.y+biggestFace.height;
			//}

			// get rid of the contours of the foreground that are on the top of the biggest detected face (restrict current implementation)
			for(int i=0;i<contours.size();i++){
				RotatedRect rect_test=minAreaRect(Mat(contours[i]));
					Rect intersectionRectangle = rect_intersect(biggestFace, rect_test.boundingRect());
					if((intersectionRectangle.width * intersectionRectangle.height) > biggestFaceArea * 0.5){
						//..
					}else{
						newcontours.push_back(contours[i]);
					}
			}
		}else{
			newcontours = contours;
		}

			for(int i=0;i<newcontours.size();i++)
				//Ignore all small insignificant areas (currently use 30% of the biggest area face)
				if(contourArea(newcontours[i])>=(biggestFaceArea * 0.3) && (biggestFaceArea > 0.0))		    
				{
					possibleElbowStart.x = -1;
					possibleElbowStart.y = -1;
					possibleElbowEnd.x = -1;
					possibleElbowEnd.y = -1;

					vector<Point2f> limb_details_temp;

					Limb limb;
	      			//default limb
	      			limb.start.x = -1;
	      			limb.start.y = -1;
	      			limb.end.x = -1;
	      			limb.end.y = -1;
	      			limb.break_point.x = -1;
	      			limb.break_point.y = -1;
					//Draw contour
					vector<vector<Point> > tcontours;
					tcontours.push_back(newcontours[i]);
					//drawContours(frame,tcontours,-1,cv::Scalar(0,0,255),2);

					//Detect Hull in current contour
					vector<vector<Point> > hulls(1);
					vector<vector<int> > hullsI(1);
					convexHull(Mat(tcontours[0]),hulls[0],false);
					convexHull(Mat(tcontours[0]),hullsI[0],false);
					//drawContours(frame,hulls,-1,cv::Scalar(0,255,0),2);

					//Find minimum area rectangle to enclose hand
					RotatedRect rect=minAreaRect(Mat(tcontours[0]));
					Point2f vertices[4];
					rect.points(vertices);

					//is a limb?					
	          			biggestUpperBody.y = biggestUpperBody.y;//+biggestFace.height;
	          			biggestUpperBody.height = biggestUpperBody.height;
	

						Rect biggestUpperBodyTemp(biggestUpperBody.tl(), biggestUpperBody.size());  //use copy for expansion

						//extends biggest upper body in width (to increase confidence in the intersection between hand and the upper body)
						cv::Size deltaSize( biggestUpperBodyTemp.width *1.5, biggestUpperBodyTemp.height);
						cv::Point offset( deltaSize.width/2, deltaSize.height/2); 
						biggestUpperBodyTemp += deltaSize*3;
						biggestUpperBodyTemp -= offset*3;

						rectangle(frame, biggestUpperBodyTemp, Scalar(255,255,255));
						rectangle(frame, rect.boundingRect(), Scalar(255,255,255));

	          			Rect potential_limb_intersections = rect_intersect(rect.boundingRect(), biggestUpperBodyTemp);

						//extend intersection rectangle
						cv::Size deltaSize2( potential_limb_intersections.width * .1, potential_limb_intersections.height * .1 );
						cv::Point offset2( deltaSize2.width/2, deltaSize2.height/2);
						potential_limb_intersections += deltaSize2;
						potential_limb_intersections -= offset2;

	          			if(potential_limb_intersections.width * potential_limb_intersections.height > rect.boundingRect().width * rect.boundingRect().height * 0.1){
							//increase detected features hierarchy level
							hierarchy = 2;
	                    	for(int m=0;m<4;m++)
	                        	if(dist(limb.start, limb.end) < dist((vertices[m] + vertices[(m+1)%4])*.5, (vertices[(m+2)%4] + vertices[(m+3)%4])*.5) && potential_limb_intersections.contains((vertices[m] + vertices[(m+1)%4])*.5))
	                            {
	                              	limb.start = (vertices[m] + vertices[(m+1)%4])*.5;
	                              	limb.end = (vertices[(m+2)%4] + vertices[(m+3)%4])*.5;
	                              	limb.limb_bounding_rectangle = rect;
	                            }
	          			}

					//Find Convex Defects
					vector<Vec4i> defects;
					if(hullsI[0].size()>0)
					{
						Point2f rect_points[4]; rect.points( rect_points );
						Point rough_palm_center;
						convexityDefects(tcontours[0], hullsI[0], defects);
						if(defects.size()>=3)
						{
							vector<Point> palm_points;
							for(int j=0;j<defects.size();j++)
							{
								int startidx=defects[j][0]; Point ptStart( tcontours[0][startidx] );
								int endidx=defects[j][1]; Point ptEnd( tcontours[0][endidx] );
								int faridx=defects[j][2]; Point ptFar( tcontours[0][faridx] );

								//Sum up all the hull and defect points to compute average
								rough_palm_center+=ptFar+ptStart+ptEnd;
								palm_points.push_back(ptFar);
								palm_points.push_back(ptStart);
								palm_points.push_back(ptEnd);
							}

							//Get palm center by 1st getting the average of all defect points, this is the rough palm center,
							//Then U chose the closest 3 points ang get the circle radius and center formed from them which is the palm center.
							rough_palm_center.x/=defects.size()*3;
							rough_palm_center.y/=defects.size()*3;
							Point closest_pt=palm_points[0];
							vector<pair<double,int> > distvec;
							for(int i=0;i<palm_points.size();i++)
								distvec.push_back(make_pair(dist(rough_palm_center,palm_points[i]),i));
							sort(distvec.begin(),distvec.end());

							//Keep choosing 3 points till you find a circle with a valid radius
							//As there is a high chance that the closes points might be in a linear line or too close that it forms a very large circle
							pair<Point,double> soln_circle;
							for(int i=0;i+2<distvec.size();i++)
							{
								Point p1=palm_points[distvec[i+0].second];
								Point p2=palm_points[distvec[i+1].second];
								Point p3=palm_points[distvec[i+2].second];
								soln_circle=circleFromPoints(p1,p2,p3);//Final palm center,radius
								if(soln_circle.second!=0)
									break;
							}

							//Find avg palm centers for the last few frames to stabilize its centers, also find the avg radius
							palm_centers.push_back(soln_circle);
							if(palm_centers.size()>10)
								palm_centers.erase(palm_centers.begin());

							Point palm_center;
							double radius=0;
							for(int i=0;i<palm_centers.size();i++)
							{
								palm_center+=palm_centers[i].first;
								radius+=palm_centers[i].second;
							}
							palm_center.x/=palm_centers.size();
							palm_center.y/=palm_centers.size();
							radius/=palm_centers.size();

							//Draw the palm center and the palm circle
							//The size of the palm gives the depth of the hand
							//circle(frame,palm_center,5,Scalar(144,144,255),3);
							//circle(frame,palm_center,radius,Scalar(144,144,255),2);

							//Detect fingers by finding points that form an almost isosceles triangle with certain thesholds
							int no_of_fingers=0;
							for(int j=0;j<defects.size();j++)
							{
								int startidx=defects[j][0]; Point ptStart( tcontours[0][startidx] );
								int endidx=defects[j][1]; Point ptEnd( tcontours[0][endidx] );
								int faridx=defects[j][2]; Point ptFar( tcontours[0][faridx] );

								double Xdist=sqrt(dist(palm_center,ptFar));
								double Ydist=sqrt(dist(palm_center,ptStart));
								double length=sqrt(dist(ptFar,ptStart));

								//circle(frame,ptStart,5,Scalar(0,0,255),3);

								double retLength=sqrt(dist(ptEnd,ptFar));
								//Play with these thresholds to improve performance
								if(length<=3*radius&&Ydist>=0.4*radius&&length>=10&&retLength>=10&&max(length,retLength)/min(length,retLength)>=0.8)
									if(min(Xdist,Ydist)/max(Xdist,Ydist)<=0.8)
									{
										if((Xdist>=0.1*radius&&Xdist<=1.3*radius&&Xdist<Ydist)||(Ydist>=0.1*radius&&Ydist<=1.3*radius&&Xdist>Ydist)){
											if(dist(ptEnd, limb.end) <= dist(limb.start, limb.end) * .1){
												//increment hierarchy level
												hierarchy = 3;
											  	limb_details_temp.push_back(Point2f(ptEnd.x, ptEnd.y));
											}
											//line( frame, ptEnd, ptFar, Scalar(0,255,0), 1 );
											no_of_fingers++;

											if(dist(ptStart, limb.start) >= std::min(rect.boundingRect().width, rect.boundingRect().height)*.2 && dist(ptStart, limb.end) >= std::min(rect.boundingRect().width, rect.boundingRect().height)*.2 && lies_on_contour(newcontours, ptStart) && dist(ptStart, ptEnd) > dist(possibleElbowStart, possibleElbowEnd)){
												possibleElbowStart = ptStart;
												possibleElbowEnd = ptEnd;
											}
										}

									}
							}

							//circle(frame,possibleElbowStart,5,Scalar(0,255,0),3);
							//circle(frame,possibleElbowEnd,5,Scalar(255,0,0),3);

							no_of_fingers=min(5,no_of_fingers);
						}

					}

					//since detected hierarchy 3 data can be very dense, it can be clustered to identify only clusters instead of actual data
					if(limb_details_temp.size() > 0){
						Mat labels;
						int cluster_number = std::min(5, (int)limb_details_temp.size());
						TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0 );
						Mat centers;
						kmeans(limb_details_temp, cluster_number, labels, criteria, 1, KMEANS_PP_CENTERS, centers);

						for(int h=0;h<centers.rows;h++)
							limb.details.push_back(Point(centers.at<float>(h,0), centers.at<float>(h,1)));

				    }

					limb.break_point = possibleElbowStart;
					potential_limbs.push_back(limb);
				}

				//frame to be used to present a skeletal data from identified end-points
				Mat skeletonFrame = Mat(frame.rows*.5,frame.cols*.5, CV_8UC3, cv::Scalar(255,255,255));
				double rad = 5;
				double scale_factor = .5;

				//face
				Point faceCenter(biggestFaceCenter.x*scale_factor, biggestFaceCenter.y*scale_factor);
				circle(skeletonFrame,faceCenter,rad,Scalar(0,255,0),2);

		        if((biggestUpperBodyCenter.x != -1) || (biggestUpperBodyCenter.y != -1)){                     
		          //neck
		          Point neckCenter(biggestUpperBodyCenter.x*scale_factor, biggestUpperBodyCenter.y*scale_factor);
				  circle(skeletonFrame,neckCenter,rad,Scalar(0,0,255),2);
				  line( skeletonFrame, faceCenter, neckCenter, Scalar(0,0,0), 1, 8 );

		          //shoulders
		          Point shoulder1Center((biggestUpperBodyCenter.x-biggestUpperBody.width/2.0)*scale_factor, biggestUpperBodyCenter.y*scale_factor);
				  circle(skeletonFrame,shoulder1Center,rad,Scalar(0,0,255),2);
				  line( skeletonFrame, shoulder1Center, neckCenter, Scalar(0,0,0), 1, 8 );
				  Point shoulder2Center((biggestUpperBodyCenter.x+biggestUpperBody.width/2.0)*scale_factor, biggestUpperBodyCenter.y*scale_factor);
				  circle(skeletonFrame,shoulder2Center,rad,Scalar(0,0,255),2);
				  line( skeletonFrame, shoulder2Center, neckCenter, Scalar(0,0,0), 1, 8 );

				  //waist
				  Point waistCenter((biggestUpperBody.tl().x + biggestUpperBody.width/2.0)*scale_factor, (biggestUpperBody.tl().y+biggestUpperBody.height)*scale_factor);
                                  //cout << waistCenter << endl;
				  circle(skeletonFrame,waistCenter,rad,Scalar(0,0,255),9);
				  line( skeletonFrame, shoulder1Center, waistCenter, Scalar(0,0,0), 1, 8 );
				  line( skeletonFrame, shoulder2Center, waistCenter, Scalar(0,0,0), 1, 8 );
		        }

				// limbs
		        for(int p=0; p<potential_limbs.size(); p++){
		          if(potential_limbs[p].break_point.x != -1 && potential_limbs[p].break_point.y != -1 && potential_limbs[p].start.x != -1 && potential_limbs[p].start.y != -1 && potential_limbs[p].end.x != -1 && potential_limbs[p].end.y != -1){
					Point limbEnd(potential_limbs[p].end.x*scale_factor, potential_limbs[p].end.y*scale_factor);
				  	circle(skeletonFrame,limbEnd,rad,Scalar(0,0,0),2);
					Point limbMiddle(potential_limbs[p].break_point.x*scale_factor, potential_limbs[p].break_point.y*scale_factor);
				  	circle(skeletonFrame,limbMiddle,rad,Scalar(0,0,0),2);
				  	Point limbStart(potential_limbs[p].start.x*scale_factor, potential_limbs[p].start.y*scale_factor);
				  	line( skeletonFrame, limbStart, limbMiddle, Scalar(0,0,0), 1, 8 );
					line( skeletonFrame, limbMiddle, limbEnd, Scalar(0,0,0), 1, 8 );
				  }else{
					  if(potential_limbs[p].start.x != -1 && potential_limbs[p].start.y != -1 && potential_limbs[p].end.x != -1 && potential_limbs[p].end.y != -1){
				  		Point limbEnd(potential_limbs[p].end.x*scale_factor, potential_limbs[p].end.y*scale_factor);
				  		circle(skeletonFrame,limbEnd,rad,Scalar(0,0,0),2);
				  		Point limbStart(potential_limbs[p].start.x*scale_factor, potential_limbs[p].start.y*scale_factor);
				  		line( skeletonFrame, limbStart, limbEnd, Scalar(0,0,0), 1, 8 );
				      }
			  	  }

				  //limb details
				  for(int l=0; l<potential_limbs[p].details.size(); l++){
					if(dist(potential_limbs[p].details[l], potential_limbs[p].break_point) != 0.0){
						Point limb_detal(potential_limbs[p].details[l].x*scale_factor, potential_limbs[p].details[l].y*scale_factor);
						circle(skeletonFrame,limb_detal,rad,Scalar(0,0,0),2);
					}
				  }
		        }



					//Below is the construction of the feature graphs (FG)


					//feature end-points: 
					//0-face
					//1-hand1
					//2-hand2
					//3-shoulder1
					//4-shoulder2
					//5-elbow1
					//6-elbow2

		    		// create the graph given in above fugure
	        		int V = 5;
	        		struct Graph* graph = createGraph(V);

					//features
					int item = 1;

					//hierarchy
					std::stringstream s00;
					s00 << "hierarchy: " << hierarchy;
					putText(skeletonFrame, s00.str(), cvPoint(500,15*(item++)), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,0), 1, CV_AA);

					//face-hand
					std::stringstream s0;
					s0 << "face-hand-1: ";
					if(potential_limbs.size() > 0){
						s0 << (int)round(sqrt(dist(faceCenter, potential_limbs[0].end)));
						addEdge(0, 1, graph, 0, faceCenter, 1, potential_limbs[0].end, sqrt(dist(faceCenter, potential_limbs[0].end))); //face-hand1
					}
					else{
						s0 << "-";
						//addEdge(0, 1, graph, 0, Point(0,0), 1, Point(0,0), 0.000001); //face-hand1
					}
					//putText(skeletonFrame, s0.str(), cvPoint(30,15*(item++)), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
					std::stringstream s;
					s << "face-hand-2: ";
					if(potential_limbs.size() > 1){
						s << (int)round(sqrt(dist(faceCenter, potential_limbs[1].end)));
						addEdge(0, 2, graph, 0, faceCenter, 2, potential_limbs[1].end, sqrt(dist(faceCenter, potential_limbs[1].end))); //face-hand2
					}else{
						s << "-";
						//addEdge(graph, 0, Point(0,0), 2, Point(0,0), 0); //face-hand2
					}
					//putText(skeletonFrame, s.str(), cvPoint(30,15*(item++)), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);


					//hand-hand
					std::stringstream s1;
					s1 << "hand-hand: ";
					if(potential_limbs.size() == 2){
						s1 << (int)round(sqrt(dist(potential_limbs[1].end, potential_limbs[0].end)));
						addEdge(1, 2, graph, 1, potential_limbs[1].end, 2, potential_limbs[0].end, sqrt(dist(potential_limbs[1].end, potential_limbs[0].end))); //hand1-hand2 (same as hand2-hand1)
					}else{
						s1 << "-";
						//addEdge(graph, 1, Point(0,0), 2, Point(0,0), 0); //hand1-hand2 (same as hand2-hand1)
					}
					//putText(skeletonFrame, s1.str(), cvPoint(30,15*(item++)), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);

					//shoulder-shoulder
					Point shoulder1((biggestUpperBody.x + biggestUpperBody.width/2.0-biggestUpperBody.width/2.0)*scale_factor, (biggestFace.y+biggestFace.height)*scale_factor);
					Point shoulder2((biggestUpperBody.x + biggestUpperBody.width/2.0+biggestUpperBody.width/2.0)*scale_factor, (biggestFace.y+biggestFace.height)*scale_factor);
					std::stringstream s2;
					s2 << "shoulder-shoulder: " << (int)round(sqrt(dist(shoulder1, shoulder2)));
					//putText(skeletonFrame, s2.str(), cvPoint(30,15*(item++)), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
					addEdge(3, 4, graph, 3, shoulder1, 4, shoulder2, sqrt(dist(shoulder1, shoulder2))); //shoulder1-shoulder2 (same as shoulder2-shoudler1)

					//elbow-elbow
					std::stringstream s3;
					s3 << "elbow-elbow: ";
					if(potential_limbs.size() == 2){
						s3 << (int)round(sqrt(dist(potential_limbs[0].break_point, potential_limbs[1].break_point)));
						//addEdge(5, 6, graph, 5, potential_limbs[0].break_point, 6, potential_limbs[1].break_point, sqrt(dist(potential_limbs[0].break_point, potential_limbs[1].break_point))); //elbow1-elbow2 (same as elbow2-elbow1)
					}else{
						s3 << "-";
						//addEdge(graph, 5, Point(0,0), 6, Point(0,0), 0); //elbow1-elbow2 (same as elbow2-elbow1)
					}
					//putText(skeletonFrame, s3.str(), cvPoint(30,15*(item++)), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);

					// show skeleton frame
					imshow("Frame2",skeletonFrame);

	        		// print the adjacency list representation of the above graph
	        		//printGraph(graph);
					string_feature_graph.push_back(*graph);

					imshow("Frame",frame);  //shows frame after frame with pre-recorded or camera stream video together with identified feature end-points
					if(waitKey(10) >= 0) break;
		}

		return -1;
}




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

//Method that creates affinity matrix from an SFG
Mat match_strings(vector<Graph> test, vector<Graph> query, int max_length){

	vector <string> added;

	int graph_c = -1;

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


int found = 0;

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


// function reads genotype from fifo-queue. Currently function wastes CPU by continuously trying to read the file 
// After genotype is read, fifo-queue is removed
void read_ann(Genotype *genotype, char *NEW_GENOTYPE_FILE_NAME, char *NEW_GENOTYPE_FITNESS_FILE_NAME, int with_removal, int node_types_specified, int bypass_lines) {	

    int times = 0;
    int redo = 1;
    while(redo){


        FILE *fp = fopen(NEW_GENOTYPE_FILE_NAME, "rb");
        if ( fp == NULL ){

			if(fp != NULL && ftell(fp) >= 0){
            		fclose(fp);
			}
            
			sleep(1);
            continue;

            read_ann(genotype, NEW_GENOTYPE_FILE_NAME, NEW_GENOTYPE_FITNESS_FILE_NAME, with_removal, node_types_specified, bypass_lines);
        }

        while ( !feof (fp) )
        {

			int lines_counter = 0;

			std::ifstream inFile(NEW_GENOTYPE_FILE_NAME); 

			std::string unused;

			while ( std::getline(inFile, unused) ){
			   ++lines_counter;
			}

			inFile.close();

			if(!bypass_lines)
			if (lines_counter < 2){ sleep(5); continue; }


            redo = 0;
            if(isEmpty(fp) && (fp != NULL && ftell(fp) >= 0)){
            	fclose(fp);

                continue;
            }else{
                times++;
                
                if(times <= 100){
	
                    int result = genotype_fread(genotype, fp, node_types_specified);

					if(fp != NULL && ftell(fp) >= 0){
                    		fclose(fp);  
 					}
                    //removing file

					if(with_removal)
                    	remove(NEW_GENOTYPE_FILE_NAME);

                    break;
                }else{
                    break;
                }
            }
            
        }
    }
    
    if((with_removal && file_exist(NEW_GENOTYPE_FILE_NAME)) || (with_removal && file_exist(NEW_GENOTYPE_FITNESS_FILE_NAME))){
        read_ann(genotype, NEW_GENOTYPE_FILE_NAME, NEW_GENOTYPE_FITNESS_FILE_NAME, with_removal, node_types_specified, bypass_lines);
    }
}

// function converts float to double, leaving 7 decimal digits
double round_to_decimal(float f) {
    char buf[42];
    sprintf(buf, "%.7g", f);
    return atof(buf);
}


// funtion feeds input to a neural network by using hnn (Haskell) package
double feed(Genotype *genotype, Mat matrix, int type, int the_class = -1, cv::Mat confusion_matrix = cv::Mat::zeros(GESTURES,GESTURES, CV_32F)) {
	//reconstruct matrices from available data to a suitable format

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
    }

    for (int i=0; i<genotype_get_size(); i++){
        flat_matrix[i] = (*genotype)->genes[i];
    }

	//result array
	double *res;
	
	int asize = genotype_get_size();
	int bsize = matrix.rows * matrix.cols;
	int csize = (int)sqrt(genotype_get_size());
	
	/*		parameters:
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

		if(!isnan(val))
			confusion_matrix.at<float>((int)the_class, 0) += val;

		for(int i=2;i<=(int)*res;i++){
			if(!isnan(*(res+i)))
				confusion_matrix.at<float>((int)the_class, i-1) += *(res+i);

			if(val < *(res+i)){
				result = i-1;
				val = *(res+i);
			}
		}
	}
	
	return result;
}

//helper functions
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
	
	
	int a=-1;
	
	//when running on server, comment this (needs to start directly, without user input)
	printf("1 for training detectors, 2 for getting output from evolved detectors, 3 for training classifying ANN, 4 for testing evolved detectors together with trained classifier, 9 for data preparation: ");
	scanf("%d", &a);

	
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
        	
			//different ways to start the child process
	    	#elif defined __APPLE__
	    	    status = posix_spawn(&pid, "/bin/sh", NULL, NULL, argvv, environ);
	    	#else
				status = posix_spawn(&pid, "/bin/sh", NULL, NULL, argvv, environ);
	    	#endif
	
			free(name_with_extension);
		}
		//eof start hyperneat algorithms


		// for every generation, use every detector on an a set of images
		for(int j=0;j<GENERATIONS;j++){			
			
			// 10 detectors, 4 gestures
			Mat detectors_outputs = Mat(DETECTORS, 2, CV_32F, cvScalar(0.));
			
			// STEP_1: read ANN, create affinity matrix, feed affinity matrix to ANN and collect outputs for every detector in a matrix (rows - detectors; columns - gesture classes)
			for(int detector_id=1;detector_id<=DETECTORS;detector_id++){

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
				
				//genotype size 442x442 maximum connections - for the recurrent network (although it can still act as a feedforward network)
				genotype_set_size(101*101);
        		Genotype ann = genotype_create();

				//read ANN that had been generated from peas framework
				read_ann(&ann, NEW_GENOTYPE_FILE_NAME, NEW_GENOTYPE_FITNESS_FILE_NAME, 1, 1, 0);

				for(int sfg=1;sfg<=30;sfg++){ 
					
					///comment out or not, depending on the experiment
					//for(int person=1;person<=PEOPLE;person++){
					//	if(person == 3) continue;


						
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
							
//temp - because the experiment required it
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
							
							//if saved affinity matrix exists, read it instead of viewing pre-recorded videos
							if(!std::ifstream(full_filename.c_str())){
								int frames = start_capture(mode, string_feature_graph_demo, -1, diagonal_size, 80, sfg);
								//make affinity matrix
								Mat M = match_strings(string_feature_graph_demo, string_feature_graph_demo, diagonal_size);

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
								FileStorage fs(full_filename, FileStorage::READ);
								fs["mat"] >> resized_M;
								fs.release();
								
								//resize the affinity matrix
								Size size(10,10);
								if (resized_M.cols != 0)
									resize(resized_M,resized_M,size);
							}

//normalize the affinity matrix (do i really need this?)
if (resized_M.cols != 0){
	normalize(resized_M, resized_M, 0.0, 1.0, NORM_MINMAX, -1);

	for(int row=0;row<resized_M.rows;row++)
		for(int column=0;column<resized_M.cols;column++)
			if(isnan(resized_M.at<float>(row, column)))
				resized_M.at<float>(row, column) = 0.0;

	                    	
							//feed affinity matrix into HyperNEAT generated ANN
							int type=0;
							double ann_output = feed(&ann, resized_M, type);

							//cout << "ANN output: " << ann_output << endl;
							// round the output. NOTE: 0.5 means that the output neuron did not fire (sigmoid(0)==0.5)
							// there is something wrong with using simply 0, therefore value 0.001 is used instead
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
						int frames = start_capture(mode, string_feature_graph_demo, -1/*-1*/, diagonal_size, 80, sfg);
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
						int frames = start_capture(mode, string_feature_graph_demo, -1/*-1*/, diagonal_size, 80, sfg);
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

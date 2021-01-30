/* \author Aaron Brown */
// Create simple 3d highway enviroment using PCL
// for exploring self-driving car sensors

//#include "render/render.h"
#include "highway.h"

#include <ostream>
#include <string>

// HACK: Using a global pointer to a stream for logging (disabled: nullptr),
//       so this stream can be shared between all UKFs instances of the cars.
//       Otherwise the interface of a bunch of functions would have had to be changed 
//       in order to pass it through to the UKF class.
std::ostream *s_logger = nullptr;


int main(int argc, char** argv)
{
	// check for logging: --log [filename]
	if (argc > 1 && std::string(argv[1]) == "--log")
	{
		if (argc > 2)
			s_logger = new std::ofstream(argv[2]);
		else
		{
			s_logger = &std::cout;
		}
	}

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);

	// set camera position and angle
	viewer->initCameraParameters();
	float x_pos = 0;
	viewer->setCameraPosition ( x_pos-26, 0, 15.0, x_pos+25, 0, 0, 0, 0, 1);

	Highway highway(viewer);

	//initHighway(viewer);

	int frame_per_sec = 30;
	int sec_interval = 10;
	int frame_count = 0;
	int time_us = 0;

	double egoVelocity = 25;

	while (frame_count < (frame_per_sec*sec_interval))
	{
		viewer->removeAllPointClouds();
		viewer->removeAllShapes();

		//stepHighway(egoVelocity,time_us, frame_per_sec, viewer);
		highway.stepHighway(egoVelocity,time_us, frame_per_sec, viewer);
		viewer->spinOnce(1000/frame_per_sec);
		frame_count++;
		time_us = 1000000*frame_count/frame_per_sec;
	}

	if (s_logger)
	{
		highway.tools.saveLog("highway_traffic_steps.csv");
		delete s_logger;
	}

}
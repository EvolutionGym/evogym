#ifndef SIM_H
#define SIM_H

#include <fstream>

#include "main.h"

#include "Environment.h"
#include "ObjectCreator.h"
#include "Camera.h"

class Sim
{

public:

	//STATIC FUNCTIONS
	static void get_version();

	// SIM VARIABLES
	ObjectCreator creator;
	Environment environment;
	int sim_time;
	bool is_rendering_enabled;

	//SIMULATION SETTIGNS
	int physics_updates_per_step;

	Sim();

	void init(int x, int y);
	bool read_object_from_file(string file_name, string object_name, double x, double y);
	bool read_robot_from_file(string file_name, string robot_name, double x, double y);


	bool read_object_from_array(Matrix <double, Dynamic, Dynamic> grid, Matrix <double, 2, Dynamic> connections, string object_name, double x, double y);
	bool read_robot_from_array(Matrix <double, Dynamic, Dynamic> grid, Matrix <double, 2, Dynamic> connections, string robot_name, double x, double y);

	void set_action(string robot_name, MatrixXd action);
	bool step();

	void force_save();
	void revert(long int sim_time);
	int get_time();
	Ref <MatrixXd> pos_at_time(long int sim_time);
	Ref <MatrixXd> vel_at_time(long int sim_time);
	Ref <MatrixXd> object_pos_at_time(long int sim_time, string object_name);
	Ref <MatrixXd> object_vel_at_time(long int sim_time, string object_name);
	double object_orientation_at_time(long int sim_time, string object_name);
	void translate_object(double x, double y, string object_name);
	Ref <MatrixXi> get_actuator_indices(string robot_name);

	//void show_debug_window();
	//void hide_debug_window();
	//vector<int> get_debug_window_pos();

	~Sim();
};

#endif // !SIM_H




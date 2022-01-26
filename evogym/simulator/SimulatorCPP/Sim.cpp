#include "Sim.h"
#include "main.h"

void Sim::get_version() {
	cout << "Using Evolution Gym Simulator v2.2.5" << "\n";
}

Sim::Sim()
{
	//INIT ENVIRONMENT
	environment = Environment();
	environment.init();

	//OBJECTS
	creator = ObjectCreator(&environment);
	
	//SIM VARIABLES
	sim_time = 0;
	Sim::is_rendering_enabled = is_rendering_enabled;
	//TODO: remove redundant variable

	//SIMULATION SETTINGS
	physics_updates_per_step = 30;
}

void Sim::init(int x, int y) {
	creator.init_grid(Vector2d(x,y));
}

bool Sim::read_object_from_file(string file_name, string object_name,  double x, double y) {
	
	if (creator.read_object_from_file(file_name, object_name, Vector2d(x, y), false)) {
		environment.save_snapshot(0);
		return true;
	}
	return false;
}

bool Sim::read_robot_from_file(string file_name, string robot_name, double x, double y) {
	
	if (creator.read_object_from_file(file_name, robot_name, Vector2d(x, y), true)) {
		environment.init_robot(robot_name);
		environment.save_snapshot(0);
		return true;
	}
	return false;
	
}

bool Sim::read_object_from_array(Matrix <double, Dynamic, Dynamic> grid, Matrix <double, 2, Dynamic> connections, string object_name, double x, double y) {
	
	Matrix <double, 1, Dynamic> flat;
	flat.resize(1, grid.rows()*grid.cols());

	int grid_width = grid.cols();
	int grid_height = grid.rows();

	if (grid.IsRowMajor) {
		flat = Matrix <double, 1, Dynamic>(Map<Matrix <double, 1, Dynamic>>(grid.data(), grid.cols()*grid.rows()));
	}
	else{
		grid.transposeInPlace();
		grid_width = grid.rows();
		grid_height = grid.cols();
		flat = Matrix <double, 1, Dynamic>(Map<Matrix <double, 1, Dynamic>>(grid.data(), grid.cols()*grid.rows()));
	}

	if (creator.read_object_from_array(object_name, flat, connections, Vector2d(grid_width, grid_height), Vector2d(x, y), false)) {
		environment.save_snapshot(0);
		return true;
	}
	return false;
}

bool Sim::read_robot_from_array(Matrix <double, Dynamic, Dynamic> grid, Matrix <double, 2, Dynamic> connections, string robot_name, double x, double y) {

	Matrix <double, 1, Dynamic> flat;
	flat.resize(1, grid.rows()*grid.cols());

	int grid_width = grid.cols();
	int grid_height = grid.rows();

	if (grid.IsRowMajor) {
		flat = Matrix <double, 1, Dynamic>(Map<Matrix <double, 1, Dynamic>>(grid.data(), grid.cols()*grid.rows()));
	}
	else {
		grid.transposeInPlace();
		grid_width = grid.rows();
		grid_height = grid.cols();
		flat = Matrix <double, 1, Dynamic>(Map<Matrix <double, 1, Dynamic>>(grid.data(), grid.cols()*grid.rows()));
	}

	if (creator.read_object_from_array(robot_name, flat, connections, Vector2d(grid_width, grid_height), Vector2d(x, y), true)) {
		environment.init_robot(robot_name);
		environment.save_snapshot(0);
		return true;
	}
	return false;
}

void Sim::set_action(string robot_name, MatrixXd action) {
	environment.set_robot_action(robot_name, action);
}

bool Sim::step() {

	for (int i = 0; i < physics_updates_per_step; i++) {
		if (environment.step())
			return true;
	}

	sim_time++;
	environment.save_snapshot(sim_time);
	return false;
}

//void Sim::render(Camera camera) {
//
//	if (!Sim::is_rendering_enabled){
//		cout << "Error: Cannot render to camera because rendering is disabled.\n";
//		return;
//	}
//	interface.render(camera);
//}

void Sim::revert(long int sim_time) {
	if (environment.revert_to_snapshot(sim_time)) {
		Sim::sim_time = sim_time;
	}
}
void Sim::force_save() {
	environment.save_snapshot(sim_time);
}

int Sim::get_time() {
	return sim_time;
}

Ref <MatrixXd> Sim::pos_at_time(long int sim_time) {
	return environment.get_pos_at_time(sim_time);
}
Ref <MatrixXd> Sim::vel_at_time(long int sim_time) {
	return environment.get_vel_at_time(sim_time);
}
double Sim::object_orientation_at_time(long int sim_time, string object_name) {
	return environment.object_orientation_at_time(sim_time, object_name);
}
void Sim::translate_object(double x, double y, string object_name) {
	environment.translate_object(x, y, object_name);
	environment.save_snapshot(sim_time);
}


Ref <MatrixXd> Sim::object_pos_at_time(long int sim_time, string object_name) {
	return environment.object_pos_at_time(sim_time, object_name);
}
Ref <MatrixXd> Sim::object_vel_at_time(long int sim_time, string object_name) {
	return environment.object_vel_at_time(sim_time, object_name);
}

Ref <MatrixXi> Sim::get_actuator_indices(string robot_name) {
	return environment.get_robot(robot_name)->get_actuator_indicies();
}

//void Sim::show_debug_window() {
//	interface.show_debug_window();
//}
//
//void Sim::hide_debug_window() {
//	interface.hide_debug_window();
//}
//
//
//vector<int> Sim::get_debug_window_pos() {
//	return Sim::interface.get_debug_window_pos();
//}

Sim::~Sim()
{
}

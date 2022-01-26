#include "main.h";

#include <sstream>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <cmath>
#include <ctime>


#include "Sim.h"
#include "Environment.h"
#include "ObjectCreator.h"
#include "SimObject.h"
#include "Interface.h"
#include "Edge.h"

using namespace std;
using namespace Eigen;

GLubyte *image_data;
//Sim test_sim = Sim();

//LOOP CONTROL
double time_0;
int target_renders_per_second = 30;
int renders_per_second;
int total_renders = 0;

bool should_render() {

	double time_1 = clock() / 1000.0;
	double elapsed_time = time_1 - time_0;

	double did_render = false;
	
	if ((double)(renders_per_second + 1) / elapsed_time < target_renders_per_second) {
		renders_per_second += 1;
		did_render = true;
	}

	if (elapsed_time > 1.0) {
		total_renders += renders_per_second;
		cout << "Steps Per Second: " << renders_per_second << ", target: " << target_renders_per_second << "\n";
		cout << "Total Steps: " << total_renders << "\n\n";
		renders_per_second = 0;
		time_0 = time_1;
	}

	return did_render;
}

//vector <double> controller_weights;
//vector <double> controller_biases;
//double controller_time = 0;


int count_switch = 0;
Matrix <double, Dynamic, 2> get_action(Ref <MatrixXi> indicies) {

	int num_indicies = indicies.size();
	//cout << num << "\n";

	Matrix <double, Dynamic, 2> out;
	out.resize(num_indicies, 2);

	
	double range = 0.8;
	for (int i = 0; i < num_indicies; i++) {
		//out.row(i) << i+7, sin( controller_time + controller_biases[i]) * controller_weights[i] * 0.15 + 1.0;
		
		double randn = ((double)rand() / (RAND_MAX));
		out.row(i) << indicies(0, i), 1 + (randn - 0.5)*range / 0.5;

		//if (count_switch > 150)
		//	out.row(i) << indicies(0, i), 0.6;
		//else if (count_switch > 40)
		//	out.row(i) << indicies(0, i), 1.4;
		//else
		//	out.row(i) << indicies(0, i), 1;

	

		//if (count_switch > 150)
		//	out.row(i) << indicies(0, i), 1.0;
		//else if (count_switch > 30)
		//	out.row(i) << indicies(0, i), 1.6;
		//else
		//	out.row(i) << indicies(0, i), 0.6;

		//out.row(i) << indicies(0, i), 1.0;
	}
	//controller_time += 0.5;

	count_switch += 1;
	return out;
}

bool is_rendering_enabled = true;

int main(int argc, char** argv)
{
	Sim::get_version();
	Sim test_sim = Sim();
	test_sim.init(50, 50);


	test_sim.read_object_from_file("sim_files/walking-ground1.sob", "floor", 0, 0);
	//test_sim.read_object_from_file("sim_files/wall-floor.sob", "floor", 0, 0);
	//test_sim.read_object_from_file("sim_files/flat-ground-soft.sob", "floor", 0, 0);
	//test_sim.read_object_from_file("sim_files/flat-ground-soft2.sob", "floor", 0, 0);

	test_sim.read_object_from_file("sim_files/block-rigid.sob", "block_rigid1", 13, 1);
	test_sim.read_object_from_file("sim_files/block-rigid.sob", "block_rigid2", 13, 3);
	test_sim.read_object_from_file("sim_files/block-rigid.sob", "block_rigid3", 13, 5);
	//test_sim.read_object_from_file("sim_files/block-rigid.sob", "block_rigid4", 4, 7);

	//test_sim.read_object_from_file("sim_files/block-soft.sob", "block_soft", 14, 1);
	//test_sim.read_object_from_file("sim_files/block-rigid.sob", "block_soft2", 9, 10);


	//test_sim.read_robot_from_file("sim_files/unstable.sob", "robot", 10, 6);
	//test_sim.read_robot_from_file("sim_files/robot1.sob", "robot", 2, 6);
	test_sim.read_robot_from_file("sim_files/jump-bot.sob", "robot1", 3, 7);
	test_sim.read_robot_from_file("sim_files/jump-bot.sob", "robot2", 1, 1);
	//test_sim.read_robot_from_file("sim_files/big1.sob", "robot", 6, 10);
	//test_sim.read_robot_from_file("sim_files/plunger.sob", "robot", 2, 15);
	//test_sim.read_robot_from_file("sim_files/test-vertical-fixed.sob", "robot", 2, 15);
	//test_sim.read_robot_from_file("sim_files/test-vertical-fixed.sob", "robot", 4, 25);

	MatrixXd grid(3,1);
	grid << 1, 1, 1;

	cout << "main grid\n" << grid << "\n";

	MatrixXd connections(2, 2); //2 x n
	connections << 0, 1, 1, 2;
	//MatrixXd connections(2, 7); //2 x n
	//connections << 0, 2, 4, 0, 2, 1, 3, 1, 3, 5, 2, 4, 3, 5;

	cout << "main connections\n" << connections << "\n";

	test_sim.read_object_from_array(grid, connections, "vbeam", 10, 6);


	MatrixXi indicies1 = test_sim.get_actuator_indices("robot1");
	cout << test_sim.get_actuator_indices("robot1") << "\n";

	MatrixXi indicies2 = test_sim.get_actuator_indices("robot2");
	cout << test_sim.get_actuator_indices("robot2") << "\n";

	Interface::init();
	Interface test_viewer = Interface(&test_sim);
	//test_viewer.init();

	Camera camera_1;
	Camera camera_2;
	if (is_rendering_enabled){
		camera_1 = Camera(Vector2d(1.2, 0.4), Vector2d(3*1.5, 1*1.5), Vector2d(600, 200), false);
		camera_2 = Camera(Vector2d(1.2, 0.4), Vector2d(6, 2), Vector2d(600, 200), true);

		camera_1.set_resolution(1200, 400);
		camera_2.set_resolution(84, 84);
		test_viewer.show_debug_window();
		//test_sim.hide_debug_window();
	}

	//for (int i = 0; i < 14; i++) {
	//	controller_weights.push_back(double(rand() % 10000) / 10000.0);
	//	//cout << controller_weights[controller_weights.size() - 1] << "\n";
	//}

	//for (int i = 0; i < 14; i++) {
	//	controller_biases.push_back(double(rand() % 10000) / 10000.0 * 3.1415 * 2);
	//	//cout << controller_biases[controller_biases.size() - 1] << "\n";
	//}

	test_sim.revert(0);

	bool done = false;

	while (!done) {

		if (should_render()) {

			if (is_rendering_enabled) {
				test_viewer.render(camera_1);
				//test_sim.render(camera_2);
			}

			test_sim.set_action("robot1", get_action(indicies1));
			test_sim.set_action("robot2", get_action(indicies2));
			done = test_sim.step();
			done = false;

			//center camera on robot
			//Matrix<double, 2, Dynamic> robot_pos = test_sim.object_pos_at_time(test_sim.get_time(), "robot1");
			//Vector2d robot_com = robot_pos.rowwise().sum() / robot_pos.cols();
			//camera_1.set_pos(Vector2d(robot_com[0]+0.5, robot_com[1] + 0.1));

			//test_sim.translate_object(0.00, 0.05, "block_rigid3");

			//cout << test_sim.get_debug_window_pos()[0] << " " << test_sim.get_debug_window_pos()[1] << "\n";

			//cout << test_sim.object_orientation_at_time(test_sim.get_time(), "block_rigid") * 180/3.14 << "\n";


			//cout << test_sim.object_pos_at_time(0, "robot").cols() << "\n\n\n";
			//cout << test_sim.get_robot_com()[0] << " " << test_sim.get_robot_com()[1] << "\n";
			//camera_1.set_pos(Vector2d(test_sim.get_robot_com()[0], 0.4));
			//test_sim.render(camera_2);
			//image_data = camera_2.get_image_data_ptr();
		}
	}
	while (true) {
		if (should_render()) {
			//test_sim.render(camera_1);
		}
	}

	return 0;
}

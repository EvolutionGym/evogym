#ifndef OBJECTCREATOR_H
#define OBJECTCREATOR_H

#include <vector>

#include "main.h"

#include "Environment.h"
#include "SimObject.h";
#include "Robot.h"
#include "Boxel.h"
#include "Edge.h"

using namespace std;

class ObjectCreator
{
private:

	//DATA HOLDERS
	int grid_width;
	int grid_height;
	vector<Boxel> grid;
	
	int world_grid_width;
	int world_grid_height;
	vector<int> world_grid;

	vector <Vector2d> pos;
	vector <Vector2d> vel;
	vector <double> masses;
	vector <bool> fixed;
	vector <Edge> edges;

	//CONSTANTS
	Vector2d cell_size = Vector2d(0.1, 0.1);

	double mass = 1.0; //1.0 / 500.0; //1.0 / 5000.0; //

	double rigid_main_edge_spring_const = 300000000.0 /3.5; // 30000.0; //10000000.0;
	double rigid_structural_edge_spring_const = rigid_main_edge_spring_const / 2.0;

	double actuator_main_edge_spring_const = 300000000.0 / 5;
	double actuator_structural_edge_spring_const = actuator_main_edge_spring_const / 2.0;

	double soft_main_edge_spring_const = 300000000.0 / 6; //100000000.0 / 5; //100000000.0 /1.5; //10000.0;
	double soft_structural_edge_spring_const = soft_main_edge_spring_const / 4.0;

	double main_edge_spring_const = 100000.0; //10000000.0;
	double other_edge_spring_const = main_edge_spring_const / 15.0;

	//HELPER FUNCTIONS - for converting grid data into SimObjects
	int get_index(int x, int y);
	bool is_valid_in_world_grid(int x, int y);
	vector <int> get_connected_components();
	void explore_grid(int parent_value, int x, int y, vector<bool>* is_explored, vector<int>* component_values);

	void reset();
	int make_point(Vector2d p, Vector2d v, double m, bool f);
	int make_edge(int a_index, int b_index, double length_eq, double spring_const);
	Edge* get_edge(int index);
	int get_point_index(int index);
	
	//REF TO ENV
	Environment* env;

public:

	ObjectCreator();
	ObjectCreator(Environment* env);
	~ObjectCreator();

	//CREATE FUNCTIONS
	void init_grid(Vector2d grid_size);
	bool read_object_from_file(string file_name, string object_name, Vector2d init_pos = Vector2d(0.0, 0.0), bool is_robot = false);
	bool read_object_from_array(string object_name, Matrix <double, 1, Dynamic> local_grid, Matrix <double, 2, Dynamic> connections, Vector2d grid_size, Vector2d init_pos = Vector2d(0.0, 0.0), bool is_robot = false);
	//bool read_from_file(string file_name, bool is_robot = false, Vector2d init_pos = Vector2d(0.0, 0.0));
	
};

#endif // OBJECTCREATOR_H
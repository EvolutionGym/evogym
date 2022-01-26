#ifndef PHYSICS_ENGINE_H
#define PHYSICS_ENGINE_H

#include <vector>
#include "main.h"
#include "SimObject.h"
#include "Robot.h"
#include "Edge.h"

using namespace std;

class PhysicsEngine
{
private:

	//PHYSICS CONSTANTS
	double viscous_drag = 0.1; //0.2 before
	double gravity = 110; //200 //100000 before

	double collision_const_ground = 2000000.0*7; //10 //200; //200
	double collision_const_obj = 300000.0*7; //10 //300; //150
	double collision_vel_damping = 140.0; //120
	double collision_base_dist_additive = 0.005; //0.0005; //0.005;

	double static_friction_const = 0.5;
	double dynamic_friction_const = 0.2;
	double friction_const = 150.0 * 8; //6


	//ENVIRONMENT VARIABLES
	//vector<Vector2d_old>* pos_old;
	//vector<Vector2d_old>* vel_old;
	//vector<double>* mass_old;
	//vector<bool>* fixed_old;

	Matrix <double, 2, Dynamic>* pos;
	Matrix <double, 2, Dynamic>* pos_last;
	Matrix <double, 2, Dynamic>* vel;
	Matrix <double, 2, Dynamic>* vel_true;
	Matrix <double, 2, Dynamic>* mass;
	Matrix <bool, 2, Dynamic>* fixed;

	Matrix <int, 1, Dynamic>* a_index;
	Matrix <int, 1, Dynamic>* b_index;
	Matrix <double, 1, Dynamic>* length_eq;
	Matrix <double, 1, Dynamic>* length_eq_goal;
	Matrix <double, 1, Dynamic>* init_length_eq;
	Matrix <double, 1, Dynamic>* spring_const;


	vector<Edge>* edges;
	vector<SimObject*>* objects;

	//COLLISION FORCES
	Matrix <double, 2, Dynamic> collision_forces;
	Matrix <double, 2, Dynamic> collision_vels;
	//vector<Vector2d_old> collision_forces;


	//MISC FUNCTIONS
	bool get_intersecting(int edgeIndex1, int edgeIndex2);
	double dist_point_edge(int point_index, int edge_index);

	/*
	bool get_intersecting_fast(int edgeIndex1, int edgeIndex2);
	bool bounding_box_1_in_2(BoundingBox bbox1, BoundingBox bbox2);*/

public:

	vector <bool>* point_is_colliding;
	
	PhysicsEngine();
	~PhysicsEngine();

	void init(
		Matrix <double, 2, Dynamic>* pos, Matrix <double, 2, Dynamic>* pos_last, Matrix <double, 2, Dynamic>* vel, Matrix <double, 2, Dynamic>* vel_true,
		Matrix <double, 2, Dynamic>* mass, Matrix <bool, 2, Dynamic>* fixed,
		Matrix <int, 1, Dynamic>* a_index, Matrix <int, 1, Dynamic>* b_index, Matrix <double, 1, Dynamic>* length_eq, Matrix <double, 1, Dynamic>* length_eq_goal, 
		Matrix <double, 1, Dynamic>* init_length_eq, Matrix <double, 1, Dynamic>* spring_const, vector<Edge>* edges, 
		vector<SimObject*>* objects);
	
	void step_rk4(double dt);
	void step_euler(double dt);
	Matrix <double, 2, Dynamic> get_pos_vel_slope(Ref<Matrix <double, 2, Dynamic>> data, int num_points);
	
	void resolve_collisions();
	void resolve_object_collisions(SimObject* object_a, BBTreeNode* a_node, SimObject* object_b, BBTreeNode* b_node, bool are_same);
	bool is_any_robot_self_colliding();
	int count_object_self_collisions(SimObject* object_a, BBTreeNode* a_node, SimObject* object_b, BBTreeNode* b_node);
	void resolve_boxel_collisions(Boxel* main, Boxel* ref);
	void resolve_point_boxel_collisions(int p_index, Boxel* main, Boxel* ref);

	void resolve_edge_constraints();
	void update_true_vel(double dt);

	//void resolve_object_collisions(int object_1_index, int object_2_index);
	//void resolve_boxel_collisions(Boxel main, Boxel ref);
	//void resolve_point_boxel_collisions(int p_index, Boxel main, Boxel ref);

	//void detect_resolve_fine_collisions(int edge_index1, int edge_index2);
	//void resolve_edge_collisions(int object_index1, int object_index2);

	void update_actuator_goals(vector<Boxel*>* actuators, vector<double>* actuations);
	void update_actuators();
};

#endif // !PHYSICS_ENGINE_H


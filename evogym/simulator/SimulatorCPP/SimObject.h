#ifndef SIMOBJECT_H
#define SIMOBJECT_H


#include "main.h"

#include <vector>
#include <map>
#include <unordered_set>

#include "Boxel.h"
#include "BBTreeNode.h"
#include "Edge.h"

using namespace std;

class SimObject
{
private:

public:

	bool is_robot;

	int min_point_index;
	int max_point_index;

	int grid_w;
	int grid_h;
	int num_suface_neighbors;

	vector<Boxel> boxels;

	Matrix <double, 1, Dynamic> surface_points_index;
	Matrix <double, 1, Dynamic> surface_edges_index;
	Matrix <double, 1, Dynamic> surface_boxels_index;
	map <int, int> surface_edge_directions;

	vector <BBTreeNode> nodes;
	int tree_root;

	void compute_surface();
	void compute_bb_tree(Ref <Matrix <double, 2, Dynamic>> pos, int w, int h);
	void recompute_bbs(Ref <Matrix <double, 2, Dynamic>> pos);
	void pretty_print(int node_index, int depth);

	//BoundingBox bounding_box;

	//vector<int> surface_points_index;
	//vector<int> surface_edges_index;
	//vector<int> surface_boxels_index;

	/*
	void compute_bounding_box(vector<Vector2d_old>* pos);
	Vector2d_old compute_com(vector<Vector2d_old>* pos, vector<double>* mass);
	*/

	SimObject();
	~SimObject();
};

#endif // !SIMOBJECT_H


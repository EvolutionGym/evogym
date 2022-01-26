#include "SimObject.h"

#include <iostream>
#include <float.h>

SimObject::SimObject()
{
	is_robot = false;
}

void SimObject::compute_bb_tree(Ref <Matrix <double, 2, Dynamic>> pos, int w, int h) {
	
	// set globals
	grid_h = h;
	grid_w = w;

	// ERROR CHECK
	if (surface_boxels_index.size() == 0) {
		cout << "ERROR: Object has no surface\n";
		return;
	}

	// INIT

	// create initial nodes for bottom up construction

	map <int, int> grid_index_to_new_index;

	for (int i = 0; i < surface_boxels_index.size(); i++) {
		nodes.push_back(BBTreeNode(surface_boxels_index[i]));
		nodes[i].bbox = boxels[surface_boxels_index[i]].get_bounding_box(pos);
		grid_index_to_new_index[boxels[surface_boxels_index[i]].grid_index] = i;
	}


	// assign neighbors based on grid representation

	for (int i = 0; i < surface_boxels_index.size(); i++) {

		int grid_index = boxels[surface_boxels_index[i]].grid_index;
		int b_index = surface_boxels_index[i];

		if (boxels[b_index].neighbors[TOP] && grid_index_to_new_index.count(grid_index - w) != 0)
			nodes[i].neighbors.insert(grid_index_to_new_index[grid_index - w]);
		if (boxels[b_index].neighbors[BOT] && grid_index_to_new_index.count(grid_index + w) != 0)
			nodes[i].neighbors.insert(grid_index_to_new_index[grid_index + w]);
		if (boxels[b_index].neighbors[LEFT] && grid_index_to_new_index.count(grid_index - 1) != 0)
			nodes[i].neighbors.insert(grid_index_to_new_index[grid_index - 1]);
		if (boxels[b_index].neighbors[RIGHT] && grid_index_to_new_index.count(grid_index + 1) != 0)
			nodes[i].neighbors.insert(grid_index_to_new_index[grid_index + 1]);

		if (nodes[i].neighbors.size() != 0)
			continue;

		if (boxels[b_index].neighbors[TOP_LEFT] && grid_index_to_new_index.count(grid_index - w - 1) != 0)
			nodes[i].neighbors.insert(grid_index_to_new_index[grid_index - w - 1]);
		if (boxels[b_index].neighbors[TOP_RIGHT] && grid_index_to_new_index.count(grid_index - w + 1) != 0)
			nodes[i].neighbors.insert(grid_index_to_new_index[grid_index - w + 1]);
		if (boxels[b_index].neighbors[BOT_LEFT] && grid_index_to_new_index.count(grid_index + w - 1) != 0)
			nodes[i].neighbors.insert(grid_index_to_new_index[grid_index + w - 1]);
		if (boxels[b_index].neighbors[BOT_RIGHT] && grid_index_to_new_index.count(grid_index + w + 1) != 0)
			nodes[i].neighbors.insert(grid_index_to_new_index[grid_index + w + 1]);
	}

	//count total number of neighbors (for self collisions)
	for (int i = 0; i < surface_boxels_index.size(); i++) {

		int grid_index = boxels[surface_boxels_index[i]].grid_index;
		int b_index = surface_boxels_index[i];

		if (grid_index_to_new_index.count(grid_index - w) != 0)
			num_suface_neighbors += 1;
		if (grid_index_to_new_index.count(grid_index + w) != 0)
			num_suface_neighbors += 1;
		if (grid_index_to_new_index.count(grid_index - 1) != 0)
			num_suface_neighbors += 1;
		if (grid_index_to_new_index.count(grid_index + 1) != 0)
			num_suface_neighbors += 1;

		if (grid_index_to_new_index.count(grid_index - w - 1) != 0)
			num_suface_neighbors += 1;
		if (grid_index_to_new_index.count(grid_index - w + 1) != 0)
			num_suface_neighbors += 1;
		if (grid_index_to_new_index.count(grid_index + w - 1) != 0)
			num_suface_neighbors += 1;
		if (grid_index_to_new_index.count(grid_index + w + 1) != 0)
			num_suface_neighbors += 1;

		/*if (boxels[b_index].neighbors[TOP] && grid_index_to_new_index.count(grid_index - w) != 0)
			num_suface_neighbors += 1;
		if (boxels[b_index].neighbors[BOT] && grid_index_to_new_index.count(grid_index + w) != 0)
			num_suface_neighbors += 1;
		if (boxels[b_index].neighbors[LEFT] && grid_index_to_new_index.count(grid_index - 1) != 0)
			num_suface_neighbors += 1;
		if (boxels[b_index].neighbors[RIGHT] && grid_index_to_new_index.count(grid_index + 1) != 0)
			num_suface_neighbors += 1;

		if (boxels[b_index].neighbors[TOP_LEFT] && grid_index_to_new_index.count(grid_index - w - 1) != 0)
			num_suface_neighbors += 1;
		if (boxels[b_index].neighbors[TOP_RIGHT] && grid_index_to_new_index.count(grid_index - w + 1) != 0)
			num_suface_neighbors += 1;
		if (boxels[b_index].neighbors[BOT_LEFT] && grid_index_to_new_index.count(grid_index + w - 1) != 0)
			num_suface_neighbors += 1;
		if (boxels[b_index].neighbors[BOT_RIGHT] && grid_index_to_new_index.count(grid_index + w + 1) != 0)
			num_suface_neighbors += 1;*/
	}

	// debug printing
	//for (int i = 0; i < surface_boxels_index.size(); i++) {

	//	int count = 0;
	//	for (auto index = nodes[i].neighbors.begin(); index != nodes[i].neighbors.end(); ++index) {
	//		cout << boxels[nodes[*index].boxel_index].grid_index << " ";
	//		count += 1;
	//	}
	//	cout << ": " << boxels[nodes[i].boxel_index].grid_index;
	//	cout << "\n";
	//}
	//cout << "\n\n\n";

	// create tress of connected components
	int a_index = -1;
	int b_index = -1;
	double min_area = DBL_MAX;

	while (true) {

		// compute min connected pair
		a_index = -1;
		b_index = -1;
		min_area = DBL_MAX;

		for (int i = 0; i < nodes.size(); i++) {
			
			if (nodes[i].is_finished)
				continue;

			for (auto neighbor = nodes[i].neighbors.begin(); neighbor != nodes[i].neighbors.end(); ++neighbor) {
				
				if (nodes[*neighbor].is_finished)
					continue;

				double new_area = BoundingBox(&nodes[i].bbox, &nodes[*neighbor].bbox).area();
				if (new_area < min_area) {
					min_area = new_area;
					a_index = i;
					b_index = *neighbor;
				}
			}
		}

		// no connected pairs left to merge - merge disconnected components
		if (a_index == -1 || b_index == -1) {

			for (int i = 0; i < nodes.size(); i++) {

				if (nodes[i].is_finished)
					continue;

				for (int j = 0; j < nodes.size(); j++) {

					if (nodes[j].is_finished || i == j)
						continue;

					double new_area = BoundingBox(&nodes[i].bbox, &nodes[j].bbox).area();
					if (new_area < min_area) {
						min_area = new_area;
						a_index = i;
						b_index = j;
					}
				}
			}
		}

		// error check
		if (a_index == -1 || b_index == -1)
			break;

		// merge
		int new_node_index = nodes.size();
		nodes.push_back(BBTreeNode(a_index, b_index));
		nodes[new_node_index].bbox = BoundingBox(&nodes[a_index].bbox, &nodes[b_index].bbox);

		for (auto neighbor = nodes[a_index].neighbors.begin(); neighbor != nodes[a_index].neighbors.end(); ++neighbor)
			nodes[new_node_index].neighbors.insert(*neighbor);
		for (auto neighbor = nodes[b_index].neighbors.begin(); neighbor != nodes[b_index].neighbors.end(); ++neighbor)
			nodes[new_node_index].neighbors.insert(*neighbor);

		// clean
		nodes[a_index].is_finished = true;
		nodes[b_index].is_finished = true;

		nodes[new_node_index].neighbors.erase(a_index);
		nodes[new_node_index].neighbors.erase(b_index);

		for (auto neighbor = nodes[new_node_index].neighbors.begin(); neighbor != nodes[new_node_index].neighbors.end(); ++neighbor) {
			nodes[*neighbor].neighbors.erase(a_index);
			nodes[*neighbor].neighbors.erase(b_index);
			nodes[*neighbor].neighbors.insert(new_node_index);
		}
	}
	
	tree_root = nodes.size() - 1;

	//pretty_print(tree_root, 0);


}

void SimObject::recompute_bbs(Ref <Matrix <double, 2, Dynamic>> pos) {
	
	for (int i = 0; i < nodes.size(); i++) {
		if (nodes[i].is_leaf)
			nodes[i].bbox = boxels[nodes[i].boxel_index].get_bounding_box(pos);
		else
			nodes[i].bbox = BoundingBox(&nodes[nodes[i].a_index].bbox, &nodes[nodes[i].b_index].bbox);
	}
}

void SimObject::pretty_print(int node_index, int depth) {
	
	for (int i = 0; i < depth; i++)
		cout << " |   ";

	if (nodes[node_index].is_leaf)
		cout << node_index << ": (L" << boxels[nodes[node_index].boxel_index].grid_index << ")\n";
	else {
		cout << node_index << ": [" << nodes[node_index].bbox.area() <<"] \n";
		pretty_print(nodes[node_index].a_index, depth + 1);
		pretty_print(nodes[node_index].b_index, depth + 1);
	}
		
}

void SimObject::compute_surface() {

	unordered_set<int> points;
	unordered_set<int> edgesloc;
	unordered_set<int> boxelsloc;

	for (int i = 0; i < boxels.size(); i++) {

		if (!boxels[i].neighbors[TOP]) {
			points.insert(boxels[i].point_top_left_index);
			points.insert(boxels[i].point_top_right_index);
			edgesloc.insert(boxels[i].edge_top_index);
			boxelsloc.insert(i);
			surface_edge_directions[boxels[i].edge_top_index] = TOP;
			/*if (Vector2d_old().dot(Vector2d_old(0, 1), edges->at(boxels[i].edge_top_index).get_normal(pos)) < 0) {
				edges->at(boxels[i].edge_top_index).swap();
			}*/
		}
		if (!boxels[i].neighbors[BOT]) {
			points.insert(boxels[i].point_bot_left_index);
			points.insert(boxels[i].point_bot_right_index);
			edgesloc.insert(boxels[i].edge_bot_index);
			boxelsloc.insert(i);
			surface_edge_directions[boxels[i].edge_bot_index] = BOT;
			/*if (Vector2d_old().dot(Vector2d_old(0, -1), edges->at(boxels[i].edge_bot_index).get_normal(pos)) < 0) {
				edges->at(boxels[i].edge_bot_index).swap();
			}*/
		}
		if (!boxels[i].neighbors[LEFT]) {
			points.insert(boxels[i].point_top_left_index);
			points.insert(boxels[i].point_bot_left_index);
			edgesloc.insert(boxels[i].edge_left_index);
			boxelsloc.insert(i);
			surface_edge_directions[boxels[i].edge_left_index] = LEFT;
			/*if (Vector2d_old().dot(Vector2d_old(-1, 0), edges->at(boxels[i].edge_left_index).get_normal(pos)) < 0) {
				edges->at(boxels[i].edge_left_index).swap();
			}*/
		}
		if (!boxels[i].neighbors[RIGHT]) {
			points.insert(boxels[i].point_top_right_index);
			points.insert(boxels[i].point_bot_right_index);
			edgesloc.insert(boxels[i].edge_right_index);
			boxelsloc.insert(i);
			surface_edge_directions[boxels[i].edge_right_index] = RIGHT;
			/*if (Vector2d_old().dot(Vector2d_old(1, 0), edges->at(boxels[i].edge_right_index).get_normal(pos)) < 0) {
				edges->at(boxels[i].edge_right_index).swap();
			}*/
		}
	}

	int count;

	surface_points_index.resize(points.size());
	count = 0;
	for (auto index = points.begin(); index != points.end(); ++index) {
		surface_points_index[count] = *index;
		count += 1;
	}

	surface_edges_index.resize(edgesloc.size());
	count = 0;
	for (auto index = edgesloc.begin(); index != edgesloc.end(); ++index) {
		surface_edges_index[count] = *index;
		count += 1;
	}

	surface_boxels_index.resize(boxelsloc.size());
	count = 0;
	for (auto index = boxelsloc.begin(); index != boxelsloc.end(); ++index) {
		surface_boxels_index[count] = *index;
		count += 1;
	}
}

/*
void SimObject::compute_bounding_box(vector<Vector2d_old>* pos) {

	if (surface_points_index.size() == 0) {
		cout << "WARNING: Bounding box cannot be computed because sim object has no surface." << "\n";
		return;
	}

	bounding_box.top_left.x() = pos->at(surface_points_index[0]).x();
	bounding_box.top_left.y() = pos->at(surface_points_index[0]).y();
	bounding_box.bot_right.x() = pos->at(surface_points_index[0]).x();
	bounding_box.bot_right.y() = pos->at(surface_points_index[0]).y();

	for (int i = 0; i < surface_points_index.size(); i++) {

		if (pos->at(surface_points_index[i]).x() < bounding_box.top_left.x())
			bounding_box.top_left.x() = pos->at(surface_points_index[i]).x();

		if (pos->at(surface_points_index[i]).y() < bounding_box.top_left.y())
			bounding_box.top_left.y() = pos->at(surface_points_index[i]).y();

		if (pos->at(surface_points_index[i]).x() > bounding_box.bot_right.x())
			bounding_box.bot_right.x() = pos->at(surface_points_index[i]).x();

		if (pos->at(surface_points_index[i]).y() > bounding_box.bot_right.y())
			bounding_box.bot_right.y() = pos->at(surface_points_index[i]).y();
	}
}


Vector2d_old SimObject::compute_com(vector<Vector2d_old>* pos, vector<double>* mass) {

	Vector2d_old sum = Vector2d_old(0.0, 0.0);
	double mass_sum = 0;

	unordered_set <int> points_index;
	for (int i = 0; i < boxels.size(); i++) {

		if (points_index.find(boxels[i].point_top_left_index) == points_index.end()) {
			sum += pos->at(boxels[i].point_top_left_index) * mass->at(boxels[i].point_top_left_index);
			mass_sum += mass->at(boxels[i].point_top_left_index);
			points_index.insert(boxels[i].point_top_left_index);
		}

		if (points_index.find(boxels[i].point_top_right_index) == points_index.end()) {
			sum += pos->at(boxels[i].point_top_right_index) * mass->at(boxels[i].point_top_right_index);
			mass_sum += mass->at(boxels[i].point_top_right_index);
			points_index.insert(boxels[i].point_top_right_index);
		}

		if (points_index.find(boxels[i].point_bot_right_index) == points_index.end()) {
			sum += pos->at(boxels[i].point_bot_right_index) * mass->at(boxels[i].point_bot_right_index);
			mass_sum += mass->at(boxels[i].point_bot_right_index);
			points_index.insert(boxels[i].point_bot_right_index);
		}

		if (points_index.find(boxels[i].point_bot_left_index) == points_index.end()) {
			sum += pos->at(boxels[i].point_bot_left_index) * mass->at(boxels[i].point_bot_left_index);
			mass_sum += mass->at(boxels[i].point_bot_left_index);
			points_index.insert(boxels[i].point_bot_left_index);
		}
	}

	if (mass_sum == 0)
		return NULL;

	return sum/mass_sum;
}
*/

SimObject::~SimObject()
{
}

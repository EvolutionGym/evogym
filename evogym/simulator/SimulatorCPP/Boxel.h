#ifndef BOXEL_H
#define BOXEL_H

#include <Eigen/Dense>
#include <vector>
#include "main.h"
#include "Edge.h"

using namespace std;
using namespace Eigen;

class Boxel
{

private:

	/*bool point_in_subtriangle(Vector2d_old param);*/

public:

	int point_top_right_index;
	int point_top_left_index;
	int point_bot_right_index;
	int point_bot_left_index;

	int edge_top_index;
	int edge_bot_index;
	int edge_left_index;
	int edge_right_index;

	int grid_index;

	//vector<int*> points;
	//vector<int*> edges;

	Vector4f points;
	Vector4f edges;

	int cell_type;

	Matrix <bool, 8, 1> neighbors;

	bool is_colliding = false;

	//Matrix_old bary_matrix_1;
	//Matrix_old bary_matrix_2;
	//bool det_is_zero_1;
	//bool det_is_zero_2;
	//Vector2d_old ref_point;

	Boxel(int cell_type, int grid_index);
	~Boxel();

	void init();
	BoundingBox get_bounding_box(Ref <Matrix <double, 2, Dynamic>> pos);
	bool point_in_boxel(Vector2d p, Ref <Matrix <double, 2, Dynamic>> pos);
	//void compute_bary_matricies(vector <Vector2d_old>* pos);
	//bool point_in_boxel(Vector2d_old p);
};

#endif // !BOXEL_H


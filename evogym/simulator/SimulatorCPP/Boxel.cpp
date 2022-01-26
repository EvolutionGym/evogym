#include "Boxel.h"
#include <iostream>

Boxel::Boxel(int cell_type, int grid_index)
{
	point_top_right_index = NULL;
	point_top_left_index = NULL;
	point_bot_right_index = NULL;
	point_bot_left_index = NULL;

	edge_top_index = NULL;
	edge_bot_index = NULL;
	edge_left_index = NULL;
	edge_right_index = NULL;

	neighbors << false, false, false, false, false, false, false, false;

	Boxel::cell_type = cell_type;
	Boxel::grid_index = grid_index;
}

void Boxel::init() {

	points = Vector4f(point_top_left_index, point_top_right_index, point_bot_right_index, point_bot_left_index);
	edges = Vector4f(edge_top_index, edge_right_index, edge_bot_index, edge_left_index);

	/*
	points = { &point_top_left_index, &point_top_right_index, &point_bot_right_index, &point_bot_left_index };
	edges = { &edge_top_index, &edge_right_index, &edge_bot_index, &edge_left_index };*/
}
BoundingBox Boxel::get_bounding_box(Ref <Matrix <double, 2, Dynamic>> pos) {

	Matrix <double, 2, 4> points_pos = pos(Eigen::all, points);
	return BoundingBox(points_pos.rowwise().minCoeff(), points_pos.rowwise().maxCoeff());
}

double cross2f(Vector2d* a, Vector2d* b) {
	return a->x() * b->y() - a->y() * b->x();
}

bool Boxel::point_in_boxel(Vector2d p, Ref <Matrix <double, 2, Dynamic>> pos) {

	Vector2d tl = pos.col(point_top_left_index) - p;
	Vector2d tr = pos.col(point_top_right_index) - p;
	Vector2d bl = pos.col(point_bot_left_index) - p;
	Vector2d br = pos.col(point_bot_right_index) - p;

	//Triangle: TL --> TR --> BR
	double a = cross2f(&tl, &tr);
	double b = cross2f(&tr, &br);
	double c = cross2f(&br, &tl);

	double tol = 0.0f;

	if (c*a >= 0.0f - tol && c*b >= 0.0f - tol)
		return true;


	//Triangle: TL <-- BR <-- BL
	double d = cross2f(&tl, &bl);
	double e = cross2f(&bl, &br);

	return (c*d >= 0.0f - tol && c*e >= 0.0f - tol);

}

//void Boxel::compute_bary_matricies(vector <Vector2d_old>* pos) {
//
//	//Triangle: TL --> TR --> BR
//	Matrix_old bcoords_1 = Matrix_old((pos->at(point_bot_right_index) - pos->at(point_top_left_index)), (pos->at(point_top_right_index) - pos->at(point_top_left_index)));
//	det_is_zero_1 = false;
//	bary_matrix_1 = bcoords_1.inverse(&det_is_zero_1, 0.000001);
//
//	//Triangle: TL --> BR --> BL
//	Matrix_old bcoords_2 = Matrix_old((pos->at(point_bot_left_index) - pos->at(point_top_left_index)), (pos->at(point_bot_right_index) - pos->at(point_top_left_index)));
//	det_is_zero_2 = false;
//	bary_matrix_2 = bcoords_2.inverse(&det_is_zero_2, 0.000001);
//
//	ref_point = pos->at(point_top_left_index);
//}
//
//bool Boxel::point_in_boxel(Vector2d_old p) {
//
//	Vector2d_old test_vec = p - ref_point;
//
//	return ((!det_is_zero_1 && point_in_subtriangle(bary_matrix_1.times(test_vec))) || 
//			(!det_is_zero_2 && point_in_subtriangle(bary_matrix_2.times(test_vec))));
//}
//
//bool Boxel::point_in_subtriangle(Vector2d_old parametric_representation) {
//	
//	if (parametric_representation.x() < 0.0 || parametric_representation.y() < 0.0)
//		return false;
//
//	if (parametric_representation.x() > 1.0 || parametric_representation.y() > 1.0)
//		return false;
//
//	if (parametric_representation.x() + parametric_representation.y() > 1.0)
//		return false;
//
//	return true;
//}

Boxel::~Boxel()
{
}

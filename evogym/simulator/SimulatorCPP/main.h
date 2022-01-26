#ifndef MAIN_H
#define MAIN_H	

#include "GL/glew.h"
#include "glfw3.h"

//#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
//namespace py = pybind11;

//#include <windows.h>
//#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Dense>
#include <Eigen/Core>

//#include <vector>
#include <iostream>

using namespace std;
using namespace Eigen;

//struct BoundingBox {
//	Vector2d_old top_left;
//	Vector2d_old bot_right;
//};

//struct Pair {
//	int a;
//	int b;
//};

enum coord_identifier { VAL_X = 0, VAL_Y = 1 };
enum direction { TOP = 0, BOT = 1, LEFT = 2, RIGHT = 3, TOP_LEFT = 4, TOP_RIGHT = 5, BOT_LEFT = 6, BOT_RIGHT = 7 };
enum cell_type { CELL_EMPTY = 0, CELL_RIGID, CELL_SOFT, CELL_ACT_H, CELL_ACT_V, CELL_FIXED};

const double F_ERROR_TOL = 0.0f;

class BoundingBox {

public:
	
	Vector2d min_pos;
	Vector2d max_pos;

	BoundingBox() {

	}

	BoundingBox(Vector2d min_pos, Vector2d max_pos) {
		BoundingBox::min_pos = min_pos;
		BoundingBox::max_pos = max_pos;
	}

	BoundingBox(BoundingBox* a, BoundingBox* b) {
		min_pos = a->min_pos.cwiseMin(b->min_pos);
		max_pos = a->max_pos.cwiseMax(b->max_pos);
	}

	BoundingBox(const BoundingBox &a) {
		min_pos = a.min_pos;
		max_pos = a.max_pos;
	}

	bool in(BoundingBox* a) {

		if (min_pos.x() > a->max_pos.x() + F_ERROR_TOL || F_ERROR_TOL + max_pos.x() < a->min_pos.x())
			return false;
		if (min_pos.y() > a->max_pos.y() + F_ERROR_TOL || F_ERROR_TOL + max_pos.y() < a->min_pos.y())
			return false;

		return true;
	}

	double area() {
		return (max_pos[0] - min_pos[0]) * (max_pos[1] - min_pos[1]);
	}

	void print() {
		cout << "\n[" << min_pos << "]" << "\n";
		cout << "[" << max_pos << "]" << "\n";
	}

};

//class Matrix_old {
//
//public:
//
//	double a;
//	double b;
//	double c;
//	double d;
//
//	Matrix_old() {
//
//		Matrix_old::a = 0;
//		Matrix_old::b = 0;
//		Matrix_old::c = 0;
//		Matrix_old::d = 0;
//	}
//
//	Matrix_old(double a, double b, double c, double d) {
//
//		Matrix_old::a = a;
//		Matrix_old::b = b;
//		Matrix_old::c = c;
//		Matrix_old::d = d;
//	}
//
//	Matrix_old(Vector2d_old col1, Vector2d_old col2) {
//
//		Matrix_old::a = col1.x();
//		Matrix_old::b = col2.x();
//		Matrix_old::c = col1.y();
//		Matrix_old::d = col2.y();
//	}
//
//	Vector2d_old times(Vector2d_old arg) {
//
//		Vector2d_old out = Vector2d_old(0, 0);
//
//		out.x() = a * arg.x() + b * arg.y();
//		out.y() = c * arg.x() + d * arg.y();
//
//		return out;
//	}
//
//	Matrix_old times(Matrix_old arg) {
//
//		Matrix_old out = Matrix_old(0, 0);
//
//		out.a = a * arg.a + b * arg.c;
//		out.b = a * arg.b + b * arg.d;
//		out.c = c * arg.a + d * arg.c;
//		out.d = c * arg.b + d * arg.d;
//
//		return out;
//	}
//
//	Matrix_old inverse(bool* det_fail, double eps) {
//
//		double det = a * d - b * c;
//
//		if (abs(det) < eps) {
//			*det_fail = true;
//			return Matrix_old();
//		}
//		*det_fail = false;
//
//
//		double na = d / det;
//		double nb = -b / det;
//		double nc = -c / det;
//		double nd = a / det;
//
//		return Matrix_old(na, nb, nc, nd);
//	}
//
//	void print() {
//		cout << "[" << a << ", " << b << "]" << "\n";
//		cout << "[" << c << ", " << d << "]" << "\n";
//	}
//
//};


#endif // !MAIN_H


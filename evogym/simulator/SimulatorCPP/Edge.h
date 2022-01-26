#ifndef EDGE_H
#define EDGE_H

#include "main.h"
#include <vector>

using namespace std;

class Edge
{
public:

	int a_index;
	int b_index;

	double act_length_eq;
	int num_actuating;
	double init_length_eq;

	double spring_const;
	int color;

	bool isColliding;
	bool isOnSurface;

	Edge(int a_index, int b_index, double length_eq, double spring_const, int color = 0);
	~Edge();

	Vector2d get_normal(Ref< Matrix <double, 2, Dynamic>> pos);
	void swap();
};

#endif // !EDGE_H




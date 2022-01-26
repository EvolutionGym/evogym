#include "Edge.h"

Edge::Edge(int a_index, int b_index, double length_eq, double spring_const, int color)
{
	Edge::a_index = a_index;
	Edge::b_index = b_index;
	Edge::init_length_eq = length_eq;
	Edge::spring_const = spring_const;
	Edge::color = color;
	Edge::isColliding = false;
	Edge::num_actuating = 0;
	Edge::act_length_eq = init_length_eq;
	Edge::isOnSurface = false;
}

Vector2d Edge::get_normal(Ref< Matrix <double, 2, Dynamic>> pos) {
	
	Vector2d slope = pos.col(b_index) - pos.col(a_index);
	Vector2d normal = Vector2d(-slope.y(), slope.x());
	
	return normal.normalized();
}

void Edge::swap() {
	int temp = a_index;
	a_index = b_index;
	b_index = temp;
}

Edge::~Edge()
{
}

#ifndef ROBOT_H
#define ROBOT_H

#include "main.h"

#include <vector>
#include <map>

#include "SimObject.h"
#include "Boxel.h"

using namespace std;

class Robot : public SimObject
{
private:

	Matrix <int, 1, Dynamic> actuator_indicies;
	map <int, int> grid_index_to_boxel_index;

public:

	void init();
	Boxel* get_actuator(int grid_index);

	Ref <MatrixXi> get_actuator_indicies();

	Robot();
	~Robot();
};

#endif // !ROBOT_H



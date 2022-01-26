#include "Robot.h"



Robot::Robot()
{
}

void Robot::init() {

	vector<int> indicies;
	for (int i = 0; i < boxels.size(); i++) {
		if (boxels[i].cell_type == CELL_ACT_H || boxels[i].cell_type == CELL_ACT_V) {
			grid_index_to_boxel_index[boxels[i].grid_index] = i;
			indicies.push_back(boxels[i].grid_index);
		}
	}
	
	actuator_indicies.resize(1, indicies.size());
	for (int i = 0; i < indicies.size(); i++)
		actuator_indicies[i] = indicies[i];
}

Ref <MatrixXi> Robot::get_actuator_indicies() {
	return actuator_indicies;
}

Boxel* Robot::get_actuator(int grid_index) {

	if (grid_index_to_boxel_index.find(grid_index) == grid_index_to_boxel_index.end()) {
		cout << "Error: Robot action inproperly defined - no actuator found at index.";
		throw;
	}
	
	return &boxels[grid_index_to_boxel_index[grid_index]];
}


Robot::~Robot()
{
}

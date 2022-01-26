#ifndef SNAPSHOT_H
#define SNAPSHOT_H


#include <vector>
#include "main.h"
#include "SimObject.h"

using namespace std;

class Snapshot
{
public:

	Matrix <double, 2, Dynamic> points_pos;
	Matrix <double, 2, Dynamic> points_vel;

	long int sim_time;

	Snapshot();

	Snapshot(long int sim_time);
	Snapshot(Ref <Matrix <double, 2, Dynamic>> points_pos, Ref <Matrix <double, 2, Dynamic>> points_vel, long int sim_time);
	Snapshot(const Snapshot& s);
	
	void set_data(Ref <Matrix <double, 2, Dynamic>> points_pos, Ref <Matrix <double, 2, Dynamic>> points_vel);

	~Snapshot();
};

#endif // !SNAPSHOT_H



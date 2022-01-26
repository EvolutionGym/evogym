#include "Snapshot.h"



Snapshot::Snapshot()
{
}

Snapshot::Snapshot(long int sim_time) {
	Snapshot::sim_time = sim_time;
}

Snapshot::Snapshot(Ref <Matrix <double, 2, Dynamic>> points_pos, Ref <Matrix <double, 2, Dynamic>> points_vel, long int sim_time){
	Snapshot::points_pos = points_pos;
	Snapshot::points_vel = points_vel;
	Snapshot::sim_time = sim_time;
}
Snapshot::Snapshot(const Snapshot& s) {
	Snapshot::points_pos = s.points_pos;
	Snapshot::points_vel = s.points_vel;
	Snapshot::sim_time = s.sim_time;
}

void Snapshot::set_data(Ref <Matrix <double, 2, Dynamic>> points_pos, Ref <Matrix <double, 2, Dynamic>> points_vel){
	Snapshot::points_pos = points_pos;
	Snapshot::points_vel = points_vel;
}

Snapshot::~Snapshot()
{
}

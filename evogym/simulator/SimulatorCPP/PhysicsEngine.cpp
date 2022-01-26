#include "PhysicsEngine.h"

#include <iostream>
#include <set>
#include <map>
#include <unordered_set>

PhysicsEngine::PhysicsEngine()
{
}

void PhysicsEngine::init(
	Matrix <double, 2, Dynamic>* pos, Matrix <double, 2, Dynamic>* pos_last, Matrix <double, 2, Dynamic>* vel, Matrix <double, 2, Dynamic>* vel_true,
	Matrix <double, 2, Dynamic>* mass, Matrix <bool, 2, Dynamic>* fixed,
	Matrix <int, 1, Dynamic>* a_index, Matrix <int, 1, Dynamic>* b_index, Matrix <double, 1, Dynamic>* length_eq, Matrix <double, 1, Dynamic>* length_eq_goal,
	Matrix <double, 1, Dynamic>* init_length_eq, Matrix <double, 1, Dynamic>* spring_const, vector<Edge>* edges,
	vector<SimObject*>* objects){

	PhysicsEngine::pos = pos;
	PhysicsEngine::pos_last = pos_last;
	PhysicsEngine::vel = vel;
	PhysicsEngine::vel_true = vel_true;
	PhysicsEngine::mass = mass;
	PhysicsEngine::fixed = fixed;

	PhysicsEngine::a_index = a_index;
	PhysicsEngine::b_index = b_index;
	PhysicsEngine::length_eq = length_eq;
	PhysicsEngine::length_eq_goal = length_eq_goal;
	PhysicsEngine::init_length_eq = init_length_eq;
	PhysicsEngine::spring_const = spring_const;

	PhysicsEngine::edges = edges;
	PhysicsEngine::objects = objects;

}

void PhysicsEngine::update_actuator_goals(vector<Boxel*>* actuators, vector<double>* actuations){

	unordered_set<int> updated_edge_index;
	map <int, int> edge_index_to_act_count;

	for (int i = 0; i < actuators->size(); i++) {
		
		Boxel* current = actuators->at(i);

		if (current->cell_type == CELL_ACT_H) {

			length_eq_goal->array()[current->edge_top_index] = 0;
			length_eq_goal->array()[current->edge_bot_index] = 0;

			edge_index_to_act_count[current->edge_top_index] = 0;
			edge_index_to_act_count[current->edge_bot_index] = 0;
			
			updated_edge_index.insert(current->edge_top_index);
			updated_edge_index.insert(current->edge_bot_index);
		}
		if (current->cell_type == CELL_ACT_V) {

			length_eq_goal->array()[current->edge_left_index] = 0;
			length_eq_goal->array()[current->edge_right_index] = 0;

			edge_index_to_act_count[current->edge_left_index] = 0;
			edge_index_to_act_count[current->edge_right_index] = 0;

			updated_edge_index.insert(current->edge_left_index);
			updated_edge_index.insert(current->edge_right_index);
		}
	}

	for (int i = 0; i < actuators->size(); i++) {

		Boxel* current = actuators->at(i);
		double actuation = actuations->at(i);

		if (current->cell_type == CELL_ACT_H) {
			
			length_eq_goal->array()[current->edge_top_index] += actuation;
			length_eq_goal->array()[current->edge_bot_index] += actuation;

			edge_index_to_act_count[current->edge_top_index] += 1;
			edge_index_to_act_count[current->edge_bot_index] += 1;
		}
		if (current->cell_type == CELL_ACT_V) {

			length_eq_goal->array()[current->edge_left_index] += actuation;
			length_eq_goal->array()[current->edge_right_index] += actuation;

			edge_index_to_act_count[current->edge_left_index] += 1;
			edge_index_to_act_count[current->edge_right_index] += 1;
		}
	}

	for (auto index = updated_edge_index.begin(); index != updated_edge_index.end(); ++index) {
		length_eq_goal->array()[*index] *= edges->at(*index).init_length_eq/edge_index_to_act_count[*index];
	}
}

void PhysicsEngine::update_actuators() {

	double convergence_factor = 0.006;
	double max_additive = 0.005;

	Array <double, 2, Dynamic> vec_q_to_p = pos->matrix()(Eigen::all, *a_index) - pos->matrix()(Eigen::all, *b_index);
	Array <double, 1, Dynamic> dist = (vec_q_to_p.row(0).square() + vec_q_to_p.row(1).square()).sqrt();

	Array <double, 1, Dynamic> additive = (length_eq_goal->array() - dist)*convergence_factor;
	
	//additive = (additive > max_additive).select(max_additive, additive);
	//additive = (additive < -max_additive).select(-max_additive, additive);


	//cout << "min " << additive.abs().minCoeff() << " | mean " << additive.abs().mean() << " | max " << additive.abs().maxCoeff() << "\n";

	*length_eq = dist + additive;

	//*length_eq = length_eq_goal->array()*convergence_factor + dist * (1-convergence_factor);

}

//bool PhysicsEngine::bounding_box_1_in_2(BoundingBox bbox1, BoundingBox bbox2) {
//
//	if (bbox1.top_left.x() > bbox2.top_left.x() && bbox1.top_left.x() < bbox2.bot_right.x()) {
//		if (bbox1.top_left.y() > bbox2.top_left.y() && bbox1.top_left.y() < bbox2.bot_right.y()) {
//			return true;
//		}
//		if (bbox1.bot_right.y() > bbox2.top_left.y() && bbox1.bot_right.y() < bbox2.bot_right.y()) {
//			return true;
//		}
//	}
//
//	if (bbox1.bot_right.x() > bbox2.top_left.x() && bbox1.bot_right.x() < bbox2.bot_right.x()) {
//		if (bbox1.top_left.y() > bbox2.top_left.y() && bbox1.top_left.y() < bbox2.bot_right.y()) {
//			return true;
//		}
//		if (bbox1.bot_right.y() > bbox2.top_left.y() && bbox1.bot_right.y() < bbox2.bot_right.y()) {
//			return true;
//		}
//	}
//
//	return false;
//}
//
//void PhysicsEngine::resolve_collisions(){
//	
//	//BOUNDING BOX COLLISIONS
//
//	//compute object bounding boxes
//	for (int i = 0; i < objects->size(); i++) {
//		objects->at(i)->compute_bounding_box(pos_old);
//	}
//
//	//get a list/set of potentially colliding objects
//	struct CollisionPair {
//		int index1;
//		int index2;
//	};
//
//	vector<CollisionPair> problem_object_pairs;
//	set <int> problem_objects;
//
//	for (int i = 0; i < objects->size(); i++) {
//		for (int j = i+1; j < objects->size(); j++) {
//
//			auto bbox1 = objects->at(i)->bounding_box;
//			auto bbox2 = objects->at(j)->bounding_box;
//
//			if (bounding_box_1_in_2(bbox1, bbox2) || bounding_box_1_in_2(bbox2, bbox1)) {
//				
//				problem_object_pairs.push_back(CollisionPair());
//				problem_object_pairs.back().index1 = i;
//				problem_object_pairs.back().index2 = j;
//
//				problem_objects.insert(i);
//				problem_objects.insert(j);
//				continue;
//			}
//		}
//	}
//
//	//FINER COLLISONS
//
//	//set up computations
//	for (int i = 0; i < edges->size(); i++) {
//		edges->at(i).isColliding = false;
//	}
//
//	collision_forces.clear();
//	point_is_colliding->clear();
//
//	for (int i = 0; i < pos_old->size(); i++) {
//
//		collision_forces.push_back(Vector2d_old(0, 0));
//		point_is_colliding->push_back(false);
//	}
//
//	for (auto index = problem_objects.begin(); index != problem_objects.end(); ++index) {
//		for (int i = 0; i < objects->at(*index)->boxels.size(); i++) {
//			objects->at(*index)->boxels.at(i).compute_bary_matricies(pos_old);
//		}
//	}
//
//	//resolve finer collisions
//	for (int i = 0; i < problem_object_pairs.size(); i++) {
//		
//		//detect_resolve_fine_collisions(problem_object_pairs[i].index1, problem_object_pairs[i].index2);
//		resolve_object_collisions(problem_object_pairs[i].index1, problem_object_pairs[i].index2);
//		//cout << problem_objects[i].index1 << " " << problem_objects[i].index2 << "\n";
//	}
//	//cout << "\n";
//}
//
//void PhysicsEngine::resolve_object_collisions(int object_1_index, int object_2_index) {
//
//	for (int i = 0; i < objects->at(object_1_index)->surface_boxels_index.size(); i++) {
//		for (int j = 0; j < objects->at(object_2_index)->surface_boxels_index.size(); j++) {
//
//			int boxel_1_index = objects->at(object_1_index)->surface_boxels_index.at(i);
//			int boxel_2_index = objects->at(object_2_index)->surface_boxels_index.at(j);
//			resolve_boxel_collisions(objects->at(object_1_index)->boxels.at(boxel_1_index), objects->at(object_2_index)->boxels.at(boxel_2_index));
//			resolve_boxel_collisions(objects->at(object_2_index)->boxels.at(boxel_2_index), objects->at(object_1_index)->boxels.at(boxel_1_index));
//		}
//	}
//}
//
//void PhysicsEngine::resolve_boxel_collisions(Boxel main, Boxel ref) {
//	
//	for (int i = 0; i < main.points.size(); i++) {
//
//		if (ref.point_in_boxel(pos_old->at(*main.points.at(i)))) {
//			point_is_colliding->at(*main.points.at(i)) = true;
//			resolve_point_boxel_collisions(*main.points.at(i), main, ref);
//		}
//	}
//}
//
//void PhysicsEngine::resolve_point_boxel_collisions(int p_index, Boxel main, Boxel ref) {
//
//	vector <int> main_edges_index;
//	vector <int> ref_edges_index;
//
//	//find all relavent edges
//
//	if (p_index == main.point_top_left_index) {
//		main_edges_index.push_back(main.edge_top_index);
//		main_edges_index.push_back(main.edge_left_index);
//	}
//	if (p_index == main.point_top_right_index) {
//		main_edges_index.push_back(main.edge_top_index);
//		main_edges_index.push_back(main.edge_right_index);
//	}
//	if (p_index == main.point_bot_left_index) {
//		main_edges_index.push_back(main.edge_bot_index);
//		main_edges_index.push_back(main.edge_left_index);
//	}
//	if (p_index == main.point_bot_right_index) {
//		main_edges_index.push_back(main.edge_bot_index);
//		main_edges_index.push_back(main.edge_right_index);
//	}
//
//	for (int i = 0; i < main_edges_index.size(); i++) {
//		
//		ref_edges_index.push_back(-1);
//		for (int j = 0; j < ref.edges.size(); j++) {
//			if (get_intersecting(main_edges_index.at(i), *ref.edges.at(j))) {
//				ref_edges_index.at(i) = *ref.edges.at(j);
//				break;
//			}
//		}
//	}
//
//	// error check
//
//	int first_good_index = -1;
//	for (int i = 0; i < main_edges_index.size(); i++) {
//
//		if (ref_edges_index.at(i) == -1)
//			continue;
//
//		first_good_index = i;
//		break;
//	}
//
//	if (first_good_index == -1) {
//		//cout << "bad " << main_edges_index.size() << "\n";
//		return;
//	}
//
//	// find closest edge
//
//	int index_of_min_dist = first_good_index;
//	double min_dist = dist_point_edge(p_index, ref_edges_index.at(first_good_index));
//
//	for (int i = 0; i < main_edges_index.size(); i++) {
//		
//		if (ref_edges_index.at(i) == -1)
//			continue;
//
//		double dist = dist_point_edge(p_index, ref_edges_index.at(i));
//		if (dist < min_dist) {
//			min_dist = dist;
//			index_of_min_dist = i;
//		}
//	}
//
//	//set normal forces
//
//	edges->at(ref_edges_index.at(index_of_min_dist)).isColliding = true;
//
//	Vector2d_old normal_force = edges->at(ref_edges_index.at(index_of_min_dist)).get_normal(pos_old) * collision_const_obj * (collision_base_dist_additive + min_dist);
//	collision_forces[p_index] += normal_force;
//	collision_forces[edges->at(ref_edges_index.at(index_of_min_dist)).a_index] -= normal_force / 2;
//	collision_forces[edges->at(ref_edges_index.at(index_of_min_dist)).b_index] -= normal_force / 2;
//	//collision_forces[p_index] += dist_point_edge(p_index, ref_edges_index.at(i)) * edges->at(ref_edges_index.at(i)).get_normal(pos_old) * collision_const_obj;
//
//
//	//set friction/tangential forces
//	
//	Vector2d_old normal = edges->at(ref_edges_index.at(index_of_min_dist)).get_normal(pos_old);
//	
//	Vector2d_old unit_tangent = normal.normalized();
//	unit_tangent = Vector2d_old(-unit_tangent.y(), unit_tangent.x());
//
//	double projected_vel_mag = unit_tangent.x() * vel_old->at(p_index).x() + unit_tangent.y() * vel_old->at(p_index).y();
//	double normal_force_mag = normal_force.abs();
//
//	if (projected_vel_mag < 0) {
//		projected_vel_mag *= -1;
//		unit_tangent *= -1;
//	}
//
//	Vector2d_old friction_force = 150 * unit_tangent * (-1) * dynamic_friction_const *  normal_force_mag * tanh(projected_vel_mag / (dynamic_friction_const *  normal_force_mag));
//	//cout << friction_force.x() << " " << friction_force.y() << "\n";
//	//cout << normal_force_mag << "\n\n\n";
//	collision_forces[p_index] += friction_force;
//	collision_forces[edges->at(ref_edges_index.at(index_of_min_dist)).a_index] -= friction_force / 2;
//	collision_forces[edges->at(ref_edges_index.at(index_of_min_dist)).b_index] -= friction_force / 2;
//	//cout << normal.x() << " " << normal.y() << " " << unit_tangent.x() << " " << unit_tangent.y() << "\n";
//
//}
//
//bool PhysicsEngine::get_intersecting(int edgeIndex1, int edgeIndex2) {
//
//	Edge* edge1 = &(edges->at(edgeIndex1));
//	Edge* edge2 = &(edges->at(edgeIndex2));
//
//	Vector2d_old diff1 = pos_old->at(edge1->b_index) - pos_old->at(edge1->a_index);
//	Vector2d_old diff2 = pos_old->at(edge2->b_index) - pos_old->at(edge2->a_index);
//
//	double det = -diff1.x()*diff2.y() + diff2.x()*diff1.y();
//
//	//edges are parallel
//	if (abs(det) < 0.0001)
//		return false;
//
//	double x_diff = pos_old->at(edge2->a_index).x() - pos_old->at(edge1->a_index).x();
//	double y_diff = pos_old->at(edge2->a_index).y() - pos_old->at(edge1->a_index).y();
//
//	//We view the edges as lines paramaterized by s and t.
//	//If the point of intersection has the property that 0 < t < 1 and 0 < s < 1, there is a collision.
//
//	double t = (-diff2.y()*x_diff + diff2.x()*y_diff) / det;
//	double s = (-diff1.y()*x_diff + diff1.x()*y_diff) / det;
//
//	if (0 < t && t < 1 && 0 < s && s < 1)
//		return true;
//
//	return false;
//}
//
//bool PhysicsEngine::get_intersecting_fast(int edgeIndex1, int edgeIndex2) {
//
//	//Implement later with x and y projections.
//	return false;
//}
//
//void PhysicsEngine::detect_resolve_fine_collisions(int object_index1, int object_index2) {
//
//	for (int i = 0; i < objects->at(object_index1)->surface_edges_index.size(); i++) {
//		for (int j = 0; j < objects->at(object_index2)->surface_edges_index.size(); j++) {
//			
//			if (get_intersecting(objects->at(object_index1)->surface_edges_index.at(i), objects->at(object_index2)->surface_edges_index.at(j))) {
//				
//				resolve_edge_collisions(objects->at(object_index1)->surface_edges_index.at(i), objects->at(object_index2)->surface_edges_index.at(j));
//
//				edges->at(objects->at(object_index1)->surface_edges_index.at(i)).isColliding = true;
//				edges->at(objects->at(object_index2)->surface_edges_index.at(j)).isColliding = true;
//			}
//		}
//	}
//}
//
//double PhysicsEngine::dist_point_edge(int point_index, int edge_index) {
//	
//	Vector2d_old vec_base = pos_old->at(edges->at(edge_index).b_index) - pos_old->at(edges->at(edge_index).a_index);
//	double base_length = vec_base.abs();
//
//	Vector2d_old vec_slant = pos_old->at(point_index) - pos_old->at(edges->at(edge_index).a_index);
//	double cross = abs(vec_base.x()*vec_slant.y() - vec_slant.x()*vec_base.y());
//
//	return cross / base_length;
//
//}
//
//void PhysicsEngine::resolve_edge_collisions(int edge_index1, int edge_index2) {
//
//	collision_forces[edges->at(edge_index1).a_index] += dist_point_edge(edges->at(edge_index1).a_index, edge_index2) * edges->at(edge_index2).get_normal(pos_old) * collision_const_obj;
//	collision_forces[edges->at(edge_index1).b_index] += dist_point_edge(edges->at(edge_index1).b_index, edge_index2) * edges->at(edge_index2).get_normal(pos_old) * collision_const_obj;
//	collision_forces[edges->at(edge_index2).a_index] += dist_point_edge(edges->at(edge_index2).a_index, edge_index1) * edges->at(edge_index1).get_normal(pos_old) * collision_const_obj;
//	collision_forces[edges->at(edge_index2).b_index] += dist_point_edge(edges->at(edge_index2).b_index, edge_index1) * edges->at(edge_index1).get_normal(pos_old) * collision_const_obj;
//
//}

void PhysicsEngine::step_euler(double dt) {

	Matrix <double, 2, Dynamic> combined;
	combined.resize(2, 2 * pos->cols());
	combined << (*pos) , (*vel);

	Matrix <double, 2, Dynamic> vel_forces = get_pos_vel_slope(combined, pos->cols());
	

	(*pos) += vel_forces.block(0, 0, 2, pos->cols()) * dt;
	(*vel) += vel_forces.block(0, pos->cols(), 2, pos->cols()) * dt;
}

void PhysicsEngine::step_rk4(double dt) {

	//SETUP
	int num_points = pos->cols();

	Matrix <double, 2, Dynamic> combined;
	combined.resize(2, 2 * pos->cols());
	combined << (*pos), (*vel);

	//COMPUTE K1 - K4
	
	//k1
	Matrix <double, 2, Dynamic> k_1 = get_pos_vel_slope(combined, num_points) * dt;	//t

	// cout << "dF\nnp.array([";
	// for (int i = k_1.cols()/2; i < k_1.cols(); i++){
	// 	cout << "[" << k_1(0, i) << ", " << k_1(1, i) << "],"; 
	// }
	// cout << "])";
	// cout << "\n\n\n\n";

	//k2
	Matrix <double, 2, Dynamic> temp = combined + (k_1*0.5);
	
	//set fixed points
	//temp(Eigen::all, Eigen::seqN(0, num_points)) = fixed->select(0.0f, temp(Eigen::all, Eigen::seqN(0, num_points)));
	//temp(Eigen::all, Eigen::seqN(num_points, num_points)) = fixed->select(0.0f, temp(Eigen::all, Eigen::seqN(num_points, num_points)));

	Matrix <double, 2, Dynamic> k_2 = get_pos_vel_slope(temp, num_points) * dt;		//t + dt/2

	//k3
	temp = combined + (k_2*0.5);
	
	//set fixed points
	//temp(Eigen::all, Eigen::seqN(0, num_points)) = fixed->select(0.0f, temp(Eigen::all, Eigen::seqN(0, num_points)));
	//temp(Eigen::all, Eigen::seqN(num_points, num_points)) = fixed->select(0.0f, temp(Eigen::all, Eigen::seqN(num_points, num_points)));

	Matrix <double, 2, Dynamic> k_3 = get_pos_vel_slope(temp, num_points) * dt;		//t + dt/2

	//k4
	temp = combined + k_3;

	//set fixed points
	//temp(Eigen::all, Eigen::seqN(0, num_points)) = fixed->select(0.0f, temp(Eigen::all, Eigen::seqN(0, num_points)));
	//temp(Eigen::all, Eigen::seqN(num_points, num_points)) = fixed->select(0.0f, temp(Eigen::all, Eigen::seqN(num_points, num_points)));

	Matrix <double, 2, Dynamic> k_4 = get_pos_vel_slope(temp, num_points) * dt;		//t

	//APPLY UPDATE
	temp = (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6;

	//set fixed points
	temp(Eigen::all, Eigen::seqN(0, num_points)) = fixed->select(0.0f, temp(Eigen::all, Eigen::seqN(0, num_points)));
	temp(Eigen::all, Eigen::seqN(num_points, num_points)) = fixed->select(0.0f, temp(Eigen::all, Eigen::seqN(num_points, num_points)));

	(*pos) += temp.block(0, 0, 2, num_points);
	(*vel) += temp.block(0, num_points, 2, num_points);

}

Matrix <double, 2, Dynamic> PhysicsEngine::get_pos_vel_slope(Ref<Matrix <double, 2, Dynamic>> data, int num_points) {

	Array <double, 2, Dynamic> vel_x_y = data.block(0, num_points, 2, num_points);

	Matrix <double, 1, Dynamic> pos_y = data(1, seqN(0, num_points)); //data.block(1, 0, 1, num_points);

	//CLEAR ALL FORCES
	Matrix <double, 2, Dynamic> points_force = MatrixXd::Constant(2, num_points, 0);

	//ADD SPRING FORCES 

	Array <double, 2, Dynamic> vec_q_to_p = data(Eigen::all, *a_index) - data(Eigen::all, *b_index);

	// cout << "length_eq\nnp.array([";
	// for (int i = 0; i < length_eq->cols(); i++){
	// 	cout << "[" << length_eq->array()(0, i) << "],"; 
	// }
	// cout << "])\n";

	// cout << "spring_const\nnp.array([";
	// for (int i = 0; i < spring_const->cols(); i++){
	// 	cout << "[" << spring_const->array()(0, i)<< "],"; 
	// }
	// cout << "])\n";

	Array <double, 1, Dynamic> dist = (vec_q_to_p.row(0).square() + vec_q_to_p.row(1).square()).sqrt();
	Array <double, 1, Dynamic> spring_force_mag = (dist - length_eq->array()) * spring_const->array() / dist;

	// cout << "dist\nnp.array([";
	// for (int i = 0; i < dist.cols(); i++){
	// 	cout << "[" << dist(0, i) << "],"; 
	// }
	// cout << "])\n";

	// cout << "spring_force_mag\nnp.array([";
	// for (int i = 0; i < spring_force_mag.cols(); i++){
	// 	cout << "[" << spring_force_mag(0, i) << "],"; 
	// }
	// cout << "])\n";
	
	vec_q_to_p(0, Eigen::all) *= spring_force_mag;
	vec_q_to_p(1, Eigen::all) *= spring_force_mag;

	points_force(Eigen::all, *a_index) -= vec_q_to_p.matrix();
	points_force(Eigen::all, *b_index) += vec_q_to_p.matrix();

	// cout << "pf1\nnp.array([";
	// for (int i = points_force.cols()/2; i < points_force.cols(); i++){
	// 	cout << "[" << points_force(0, i) << ", " << points_force(1, i) << "],"; 
	// }
	// cout << "])\n";

	//ADD VISCOUS DRAG
	points_force += vel_x_y.matrix() * (-viscous_drag);

	// cout << "pf2\nnp.array([";
	// for (int i = points_force.cols()/2; i < points_force.cols(); i++){
	// 	cout << "[" << points_force(0, i) << ", " << points_force(1, i) << "],"; 
	// }
	// cout << "])\n";

	//ADD GRAVITY
	points_force(1, Eigen::all) += ((mass->row(0)) * (-gravity));

	// cout << "pf3\nnp.array([";
	// for (int i = points_force.cols()/2; i < points_force.cols(); i++){
	// 	cout << "[" << points_force(0, i) << ", " << points_force(1, i) << "],"; 
	// }
	// cout << "])\n";

	//ADD OBJECT COLLISION
	points_force += collision_forces;

	// cout << "pf4\nnp.array([";
	// for (int i = points_force.cols()/2; i < points_force.cols(); i++){
	// 	cout << "[" << points_force(0, i) << ", " << points_force(1, i) << "],"; 
	// }
	// cout << "])\n";

	//ADD GROUND COLLISON AND FRICTION
	
	//setup
	Matrix <bool, 1, Dynamic> is_colliding_with_ground = pos_y.array() < 0;
	Matrix <double, 1, Dynamic> normal_force = is_colliding_with_ground.select(pos_y, 0) * (-collision_const_ground);

	//ground collision
	points_force(1, Eigen::all) += normal_force;

	//friction = -constants * normal force * tanh(x_vel /( normal force * constants))
	Array <double, 1, Dynamic> friction_temp = (data.block(0, num_points, 1, num_points).array() * normal_force.array() * dynamic_friction_const).tanh();
	points_force(0, Eigen::all) += (-10 * dynamic_friction_const) * normal_force.cwiseProduct(is_colliding_with_ground.select(friction_temp.matrix(), 0));


	// cout << "pf5\nnp.array([";
	// for (int i = points_force.cols()/2; i < points_force.cols(); i++){
	// 	cout << "[" << points_force(0, i) << ", " << points_force(1, i) << "],"; 
	// }
	// cout << "])\n";
	
	//ADD INTER OBJECT COLLISON
	//for (int i = 0; i < pos_old->size(); i++) {
	//	points_force[i] += collision_forces[i];
	//}

	//STRUCTURE OUTPUT
	Matrix <double, 2, Dynamic> vel_forces = vel_x_y;
	vel_forces.conservativeResize(2, num_points * 2);

	//vel_forces.block(0, 0, 2, num_points) = data.block(0, num_points, 2, num_points);
	//vel_forces.block(0, num_points, 2, num_points) = points_force.cwiseQuotient(*mass);
	vel_forces(Eigen::all, Eigen::seqN(num_points, num_points)) = points_force.cwiseQuotient(*mass);

	return vel_forces;
}

void PhysicsEngine::update_true_vel(double dt) {
	
	*vel_true = (*pos - *pos_last)/ dt;
	*pos_last = *pos;
	
}
void PhysicsEngine::resolve_edge_constraints() {

	double thresh_diff = 0.25;
	double thresh_diff_rigid = 0.03;

	Array <double, 2, Dynamic> vec_q_to_p = (*pos)(Eigen::all, *a_index) - (*pos)(Eigen::all, *b_index);
	Array <double, 1, Dynamic> dist = (vec_q_to_p.row(0).square() + vec_q_to_p.row(1).square()).sqrt();

	Array <double, 1, Dynamic> additive_mag_overshoot = (dist - init_length_eq->array()*(1 + thresh_diff)) / (dist * 2);
	Array <double, 1, Dynamic> additive_mag_undershoot = (dist - init_length_eq->array()*(1 - thresh_diff)) / (dist * 2);

	Array <double, 2, Dynamic> overshoot_correction = vec_q_to_p;
	overshoot_correction(0, Eigen::all) *= additive_mag_overshoot;
	overshoot_correction(1, Eigen::all) *= additive_mag_overshoot;

	Array <double, 2, Dynamic> undershoot_correction = vec_q_to_p;
	undershoot_correction(0, Eigen::all) *= additive_mag_undershoot;
	undershoot_correction(1, Eigen::all) *= additive_mag_undershoot;


	Array <double, 1, Dynamic> stress = dist/ init_length_eq->array();

	overshoot_correction(0, Eigen::all) = (stress > (1 + thresh_diff)).select(overshoot_correction(0, Eigen::all), 0.0f);
	overshoot_correction(1, Eigen::all) = (stress > (1 + thresh_diff)).select(overshoot_correction(1, Eigen::all), 0.0f);

	undershoot_correction(0, Eigen::all) = (stress < (1 - thresh_diff)).select(undershoot_correction(0, Eigen::all), 0.0f);
	undershoot_correction(1, Eigen::all) = (stress < (1 - thresh_diff)).select(undershoot_correction(1, Eigen::all), 0.0f);

	Array <bool, 2, Dynamic> a_fixed = (*fixed)(Eigen::all, *a_index);
	Array <bool, 2, Dynamic> b_fixed = (*fixed)(Eigen::all, *b_index);

	//double up_count = (double)(stress > 1.75).count() / (double)stress.size();
	//double low_count = (double)(stress < 0.5).count() / (double)stress.size();
	//if(low_count > 0.01)
	//	cout << low_count << "\n";
	//if (up_count >= 0.01 && low_count >= 0.01)



	//overshoot_correction(0, Eigen::all) = (stress < 2.0).select(overshoot_correction(0, Eigen::all), 0.0f);
	//overshoot_correction(1, Eigen::all) = (stress < 2.0).select(overshoot_correction(1, Eigen::all), 0.0f);

	//undershoot_correction(0, Eigen::all) = (stress > 0.0).select(undershoot_correction(0, Eigen::all), 0.0f);
	//undershoot_correction(1, Eigen::all) = (stress > 0.0).select(undershoot_correction(1, Eigen::all), 0.0f);


	//cout << "Overshoots: \n";
	//for (int i = 0; i < overshoot_correction.cols(); i++) {
	//	if (overshoot_correction(0, i) != 0 || overshoot_correction(1, i) != 0)
	//		cout << i << " " << overshoot_correction(0, i) << " " << overshoot_correction(1, i) << "\n";
	//}
	//cout << "\nUndershoots: \n\n";
	//for (int i = 0; i < undershoot_correction.cols(); i++) {
	//	if (undershoot_correction(0, i) != 0 || undershoot_correction(1, i) != 0)
	//		cout << i << " " << undershoot_correction(0, i) << " " << undershoot_correction(1, i) << "\n";
	//}

	//Array <double, 2, Dynamic> correction = (overshoot_correction.matrix() + undershoot_correction.matrix())*0.5;
	//cout << "min " << correction.abs().minCoeff() << " | mean " << correction.abs().mean() << " | max " << correction.abs().maxCoeff() << "\n";

	//(*pos)(Eigen::all, *a_index) -= (overshoot_correction.matrix() + undershoot_correction.matrix())*0.8;
	//(*pos)(Eigen::all, *b_index) += (overshoot_correction.matrix() + undershoot_correction.matrix())*0.8;
	(*pos)(Eigen::all, *a_index) -= a_fixed.select(0.0, (overshoot_correction.matrix() + undershoot_correction.matrix())*0.8);
	(*pos)(Eigen::all, *b_index) += b_fixed.select(0.0, (overshoot_correction.matrix() + undershoot_correction.matrix())*0.8);
	//(*pos)(Eigen::all, *a_index) -= a_fixed.select(0.0, (overshoot_correction.matrix() + undershoot_correction.matrix())*0.1);
	//(*pos)(Eigen::all, *b_index) += b_fixed.select(0.0, (overshoot_correction.matrix() + undershoot_correction.matrix())*0.1);

	for (int i = 0; i < objects->size(); i++) {
		for (int j = 0; j < objects->at(i)->boxels.size(); j++) {

			Boxel* current = &objects->at(i)->boxels.at(j);

			if (current->cell_type != CELL_RIGID)
				continue;

			for (int k = 0; k < current->edges.size(); k++) {

				int edge_index = current->edges(k);
				Vector2d vec_q_to_p = pos->col(edges->at(edge_index).a_index) - pos->col(edges->at(edge_index).b_index);
				double dist = vec_q_to_p.norm();

				double additive_mag_overshoot = (dist - (*init_length_eq)(0, edge_index)*(1 + thresh_diff_rigid)) / (dist * 2);
				double additive_mag_undershoot = (dist - (*init_length_eq)(0, edge_index)*(1 - thresh_diff_rigid)) / (dist * 2);

				Vector2d overshoot_correction = vec_q_to_p;
				overshoot_correction *= additive_mag_overshoot;

				Vector2d undershoot_correction = vec_q_to_p;
				undershoot_correction *= additive_mag_undershoot;

				double stress = dist / (*init_length_eq)(0, edge_index);

				if (stress < (1 + thresh_diff_rigid))
					overshoot_correction = Vector2d(0, 0);

				if (stress > (1 - thresh_diff_rigid))
					undershoot_correction = Vector2d(0, 0);

				pos->col(edges->at(edge_index).a_index) -= (overshoot_correction.matrix() + undershoot_correction.matrix())*0.5;
				pos->col(edges->at(edge_index).b_index) += (overshoot_correction.matrix() + undershoot_correction.matrix())*0.5;
			}
		}
	}
	//cout << "REC" << "\n";

}

void PhysicsEngine::resolve_collisions() {

	for (int i = 0; i < objects->size(); i++) {
		objects->at(i)->recompute_bbs(*pos);
	}

	//visualization only
	for (int i = 0; i < objects->size(); i++) {
		for (int j = 0; j < objects->at(i)->surface_boxels_index.size(); j++) {
			objects->at(i)->boxels[objects->at(i)->surface_boxels_index[j]].is_colliding = false;
		}
	}

	while (point_is_colliding->size() < pos->cols())
		point_is_colliding->push_back(false);

	for (int i = 0; i < pos->cols(); i++) {
		point_is_colliding->at(i) = false;
	}

	for (int i = 0; i < edges->size(); i++) {
		edges->at(i).isColliding = false;
	}


	collision_forces = MatrixXd::Zero(2, pos->cols());
	collision_vels = MatrixXd::Zero(2, pos->cols());
	for (int i = 0; i < objects->size(); i++) {
		for (int j = i; j < objects->size(); j++) {
			// if (i == j)
			// 	continue;
			resolve_object_collisions(objects->at(i), &objects->at(i)->nodes[objects->at(i)->tree_root], objects->at(j), &objects->at(j)->nodes[objects->at(j)->tree_root], i==j);
		}
	}


}
void PhysicsEngine::resolve_object_collisions(SimObject* object_a, BBTreeNode* a_node, SimObject* object_b, BBTreeNode* b_node, bool are_same) {

	if (!a_node->bbox.in(&b_node->bbox)){
		// cout << "bbox\n";
		// a_node->bbox.print();
		// cout << "\nNOT in bbox\n";
		// b_node->bbox.print();
		// cout << "\n\n";
		return;
	}

	// cout << "bbox\n";
	// a_node->bbox.print();
	// cout << "\nin bbox\n";
	// b_node->bbox.print();
	// cout << "\n\n";

	if (a_node->is_leaf && b_node->is_leaf) {
		object_a->boxels[a_node->boxel_index].is_colliding = true;
		object_b->boxels[b_node->boxel_index].is_colliding = true;
		
		//if (are_same) {
		//	if (object_a->boxels[a_node->boxel_index].grid_index == object_b->boxels[b_node->boxel_index].grid_index)
		//		return;

		//	int h = object_a->grid_h;
		//	int w = object_a->grid_w;

		//	int x1 = object_a->boxels[a_node->boxel_index].grid_index % w;
		//	int y1 = object_a->boxels[a_node->boxel_index].grid_index / w;

		//	int x2 = object_b->boxels[b_node->boxel_index].grid_index % w;
		//	int y2 = object_b->boxels[b_node->boxel_index].grid_index / w;

		//	if (abs(x1 - x2) < 1.001 && abs(y1 - y2) < 1.001)
		//		return;
		//}
		
		// cout << "cb: " << a_node->boxel_index << " " << b_node->boxel_index << "\n";
		resolve_boxel_collisions(&object_a->boxels[a_node->boxel_index], &object_b->boxels[b_node->boxel_index]);
		resolve_boxel_collisions(&object_b->boxels[b_node->boxel_index], &object_a->boxels[a_node->boxel_index]);
		return;
	}
		
	if (a_node->is_leaf) {
		resolve_object_collisions(object_a, a_node, object_b, &object_b->nodes[b_node->a_index], are_same);
		resolve_object_collisions(object_a, a_node, object_b, &object_b->nodes[b_node->b_index], are_same);
		return;
	}

	if (b_node->is_leaf) {
		resolve_object_collisions(object_a, &object_a->nodes[a_node->a_index], object_b, b_node, are_same);
		resolve_object_collisions(object_a, &object_a->nodes[a_node->b_index], object_b, b_node, are_same);
		return;
	}

	if (a_node->bbox.area() < b_node->bbox.area()) {
		resolve_object_collisions(object_a, a_node, object_b, &object_b->nodes[b_node->a_index], are_same);
		resolve_object_collisions(object_a, a_node, object_b, &object_b->nodes[b_node->b_index], are_same);
	}
	else {
		resolve_object_collisions(object_a, &object_a->nodes[a_node->a_index], object_b, b_node, are_same);
		resolve_object_collisions(object_a, &object_a->nodes[a_node->b_index], object_b, b_node, are_same);
	}

}

bool PhysicsEngine::is_any_robot_self_colliding() {

	for (int i = 0; i < objects->size(); i++) {
		if (!objects->at(i)->is_robot)
			continue;
		int num_nonadj_self_collisions = count_object_self_collisions(
			objects->at(i),
			&objects->at(i)->nodes[objects->at(i)->tree_root],
			objects->at(i),
			&objects->at(i)->nodes[objects->at(i)->tree_root]);

		if (num_nonadj_self_collisions * 1.0 > objects->at(i)->surface_boxels_index.size())
			return true;
	}

	return false;

	//if (num_nonadj_self_collisions * 1.0 / robot->surface_boxels_index.size() > 0.5)
		//cout << num_nonadj_self_collisions * 1.0 / robot->surface_boxels_index.size() << "\n";
	
	//cout << count << " " << robot->num_suface_neighbors << "\n";
}

int PhysicsEngine::count_object_self_collisions(SimObject* object_a, BBTreeNode* a_node, SimObject* object_b, BBTreeNode* b_node) {
	if (!a_node->bbox.in(&b_node->bbox))
		return 0;

	if (a_node->is_leaf && b_node->is_leaf) {

		if (object_a->boxels[a_node->boxel_index].grid_index == object_b->boxels[b_node->boxel_index].grid_index)
			return 0; 

		int h = object_a->grid_h;
		int w = object_a->grid_w;

		int x1 = object_a->boxels[a_node->boxel_index].grid_index % w;
		int y1 = object_a->boxels[a_node->boxel_index].grid_index / w;

		int x2 = object_b->boxels[b_node->boxel_index].grid_index % w;
		int y2 = object_b->boxels[b_node->boxel_index].grid_index / w;

		if (abs(x1 - x2) < 1.001 && abs(y1 - y2) < 1.001)
			return 0;
		//object_a->boxels[a_node->boxel_index].is_colliding = true;
		//object_b->boxels[b_node->boxel_index].is_colliding = true;
		return 1;
	}

	int count = 0;
	if (a_node->is_leaf) {
		count += count_object_self_collisions(object_a, a_node, object_b, &object_b->nodes[b_node->a_index]);
		count += count_object_self_collisions(object_a, a_node, object_b, &object_b->nodes[b_node->b_index]);
		return count;
	}

	if (b_node->is_leaf) {
		count += count_object_self_collisions(object_a, &object_a->nodes[a_node->a_index], object_b, b_node);
		count += count_object_self_collisions(object_a, &object_a->nodes[a_node->b_index], object_b, b_node);
		return count;
	}

	if (a_node->bbox.area() < b_node->bbox.area()) {
		count += count_object_self_collisions(object_a, a_node, object_b, &object_b->nodes[b_node->a_index]);
		count += count_object_self_collisions(object_a, a_node, object_b, &object_b->nodes[b_node->b_index]);
	}
	else {
		count += count_object_self_collisions(object_a, &object_a->nodes[a_node->a_index], object_b, b_node);
		count += count_object_self_collisions(object_a, &object_a->nodes[a_node->b_index], object_b, b_node);
	}
	return count;
}

void PhysicsEngine::resolve_boxel_collisions(Boxel* main, Boxel* ref) {

	for (int i = 0; i < main->points.size(); i++) {

		bool is_neighbor = false;
		for (int j = 0; j < ref->points.size(); j++) {
			if (main->points[i] == ref->points[j])
				is_neighbor = true;
		}
		if (is_neighbor)
			continue;

		if (ref->point_in_boxel(pos->col(main->points[i]), *pos)) {
			point_is_colliding->at(main->points[i]) = true;
			resolve_point_boxel_collisions(main->points[i], main, ref);
		}
	}
}

void PhysicsEngine::resolve_point_boxel_collisions(int p_index, Boxel* main, Boxel* ref) {
	vector <int> main_edges_index;

	//find all relavent edges

	if (p_index == main->point_top_left_index) {
		//if(edges->at(main->edge_top_index).isOnSurface)
			main_edges_index.push_back(main->edge_top_index);
		//if (edges->at(main->edge_left_index).isOnSurface)
			main_edges_index.push_back(main->edge_left_index);
	}
	if (p_index == main->point_top_right_index) {
		//if (edges->at(main->edge_top_index).isOnSurface)
			main_edges_index.push_back(main->edge_top_index);
		//if (edges->at(main->edge_right_index).isOnSurface)
			main_edges_index.push_back(main->edge_right_index);
	}
	if (p_index == main->point_bot_left_index) {
		//if (edges->at(main->edge_bot_index).isOnSurface)
			main_edges_index.push_back(main->edge_bot_index);
		//if (edges->at(main->edge_left_index).isOnSurface)
			main_edges_index.push_back(main->edge_left_index);
	}
	if (p_index == main->point_bot_right_index) {
		//if (edges->at(main->edge_bot_index).isOnSurface)
			main_edges_index.push_back(main->edge_bot_index);
		//if (edges->at(main->edge_right_index).isOnSurface)
			main_edges_index.push_back(main->edge_right_index);
	}

	vector <int> ref_edges_index;
	vector <double> ref_edges_score;

	for (int i = 0; i < ref->edges.size(); i++) {
		
		if (!edges->at(ref->edges[i]).isOnSurface)
			continue;

		bool is_edge_viable = false;
		for (int j = 0; j < main_edges_index.size(); j++) {
			if (get_intersecting(main_edges_index.at(j), ref->edges[i])) {
				is_edge_viable = true;
				break;
			}
		}

		if (is_edge_viable) {
			ref_edges_index.push_back(ref->edges[i]);
			ref_edges_score.push_back(dist_point_edge(p_index, ref->edges[i]));
		}

		/*for (int j = 0; j < main_edges_index.size(); j++) {
			if (get_intersecting(main_edges_index.at(j), ref->edges[i])) {
				if (ref_edges_index[i] == -1) {
					ref_edges_index[i] = ref->edges[i];
					ref_edges_score[i] = dist_point_edge(p_index, ref_edges_index[i]);
					continue;
				}
				double dist = dist_point_edge(p_index, ref->edges[i]);
				if (dist < ref_edges_score[i]) {
					ref_edges_score[i] = dist;
					ref_edges_index[i] = ref->edges[i];
				}
			}
		}*/
	}
	//for (int i = 0; i < main_edges_index.size(); i++) {

	//	ref_edges_index.push_back(-1);
	//	for (int j = 0; j < ref->edges.size(); j++) {
	//		
	//		if (edges->at(ref->edges[j]).isColliding)
	//			continue;
	//		if (get_intersecting(main_edges_index.at(i), ref->edges[j])) {
	//			ref_edges_index.at(i) = ref->edges[j];
	//			break;
	//		}
	//	}
	//}

	// error check

	//int first_good_index = -1;
	//for (int i = 0; i < main_edges_index.size(); i++) {

	//	if (ref_edges_index.at(i) == -1)
	//		continue;

	//	first_good_index = i;
	//	break;
	//}

	//if (first_good_index == -1) {
	//	//cout << "bad " << main_edges_index.size() << "\n";
	//	return;
	//}

	//// find closest edge

	//int index_of_min_dist = first_good_index;
	//double min_dist = dist_point_edge(p_index, ref_edges_index.at(first_good_index));

	//for (int i = 0; i < main_edges_index.size(); i++) {

	//	if (ref_edges_index.at(i) == -1)
	//		continue;

	//	double dist = dist_point_edge(p_index, ref_edges_index.at(i));
	//	if (dist < min_dist) {
	//		min_dist = dist;
	//		index_of_min_dist = i;
	//	}
	//}

	//set normal forces

	if (ref_edges_index.size() == 0)
		return;

	int index_of_min_dist = 0;
	double min_dist = ref_edges_score[0];

	for (int i = 0; i < ref_edges_index.size(); i++) {
		if (ref_edges_score[i] < min_dist) {
			min_dist = ref_edges_score[i];
			index_of_min_dist = i;
		}
	}

	edges->at(ref_edges_index.at(index_of_min_dist)).isColliding = true;

	Vector2d normal_force = edges->at(ref_edges_index.at(index_of_min_dist)).get_normal(*pos) * collision_const_obj * (collision_base_dist_additive + min_dist);

	//set velocity dampening in collision
	Vector2d normal = edges->at(ref_edges_index.at(index_of_min_dist)).get_normal(*pos);
	Vector2d unit_norm = normal.normalized();

	int a_point = edges->at(ref_edges_index.at(index_of_min_dist)).a_index;
	int b_point = edges->at(ref_edges_index.at(index_of_min_dist)).b_index;


	Vector2d vel_additive1 = unit_norm * (-1) * (unit_norm.x() * vel_true->col(p_index).x() + unit_norm.y() * vel_true->col(p_index).y()) * collision_vel_damping;
	Vector2d vel_additive2 = unit_norm * (1) * (unit_norm.x() * vel_true->col(a_point).x() + unit_norm.y() * vel_true->col(a_point).y()) * collision_vel_damping;
	Vector2d vel_additive3 = unit_norm * (1) * (unit_norm.x() * vel_true->col(b_point).x() + unit_norm.y() * vel_true->col(b_point).y()) * collision_vel_damping;
	//collision_vels.col(p_index) += vel_additive;

	//apply collision forces
	collision_forces.col(p_index) += normal_force + vel_additive1;
	collision_forces.col(edges->at(ref_edges_index.at(index_of_min_dist)).a_index) -= normal_force / 2 + vel_additive2 / 2;
	collision_forces.col(edges->at(ref_edges_index.at(index_of_min_dist)).b_index) -= normal_force / 2 + vel_additive3 / 2;
	//collision_forces[p_index] += dist_point_edge(p_index, ref_edges_index.at(i)) * edges->at(ref_edges_index.at(i)).get_normal(pos) * collision_const_obj;


	//set friction/tangential forces

	Vector2d unit_tangent = normal.normalized();
	unit_tangent = Vector2d(-unit_tangent.y(), unit_tangent.x());

	//double projected_vel_mag = unit_tangent.x() * vel->col(p_index).x() + unit_tangent.y() * vel->col(p_index).y();
	double projected_vel_mag = unit_tangent.x() * vel_true->col(p_index).x() + unit_tangent.y() * vel_true->col(p_index).y();
	double normal_force_mag = normal_force.norm();

	if (projected_vel_mag < 0) {
		projected_vel_mag *= -1;
		unit_tangent *= -1;
	}

	Vector2d friction_force = friction_const * unit_tangent * (-1) * dynamic_friction_const *  normal_force_mag * tanh(projected_vel_mag / (dynamic_friction_const *  normal_force_mag));
	//cout << friction_force.x() << " " << friction_force.y() << "\n";
	//cout << normal_force_mag << "\n\n\n";
	collision_forces.col(p_index) += friction_force;
	collision_forces.col(edges->at(ref_edges_index.at(index_of_min_dist)).a_index) -= friction_force / 2;
	collision_forces.col(edges->at(ref_edges_index.at(index_of_min_dist)).b_index) -= friction_force / 2;

	//cout << normal.x() << " " << normal.y() << " " << unit_tangent.x() << " " << unit_tangent.y() << "\n";


	//cout << "Frictional Force Caclulation: " << "\n";
	//cout << "v: " << vel->col(p_index)[0] << " " << vel->col(p_index)[1] << "\n";
	//cout << "vt: " << vel_true.col(p_index)[0] << " " << vel_true.col(p_index)[1] << "\n";
	//cout << "s: " << unit_tangent[0] << " " << unit_tangent[1] << "\n";
	//cout << "f: " << friction_force[0] << " " << friction_force[1] << "\n\n";
}

bool PhysicsEngine::get_intersecting(int edgeIndex1, int edgeIndex2) {

	Edge* edge1 = &(edges->at(edgeIndex1));
	Edge* edge2 = &(edges->at(edgeIndex2));

	Vector2d diff1 = pos->col(edge1->b_index) - pos->col(edge1->a_index);
	Vector2d diff2 = pos->col(edge2->b_index) - pos->col(edge2->a_index);

	double det = -diff1.x()*diff2.y() + diff2.x()*diff1.y();

	//edges are parallel
	if (abs(det) < 0.0001)
		return false;

	double x_diff = pos->col(edge2->a_index).x() - pos->col(edge1->a_index).x();
	double y_diff = pos->col(edge2->a_index).y() - pos->col(edge1->a_index).y();

	//We view the edges as lines paramaterized by s and t.
	//If the point of intersection has the property that 0 < t < 1 and 0 < s < 1, there is a collision.

	double t = (-diff2.y()*x_diff + diff2.x()*y_diff) / det;
	double s = (-diff1.y()*x_diff + diff1.x()*y_diff) / det;

	if (-F_ERROR_TOL + 0.0f <= t && -F_ERROR_TOL + t <= 1 && -F_ERROR_TOL + 0.0f <= s && -F_ERROR_TOL + s <= 1)
		return true;

	return false;
}

double PhysicsEngine::dist_point_edge(int point_index, int edge_index) {
	
	Vector2d vec_base = pos->col(edges->at(edge_index).b_index) - pos->col(edges->at(edge_index).a_index);

	double base_length = vec_base.norm();
	if (base_length < 0.000001)
		return 1000.0;

	Vector2d vec_slant = pos->col(point_index) - pos->col(edges->at(edge_index).a_index);
	double cross = abs(vec_base.x()*vec_slant.y() - vec_slant.x()*vec_base.y());

	return cross / base_length;
}

PhysicsEngine::~PhysicsEngine()
{
}

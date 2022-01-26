#include "Interface.h"
#include "Colors.h"

Interface::Interface(Sim* sim)
{
	//RENDERING
	int debug_window_width = 640;
	int debug_window_height = 640;
	last = Vector2d(debug_window_width, debug_window_height);

	if (!glfwInit()) {
		cout << "Error initializing GLFW.\n";
		return;
	}

	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	debug_window = glfwCreateWindow(debug_window_width, debug_window_height, "Evolution Gym - Debug Window", NULL, NULL);
	if (!debug_window)
	{
		glfwTerminate();
		cout << "GLFW failed to create rendering winidow.\n";
		return;
	}

	glfwMakeContextCurrent(debug_window);

	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		// Problem: glewInit failed, something is seriously wrong. 
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
	}
	fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

	debug_window_showing = false;

	//DATA
	Interface::pos = sim->environment.get_pos();
	Interface::edges = sim->environment.get_edges();
	Interface::objects = sim->environment.get_objects();

	Interface::point_is_colliding = &(sim->environment.point_is_colliding);
}

void Interface::init() {

	
}

GLFWwindow* Interface::get_debug_window_ref(){
	return debug_window;
}

void Interface::render(Camera camera, bool hide_background, bool hide_grid, bool hide_edges, bool hide_boxels, bool dont_clear){

	if (!camera.is_renderable && debug_window_showing)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glfwMakeContextCurrent(debug_window);

		if (abs(last.x() - camera.resolution.x()) > 1 || abs(last.y() - camera.resolution.y()) > 1) {
			glfwSetWindowSize(debug_window, camera.resolution.x(), camera.resolution.y());
			glViewport(0, 0, camera.resolution.x(), camera.resolution.y());
			last = camera.resolution;
		}


		// Set background color to black and opaque
		if (!hide_background)
			glClearColor(244.0/255.0, 245.0/255.0, 247.0/255.0, 1.0);
		else
			glClearColor(1.0, 1.0, 1.0, 1.0);

		// Clear the color buffer (background)
		if (!dont_clear)
			glClear(GL_COLOR_BUFFER_BIT);

		if (!hide_grid)
			render_grid(camera);
		//render_points(camera);
		//render_bounding_boxes(camera);
		if (!hide_boxels)
			render_boxels(camera);
		if (!hide_edges)
			render_edges(camera);
		//render_object_points(camera);
		//render_edge_normals(camera);


		// Render now
		glFlush();
		glFinish();

		/* Swap front and back buffers */
		glfwSwapBuffers(debug_window);

		/* Poll for and process events */
		glfwPollEvents();
	}
	if (camera.is_renderable) {
		
		camera.set_focus();
		if (abs(last.x() - camera.resolution.x()) > 1 || abs(last.y() - camera.resolution.y()) > 1) {
			glViewport(0, 0, camera.resolution.x(), camera.resolution.y());
			last = camera.resolution;
		}

		// Set background color to black and opaque
		//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClearColor(244.0 / 255.0, 245.0 / 255.0, 247.0 / 255.0, 1.0);

		// Clear the color buffer (background)
		if (!hide_background)
			glClearColor(244.0 / 255.0, 245.0 / 255.0, 247.0 / 255.0, 1.0);
		else
			glClearColor(1.0, 1.0, 1.0, 1.0);

		// Clear the color buffer (background)
		if (!dont_clear)
			glClear(GL_COLOR_BUFFER_BIT);


		if (!hide_grid)
			render_grid(camera);
		if (!hide_boxels)
			render_boxels(camera);
		if (!hide_edges)
			render_edges(camera);

		glFlush();
		glFinish();
		glReadPixels(0, 0, camera.resolution.x() , camera.resolution.y(), GL_RGB, GL_UNSIGNED_BYTE, camera.image_data);
	}
}

void Interface::render_edges(Camera camera) {
	
	Vector2d coord;
	for (int i = 0; i < edges->size(); i++) {

		if (!edges->at(i).isOnSurface)
			continue;
		glColor3f(0.8, 0.8, 0.8);

		glBegin(GL_LINES);

		//if (edges->at(i).color == 0)
		//	glColor3f(0.5, 0.5, 0.5);
		//else if (edges->at(i).color == 1)
		//	glColor3f(0.2, 0.2, 0.2);
		//else if (edges->at(i).color == 2)
		//	glColor3f(0.3, 1, 0.3);

		if (edges->at(i).isOnSurface)
			glColor3f(0, 63.0/255.0, 92.0/255.0);
		//if (edges->at(i).isColliding)
		//	glColor3f(1.0, 0.3, 0.3);
			//glColor3f(1, 0.3, 0.3);

		int a_index = edges->at(i).a_index;
		int b_index = edges->at(i).b_index;

		coord = camera.world_to_camera(pos->col(a_index));
		glVertex2f(coord.x(), coord.y());

		coord = camera.world_to_camera(pos->col(b_index));
		glVertex2f(coord.x(), coord.y());
		
		glEnd();

	}
}

void Interface::render_grid(Camera camera) {

	double min_x = int((camera.pos.x() - camera.size.x())*10.0) - 1;
	double max_x = int((camera.pos.x() + camera.size.x())*10.0) + 1;
	double min_y = int((camera.pos.y() - camera.size.y())*10.0) - 1;
	double max_y = int((camera.pos.y() + camera.size.y())*10.0) + 1;

	Vector2d coord;
	for (int i = min_x; i <= max_x; i++) {

		glBegin(GL_LINES);

		glColor3f(0.8, 0.8, 0.8);

		Vector2d p1 = Vector2d(i*0.1, max_y*0.1);

		coord = camera.world_to_camera(p1);
		glVertex2f(coord.x(), coord.y());

		p1 = Vector2d(i*0.1, min_y*0.1);

		coord = camera.world_to_camera(p1);
		glVertex2f(coord.x(), coord.y());

		glEnd();
	}
	for (int i = min_y; i <= max_y; i++) {

		glBegin(GL_LINES);

		glColor3f(0.8, 0.8, 0.8);

		Vector2d p1 = Vector2d(max_x*0.1, i*0.1);

		coord = camera.world_to_camera(p1);
		glVertex2f(coord.x(), coord.y());

		p1 = Vector2d(min_x*0.1, i*0.1);

		coord = camera.world_to_camera(p1);
		glVertex2f(coord.x(), coord.y());

		glEnd();
	}
}


void Interface::render_edge_normals(Camera camera) {

	for (int i = 0; i < edges->size(); i++) {

		if (edges->at(i).color != 2)
			continue;

		glBegin(GL_LINES);

		glColor3f(0.3, 0.3, 1);

		int a_index = edges->at(i).a_index;
		int b_index = edges->at(i).b_index;

		Vector2d start = (pos->col(a_index) + pos->col(b_index)) * 0.5;

		Vector2d diff = pos->col(b_index) - pos->col(a_index);
		Vector2d norm = Vector2d(-diff.y(), diff.x()).normalized();

		Vector2d end = start + norm * 0.1;

		start = camera.world_to_camera(start);
		end = camera.world_to_camera(end);

		glVertex2f(start.x(), start.y());
		glVertex2f(end.x(), end.y());
		glEnd();

	}
}

void Interface::render_points(Camera camera){

	double size = 0.001f;
	glColor3f(1, 1, 1);
	Vector2d coord;

	for (int i = 0; i < pos->cols(); i++) {		

		glBegin(GL_QUADS);

		coord = camera.world_to_camera(Vector2d(pos->col(i).x() - size, pos->col(i).y() - size));
		glVertex2f(coord.x(), coord.y());

		coord = camera.world_to_camera(Vector2d(pos->col(i).x() + size, pos->col(i).y() - size));
		glVertex2f(coord.x(), coord.y());
		
		coord = camera.world_to_camera(Vector2d(pos->col(i).x() + size, pos->col(i).y() + size));
		glVertex2f(coord.x(), coord.y());
		
		coord = camera.world_to_camera(Vector2d(pos->col(i).x() - size, pos->col(i).y() + size));
		glVertex2f(coord.x(), coord.y());

		glEnd();
	}
}

void Interface::render_object_points(Camera camera) {

	double size = 0.01f;
	glColor3f(1, 1, 1);
	int index;
	Vector2d coord;


	for (int i = 0; i < objects->size(); i++) {
		for (int j = 0; j < objects->at(i)->surface_points_index.size(); j++) {
			
			index = objects->at(i)->surface_points_index[j];

			glColor3f(1, 1, 1);
			if (point_is_colliding->size() > index && point_is_colliding->at(index))
				glColor3f(0.8, 0.3, 0.3);
			
			glBegin(GL_QUADS);

			coord = camera.world_to_camera(Vector2d(pos->col(index).x() - size, pos->col(index).y() - size));
			glVertex2f(coord.x(), coord.y());

			coord = camera.world_to_camera(Vector2d(pos->col(index).x() + size, pos->col(index).y() - size));
			glVertex2f(coord.x(), coord.y());

			coord = camera.world_to_camera(Vector2d(pos->col(index).x() + size, pos->col(index).y() + size));
			glVertex2f(coord.x(), coord.y());

			coord = camera.world_to_camera(Vector2d(pos->col(index).x() - size, pos->col(index).y() + size));
			glVertex2f(coord.x(), coord.y());

			glEnd();
		}
	}
}

void Interface::render_bounding_boxes(Camera camera) {

	//glColor3f(0.25, 0.25, 0.25);
	//Vector2d coord;

	//for (int i = 0; i < objects->size(); i++) {

	//	glBegin(GL_QUADS);

	//	coord = camera.world_to_camera(objects->at(i)->bounding_box.top_left);
	//	glVertex2f(coord.x(), coord.y());

	//	coord = camera.world_to_camera(Vector2d(objects->at(i)->bounding_box.top_left.x(), objects->at(i)->bounding_box.bot_right.y()));
	//	glVertex2f(coord.x(), coord.y());

	//	coord = camera.world_to_camera(objects->at(i)->bounding_box.bot_right);
	//	glVertex2f(coord.x(), coord.y());

	//	coord = camera.world_to_camera(Vector2d(objects->at(i)->bounding_box.bot_right.x(), objects->at(i)->bounding_box.top_left.y()));
	//	glVertex2f(coord.x(), coord.y());

	//	glEnd();
	//}
}
void setColorInterval(double a1, double b1, double c1, double a2, double b2, double c2, double actuation) {
	
	double percent = 1.5 - actuation;

	if (percent < 0)
		percent = 0.0;
	if (percent > 1)
		percent = 1.0;

	glColor3f(a1*(1.0 - percent) + a2 * percent, b1*(1.0 - percent) + b2 * percent, c1*(1.0 - percent) + c2 * percent);

}
void setColorInterval(double a1, double b1, double c1, double a2, double b2, double c2, double a3, double b3, double c3, double actuation) {

	if (actuation > 1) {

		double percent = 2.0 - actuation - 0.3;
		if (percent < 0)
			percent = 0.0;
		if (percent > 1)
			percent = 1.0;

		glColor3f(a3*(1.0 - percent) + a2 * percent, b3*(1.0 - percent) + b2 * percent, c3*(1.0 - percent) + c2 * percent);
	}

	if (actuation < 1) {

		double percent = 1.0 - actuation + 0.3;
		if (percent < 0)
			percent = 0.0;
		if (percent > 1)
			percent = 1.0;

		glColor3f(a2*(1.0 - percent) + a1 * percent, b2*(1.0 - percent) + b1 * percent, c2*(1.0 - percent) + c1 * percent);
	}

	//glColor3f(a1*(1.0-percent) + a2*percent, b1*(1.0-percent) + b2*percent, c1*(1.0-percent) + c2*percent);
}
void setColorInterval(double actuation, bool is_act_vert) {

	if (actuation > 1) {

		int index = int((2.0 - actuation) * 50.0);
		if (index < 0)
			index = 0;
		if (index >= 50)
			index = 50;

		if (is_act_vert)
			glColor3f(blue[index][0], blue[index][1], blue[index][2]);
		else
			glColor3f(orange[index][0], orange[index][1], orange[index][2]);

		//glColor3f(plasma[index][0], plasma[index][1], plasma[index][2]);
	}
	if (actuation < 1) {

		int index = int((1.0 - actuation) * 50.0);
		if (index < 0)
			index = 0;
		if (index >= 50)
			index = 50;

		if (is_act_vert)
			glColor3f(blue[index+50][0], blue[index+50][1], blue[index+50][2]);
		else
			glColor3f(orange[index+50][0], orange[index+50][1], orange[index+50][2]);

		//glColor3f(plasma[index][0], plasma[index][1], plasma[index][2]);
	}
}

void Interface::render_boxels(Camera camera) {
	
	Vector2d coord;

	for (int i = 0; i < objects->size(); i++) {
		for (int j = 0; j < objects->at(i)->boxels.size(); j++) {

			Boxel* current = &objects->at(i)->boxels.at(j);

			if (current->cell_type == CELL_FIXED)
				glColor3f(0.15, 0.15, 0.15);

			if (current->cell_type == CELL_RIGID)
				glColor3f(0.15, 0.15, 0.15);
				
			if (current->cell_type == CELL_SOFT)
				glColor3f(0.75, 0.75, 0.75);



			if (current->cell_type == CELL_ACT_H) {
				//glColor3f(0.55, 0.40, 0.25);
				//glColor3f(255.0/255.0, 185.0/255.0, 56.0/255.0);

				int edge1_a_index = edges->at(current->edge_top_index).a_index;
				int edge1_b_index = edges->at(current->edge_top_index).b_index;

				int edge2_a_index = edges->at(current->edge_bot_index).a_index;
				int edge2_b_index = edges->at(current->edge_bot_index).b_index;

				Vector2d diff1 = pos->col(edge1_b_index) - pos->col(edge1_a_index);
				Vector2d diff2 = pos->col(edge2_b_index) - pos->col(edge2_a_index);
				double avg_act = (diff1.norm() + diff2.norm()) / 2.0;

				double percent = avg_act / edges->at(current->edge_top_index).act_length_eq;
			

			/*	setColorInterval(
					255.0 / 255.0, 56.0 / 255.0, 56.0 / 255.0,
					255.0 / 255.0, 255.0 / 255.0, 56.0 / 255.0,
					percent);*/

				//setColorInterval(
				//	0.3, 0.3, 0.3,
				//	255.0 / 255.0, 150.0 / 255.0, 56.0 / 255.0,
				//	0.7, 0.7, 0.7,
				//	percent);

				setColorInterval(percent, false);
			}

			

			if (current->cell_type == CELL_ACT_V) {
				//glColor3f(0.55, 0.40, 0.25);
				//glColor3f(255.0/255.0, 185.0/255.0, 56.0/255.0);

				int edge1_a_index = edges->at(current->edge_left_index).a_index;
				int edge1_b_index = edges->at(current->edge_left_index).b_index;

				int edge2_a_index = edges->at(current->edge_right_index).a_index;
				int edge2_b_index = edges->at(current->edge_right_index).b_index;

				Vector2d diff1 = pos->col(edge1_b_index) - pos->col(edge1_a_index);
				Vector2d diff2 = pos->col(edge2_b_index) - pos->col(edge2_a_index);
				double avg_act = (diff1.norm() + diff2.norm()) / 2.0;

				double percent = avg_act / edges->at(current->edge_left_index).act_length_eq;

				//setColorInterval(
				//	0.3, 0.3, 0.3,
				//	56.0 / 255.0, 150.0 / 255.0, 255.0 / 255.0,
				//	0.7, 0.7, 0.7,
				//	percent);

				//setColorInterval(
				//	56.0 / 255.0, 56.0 / 255.0, 255.0 / 255.0,
				//	56.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0,
				//	percent);

				setColorInterval(percent, true);
			}
				//glColor3f(56.0 / 255.0, 119.0 / 255.0, 222 / 255.0);

			//if(current->is_colliding)
				//glColor3f(0.65, 0.25, 0.25);


			glBegin(GL_QUADS);

			coord = camera.world_to_camera(pos->col(current->point_top_left_index));
			glVertex2f(coord.x(), coord.y());

			coord = camera.world_to_camera(pos->col(current->point_top_right_index));
			glVertex2f(coord.x(), coord.y());

			coord = camera.world_to_camera(pos->col(current->point_bot_right_index));
			glVertex2f(coord.x(), coord.y());

			coord = camera.world_to_camera(pos->col(current->point_bot_left_index));
			glVertex2f(coord.x(), coord.y());

			glEnd();
		}
	}
}


void Interface::render_encoded_boxels(Camera camera){
	
	Vector2d coord;
	for (int i = 0; i < objects->size(); i++) {
		for (int j = 0; j < objects->at(i)->boxels.size(); j++) {

			Boxel* current = &objects->at(i)->boxels.at(j);

			if (current->cell_type == CELL_EMPTY)
				continue;

			color_byte cell_color = get_encoded_color(current->cell_type);
			glColor3ub(cell_color.r, cell_color.g, cell_color.b);

			glBegin(GL_QUADS);

			coord = camera.world_to_camera(pos->col(current->point_top_left_index));
			glVertex2f(coord.x(), coord.y());

			coord = camera.world_to_camera(pos->col(current->point_top_right_index));
			glVertex2f(coord.x(), coord.y());

			coord = camera.world_to_camera(pos->col(current->point_bot_right_index));
			glVertex2f(coord.x(), coord.y());

			coord = camera.world_to_camera(pos->col(current->point_bot_left_index));
			glVertex2f(coord.x(), coord.y());

			glEnd();
		}
	}
}

Interface::color_byte Interface::get_encoded_color(int cell_type) {

	color_byte out = color_byte((GLubyte)0, (GLubyte)0, (GLubyte)0);

	cell_type -= 1;

	GLubyte color_options [8] = { (GLubyte)128, (GLubyte)64, (GLubyte)32, 
		(GLubyte)16, (GLubyte)8, (GLubyte)4, (GLubyte)2, (GLubyte)1 };

	if (cell_type >= 0 && cell_type < 8)
		out.r = color_options [cell_type % 8];

	if (cell_type >= 8 && cell_type < 16)
		out.g = color_options[cell_type % 8];

	if (cell_type >= 16 && cell_type < 24)
		out.b = color_options[cell_type % 8];

	//cout << cell_type << " " << (int)out.r << " " << (int)out.g << " " << (int)out.b << "\n";

	return out;
}

void Interface::show_debug_window() {
	glfwShowWindow(debug_window);
	debug_window_showing = true;
}

void Interface::hide_debug_window() {
	glfwHideWindow(debug_window);
	debug_window_showing = false;
}

vector<int> Interface::get_debug_window_pos() {

	int xpos, ypos;
	glfwGetWindowPos(debug_window, &xpos, &ypos);

	vector<int> out = { xpos, ypos };
	return out;
}

Interface::~Interface()
{
}

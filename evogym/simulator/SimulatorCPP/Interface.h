#ifndef INTERFACE_H
#define INTERFACE_H

#include "main.h"

//#include "GL/glew.h"
//#include "glfw3.h"
//#include "GL/gl.h"
//
//#include <vector>
//#include <Eigen/Dense>

#include "Sim.h"
#include "Environment.h"
#include "SimObject.h"
#include "Camera.h"

#include "Edge.h"

using namespace std;
using namespace Eigen;

class Interface
{
private:

	//RENDERING
	GLFWwindow* debug_window;
	bool debug_window_showing;
	Vector2d last;

	//DATA
	Matrix <double, 2, Dynamic>* pos;
	vector <Edge>* edges;
	vector <SimObject*>* objects;

	vector <bool>* point_is_colliding;

	//COLORS
	struct color_byte {

		GLubyte r;
		GLubyte g;
		GLubyte b;

		color_byte(GLubyte ra, GLubyte ga, GLubyte ba) : r(ra), g(ga), b(ba) {}
	};

	color_byte get_encoded_color(int cell_type);
	
	void render_edges(Camera camera);	
	void render_edge_normals(Camera camera);
	void render_points(Camera camera);
	void render_object_points(Camera camera);
	void render_bounding_boxes(Camera camera);
	void render_boxels(Camera camera);
	void render_grid(Camera camera);
	void render_encoded_boxels(Camera camera);


public:

	Interface(Sim* sim);
	~Interface();

	static void init();
	void render(Camera camera, bool hide_background = false, bool hide_grid = false, bool hide_edges = false, bool hide_boxels = false, bool dont_clear = false);

	void show_debug_window();
	void hide_debug_window();
	vector<int> get_debug_window_pos();

	GLFWwindow* get_debug_window_ref();
};

#endif // !INTERFACE_H


#ifndef CAMERA_H
#define CAMERA_H

#include "GL/glew.h"
#include "glfw3.h"

#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

class Camera
{
private:

	void init_gl();

public:

	Vector2d pos;
	Vector2d size;
	Vector2d scale_factor;
	Vector2d resolution;

	bool is_renderable;

	GLubyte *image_data;
	GLuint frameBuffer;
	GLuint colorBuffer;

	Camera();
	Camera(Vector2d pos, Vector2d size, Vector2d resolution, bool is_renderable);
	Camera(bool is_renderable);

	void set_pos(Vector2d pos);
	void set_size(Vector2d size);
	void set_resolution(Vector2d resolution);
	void set_pos(double x, double y);
	void set_size(double x, double y);
	void set_resolution(double x, double y);

	int get_resolution_width();
	int get_resolution_height();

	void set_focus();
	Vector2d world_to_camera(Vector2d coord);

	int get_image_data_size();
	GLubyte* get_image_data_ptr();

	

	~Camera();
};

#endif // !CAMERA_H



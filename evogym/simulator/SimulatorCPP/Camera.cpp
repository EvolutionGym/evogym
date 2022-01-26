#include "Camera.h"

Camera::Camera() {

}

Camera::Camera(Vector2d pos, Vector2d size, Vector2d resolution, bool is_renderable) {
	
	Camera::pos = pos;
	Camera::size = size;
	Camera::resolution = resolution;
	Camera::is_renderable = is_renderable;

	Camera::scale_factor = Vector2d(2.0, 2.0).cwiseQuotient(Camera::size);


	if (Camera::is_renderable) {
		init_gl();
	}
}

Camera::Camera(bool is_renderable){

	Camera::pos = Vector2d(0,0);
	Camera::size = Vector2d(1.0, 1.0);
	Camera::resolution = Vector2d(320, 320);
	Camera::is_renderable = is_renderable;

	Camera::scale_factor = Vector2d(2.0, 2.0).cwiseQuotient(Camera::size);

	if (Camera::is_renderable) {
		init_gl();
	}
}

void Camera::init_gl() {

	// New non-screen buffer for rendering
	glGenFramebuffers(1, &frameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

	//// Set the size of the frame buffer
	//glViewport(0, 0, Camera::resolution.x(), Camera::resolution.y());

	// Texture holds the color data for the frame buffer
	glGenTextures(1, &colorBuffer);
	glBindTexture(GL_TEXTURE_2D, colorBuffer);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, resolution.x(), resolution.y(), 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorBuffer, 0);

	image_data = (GLubyte*)malloc(3 * resolution.x() * resolution.y());

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		cout << "Error creating GL FBO.\n";

	if (!image_data)
		cout << "Error allocating memory for camera image.\n";
}

void Camera::set_focus() {
	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
	glBindTexture(GL_TEXTURE_2D, colorBuffer);
}

Vector2d Camera::world_to_camera(Vector2d coord) {
	if (!is_renderable)
		return (coord - pos).cwiseProduct(scale_factor);
	return (coord - pos).cwiseProduct(scale_factor).cwiseProduct(Vector2d(1.0, -1.0));
}

void Camera::set_pos(Vector2d pos) {
	Camera::pos = pos;
}

void Camera::set_size(Vector2d size) {

	Camera::size = size;
	Camera::scale_factor = Vector2d(2.0, 2.0).cwiseQuotient(Camera::size);
}

void Camera::set_resolution(Vector2d resolution) {

	Camera::resolution = resolution;

	if (!is_renderable)
		return;

	glBindTexture(GL_TEXTURE_2D, colorBuffer);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, Camera::resolution.x(), Camera::resolution.y(), 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorBuffer, 0);


	free(image_data);
	image_data = (GLubyte*)malloc(3 * resolution.x() * resolution.y());

	if (!image_data)
		cout << "Error allocating memory for camera image.\n";
}

void Camera::set_pos(double x, double y) {
	Camera::pos = Vector2d(x, y);
}

void Camera::set_size(double x, double y) {

	Camera::size = Vector2d(x, y);
	Camera::scale_factor = Vector2d(2.0, 2.0).cwiseQuotient(Camera::size);
}

void Camera::set_resolution(double x, double y) {

	Camera::resolution = Vector2d(x, y);

	if (!is_renderable)
		return;

	glBindTexture(GL_TEXTURE_2D, colorBuffer);
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, resolution.x(), resolution.y(), 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorBuffer, 0);

	free(image_data);
	image_data = (GLubyte*)malloc(3 * resolution.x() * resolution.y());

	if (!image_data)
		cout << "Error allocating memory for camera image.\n";
}


int Camera::get_resolution_width() {
	return resolution.x();
}
int Camera::get_resolution_height() {
	return resolution.y();
}


GLubyte* Camera::get_image_data_ptr() {
	return image_data;
}
int Camera::get_image_data_size() {
	return 3 * resolution.x() * resolution.y();
}

Camera::~Camera()
{
}

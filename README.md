rayTrace
========

Implements ray tracing in C++ with OpenGL used for rendering and GLUT for context creation.

The serial implementation of the ray tracer is in the ser_ray.cpp file which also includes the rendering functionality.
The parallell(CUDA) implementation is in the kernel.cu file.The vector primitives are defined in the modifiers.h header.
vShader.sh and fShader.sh are the GL shading language shaders required for rendering the image.


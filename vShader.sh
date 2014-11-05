#version 330

in vec3 vertex;
in vec3 texCoord;

uniform mat4 Projection;
uniform mat4 View;
uniform mat4 Model;

out vec2 textureCoord;


void main() {
	mat4 MVP = Projection * View * Model;
	textureCoord = texCoord.xy;
	gl_Position = MVP*vec4(vertex, 1.0);

}
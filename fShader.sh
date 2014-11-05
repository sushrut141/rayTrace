#version 330

in vec2 textureCoord;
uniform sampler2D textureSample;

out vec4 fragColor;

void main () {

	//fragColor = vec4(1.0,1.0,0.0,1.0);
	fragColor = texture(textureSample, textureCoord);
	//fragColor = textureCoord;
}
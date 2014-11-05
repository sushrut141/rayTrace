#include <GL\glew.h>
#include <GL\freeglut.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include "modifiers.h"
#include <glm.hpp>
#include <gtc\matrix_transform.hpp>
#include <gtc\type_ptr.hpp>



enum geometry  { NONE, PLANE, SPHERE };
enum material{DIFFUSE,SPECULAR,REFRACTIVE};

class Object
{
public:
	int material;
	vector3 color;
	vector3 emissive;
	int type;

	Object();
	Object(vector3& a,vector3& b):color(a),emissive(b),material(DIFFUSE){}
	Object(vector3& a,vector3& b,int m):color(a),emissive(b),material(m){}
	virtual void applyCamera(cMatrix& camera) = 0;

};

class Sphere : public Object
{
public:
	vector3 center;
	double radius;

	Sphere():center(vector3(0.0,0.0,0.0)),radius(1.0){}
	Sphere(vector3& a,double r,vector3& colors,vector3& emis,int m):center(a),radius(r),Object(colors,emis,m){this->type = SPHERE;}
	void applyCamera(cMatrix& camera);
};

void Sphere::applyCamera(cMatrix& camera)
{
	this->center = camera.mult(this->center);
}



class Plane : public Object {
    public:
	vector3 normal,normal_;
	vector3 point,point_;
	
	Plane();
	Plane(vector3 normal, vector3 point, vector3 color, vector3 emissive, int material);
	~Plane();
	void applyCamera(cMatrix& camera);
	void save();
	
};

Plane::Plane(vector3 normal, vector3 point, vector3 color, vector3 emissive, int material):normal(normal),point(point),normal_(normal),point_(point),Object(color,emissive,material){this->type=1;}
void Plane::applyCamera(cMatrix& camera) {
	vector3 p = this->point_ + this->normal_;
	this->point = camera.mult(this->point_);
	p = camera.mult(p);
	this->normal = (p - this->point).unit();
}
void Plane::save() {
	this->normal_ = this->normal;
	this->point_ = this->point;
}

class __vector {
  public:
	double x, y, z;

	__vector();
	__vector(double x, double y, double z);

	double dot(__vector w);
	double operator*(__vector w);
	__vector cross(__vector w);
	__vector add(__vector w);
	__vector sub(__vector w);
	__vector mult(double s);
	__vector h(__vector w);
	double length();
	__vector unit();
};


struct __object
{
	int type;
	vector3 center;
	vector3 color;
	vector3 emission;
	vector3 normal;
	vector3 point;
	double radius;
	int material;
};

struct _vertex_
{
	float x,y,z;
	float tx,ty;
	_vertex_():x(0.0),y(0.0),z(0.0),tx(0.0),ty(0.0){}
};


struct __ray
{
	vector3 origin;
	vector3 direction;
};


struct __intersection
{
	bool intersects;
	float t;

	__ray ray;
	vector3 normal;
};


///////////global variables/////////
float width = 640.0,height = 320.0;
unsigned char* frame_buffer_host = NULL;
unsigned char* frame_buffer_device = NULL;
double* samples_buffer = NULL;
__object* objects_buffer = NULL;
float* rand_ = NULL;
float *__rand = NULL;
float* rand_bounce=NULL;

__object* _obj;
unsigned int length;

/////////////////////Device Function Declarations////////////////////
extern "C" bool initRayTracer(int,int,unsigned char*&,unsigned char*&,double*&,__object*&,int,__object*&);
extern "C" void launchKernel(int,int,unsigned char*&,unsigned char*&,double*&,float*&,float*&,__object*&,int);
//extern "C" void randCheck(float*,float*);
extern "C" void randGen(float*&,float*&);

void serRayTrace(float width,float height,float *rand_host,unsigned char* frame_host);
vector3 newRayTrace(__ray,__object*,float*rand_host,int, int,int);
double erand();
////////////Rendering functions//////////

bool initGL(int,char**,int,int);
void createVBO();
GLuint initTexture(unsigned char*&,int,int);
std::string readShaderCode(const char*);
void initProgram(GLuint*, GLuint , GLuint, const char*, const char*);
void attribLocation();
void display();
void reshape(int,int);

void printFile(unsigned char*&);
///////////atrributes////////////
GLuint vertexShader,fragmentShader,program,tid,vao;
GLuint vertexID,indexID,texID,tex_loc;	

//////////atrribute locations///////
GLint vertexLoc,TexCoord_loc,textureSample;


int main(int argc,char**argv)
{

	 bool tr = initGL(argc,argv,width,height);
	 if(tr!=true){
		 std::cout<<"rendering context not created";
		 exit(-1);
	 }
	 initProgram(&program,vertexShader,fragmentShader,"vShader.sh","fShader.sh");
		std::cout<<"Program ID"<<" "<<program<<std::endl;
	 createVBO();
		std::cout<<"vertex ID"<<" "<<vertexID<<std::endl;
		std::cout<<"tex ID"<<" "<<texID<<std::endl;

	float width = 640;
	float height = 320;
	
	std::vector<Object*> objects;


	objects.push_back(new Plane(vector3( 1, 0, 0), vector3(-456,   0,   0), vector3(0.75, 0.25, 0.25), vector3(), DIFFUSE)); // left
	objects.push_back(new Plane(vector3(-1, 0, 0), vector3( 456,   0,   0), vector3(0.25, 0.25, 0.75), vector3(), DIFFUSE)); // right
	objects.push_back(new Plane(vector3( 0, 0, 1), vector3(   0,   0,-456), vector3(0.75, 0.75, 0.75), vector3(), DIFFUSE)); // far
	objects.push_back(new Plane(vector3( 0, 0,-1), vector3(   0,   0,2456), vector3(), vector3(4,4,4), DIFFUSE)); // near
	objects.push_back(new Plane(vector3( 0,-1, 0), vector3(   0, 256,   0), vector3(0.75, 0.75, 0.75), vector3(), DIFFUSE)); // top
	objects.push_back(new Plane(vector3( 0, 1, 0), vector3(   0,-256,   0), vector3(), vector3(4,4,4), DIFFUSE)); // bottom
	objects.push_back(new Sphere(vector3(-228,-106,-128), 150, vector3(0.6509,0.390567,0.6509), vector3(),         DIFFUSE  ));
	objects.push_back(new Sphere(vector3( 228,-106, 0), 150, vector3(0.6509,0.6509,0.390567), vector3(),         DIFFUSE));
	//objects.push_back(new Sphere(vector3( 356, 1024*4+240.0, -356), 1024*4, vector3(0,0,0),           vector3(5,5,5), DIFFUSE  )); // light source
	objects.push_back(new Sphere(vector3( 0, 1024*4+240.0, 0), 1024*4, vector3(0,0,0),           vector3(4,4,4), DIFFUSE  )); // light source
	//objects.push_back(new Sphere(vector3(-128, 256,  28), 128, vector3(0,0,0),           vector3(10,10,10), DIFFUSE  )); // light source



	
	vector3  origin(-232,164,1455);

		vector3 forward = (vector3(0,0,0)-origin).unit();
		vector3    up(0, 1,  0);
		vector3 right = (forward.cross(up)).unit();

		up = (right.cross(forward)).unit();

		double cam[] = {  right.x,   right.y,   right.z,	-(right*origin),
						  up.x,      up.y,      up.z,		-(up*origin),
						  forward.x, forward.y, forward.z,  -(forward*origin),
						   0,         0,         0,         1 };

		for(int i=0;i<4;i++){
			std::cout<<std::endl;
			for(int j=0;j<4;j++)
				std::cout<<cam[i*4 + j]<<" ";
		}

		cMatrix camera(4, 4, cam);

	for(unsigned int i=0;i<objects.size();i++)
		objects[i]->applyCamera(camera);
	//std::cout<<"sphere center"<<std::endl;
	//std::cout<<((Sphere*)objects[0])->center.x<<" "<<((Sphere*)objects[0])->center.y<<" "<<((Sphere*)objects[0])->center.z<<std::endl;

	  length = objects.size();
	  _obj = new __object[sizeof(__object)*length];

	 for (unsigned int i = 0; i < length; i++) {

				_obj[i].type = objects[i]->type;
				_obj[i].material = objects[i]->material;

				_obj[i].color.x = objects[i]->color.x;
				_obj[i].color.y = objects[i]->color.y;
				_obj[i].color.z = objects[i]->color.z;

				_obj[i].emission.x = objects[i]->emissive.x;
				_obj[i].emission.y = objects[i]->emissive.y;
				_obj[i].emission.z = objects[i]->emissive.z;

				if (objects[i]->type == SPHERE) {

					_obj[i].center.x = ((Sphere *)objects[i])->center.x;
					_obj[i].center.y = ((Sphere *)objects[i])->center.y;
					_obj[i].center.z = ((Sphere *)objects[i])->center.z;
					_obj[i].radius   = ((Sphere *)objects[i])->radius;

				} else {
					_obj[i].normal.x = ((Plane *)objects[i])->normal.x;
					_obj[i].normal.y = ((Plane *)objects[i])->normal.y;
					_obj[i].normal.z = ((Plane *)objects[i])->normal.z;
					_obj[i].point.x  = ((Plane *)objects[i])->point.x;
					_obj[i].point.y  = ((Plane *)objects[i])->point.y;
					_obj[i].point.z  = ((Plane *)objects[i])->point.z;
				}
			}

	 vertexLoc = glGetAttribLocation(program,"vertex");
		std::cout<<"vertex"<<" "<<vertexLoc<<std::endl;
	 TexCoord_loc = glGetAttribLocation(program,"texCoord");
		std::cout<<"texCoord"<<" "<<TexCoord_loc<<std::endl;
	 GLint m = glGetUniformLocation(program,"Model");
		std::cout<<"Model"<<" "<<m<<std::endl;
	 GLint v = glGetUniformLocation(program,"View");
		std::cout<<"View"<<" "<<v<<std::endl;
	 GLint p = glGetUniformLocation(program,"Projection");
		std::cout<<"Projection"<<" "<<p<<std::endl;
	 textureSample = glGetUniformLocation(program, "textureSample");
		std::cout<<"textureSample"<<" "<<textureSample<<std::endl;
	 glUniform1i(textureSample,0);


	 glm::mat4 projection = glm::ortho(0.0f, (float)width, (float)height, 0.0f, -5.0f, 5.0f); 
	 glm::mat4 view       = glm::mat4(1.0f);
	 glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f,0.0f,-2.0f));
	 
	 glUniformMatrix4fv(m,1,GL_FALSE,&model[0][0]);
	 
	 
	 glUniformMatrix4fv(v,1,GL_FALSE,&view[0][0]);
	 
	 
	 glUniformMatrix4fv(p,1,GL_FALSE,&projection[0][0]);

	 /////Device functions/////////
	 if(!initRayTracer(width,height,frame_buffer_host,frame_buffer_device,samples_buffer,objects_buffer,length,_obj)){
		 std::cout<<" Memory allocation failed"<<std::endl;
		 exit(-1);
	 }
	 int size_r = sizeof(float)*640*320*88;
	 float* rand_ = (float*)malloc(size_r);
	 randGen(rand_,rand_bounce);

	 launchKernel(width,height,frame_buffer_device,frame_buffer_host,samples_buffer,rand_,rand_bounce,objects_buffer,length);

	 //serRayTrace(width,height,rand_,frame_buffer_host);

	 
	 printFile(frame_buffer_host);
	 //////////////Rendering Functions/////////////////
	 std::cout<<"out "<<(int)frame_buffer_host<<std::endl;
	 //tex_loc = initTexture(frame_buffer_host,width,height);
	 glActiveTexture(GL_TEXTURE0);
	 glGenTextures(1,&tex_loc);
	 glBindTexture(GL_TEXTURE_2D,tex_loc);
	 glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,width,height,0,GL_RGB,GL_UNSIGNED_BYTE,frame_buffer_host);
	 glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	 glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		std::cout<<"texture"<<" "<<tex_loc<<std::endl;
	 

	 glutDisplayFunc(display);
	 glutReshapeFunc(reshape);
	 glutMainLoop();

}


bool initGL(int argc,char**argv,int width,int height)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE);
	glutInitWindowSize(width,height);
	glutCreateWindow("Ray_Trace");
	glewInit();
	return true;
}
	
GLuint initTexture(unsigned char*& frame_buffer,int width,int height)
{
	std::cout<<"in "<<(int)frame_buffer<<std::endl;
	glActiveTexture(GL_TEXTURE0);
	GLuint texID;
	glGenTextures(1,&texID);
	glBindTexture(GL_TEXTURE_2D,texID);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,width,height,0,GL_RGB,GL_UNSIGNED_BYTE,frame_buffer);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	return texID;
}

void initProgram(GLuint* glProgram, GLuint glShaderV, GLuint glShaderF, const char* vertex_shader, const char* fragment_shader) {

	glShaderV = glCreateShader(GL_VERTEX_SHADER);
	glShaderF = glCreateShader(GL_FRAGMENT_SHADER);
	const char* adapter[1];

	std::string vShaderSource = readShaderCode(vertex_shader);
	std::string fShaderSource = readShaderCode(fragment_shader);

	adapter[0] = vShaderSource.c_str();
	glShaderSource(glShaderV, 1, adapter, NULL);
	adapter[0] = fShaderSource.c_str();
	glShaderSource(glShaderF, 1, adapter, NULL);

	glCompileShader(glShaderV);
	glCompileShader(glShaderF);
	*glProgram = glCreateProgram();

	//glBindAttribLocation(glProgram,0,"vertex");
	//glBindAttribLocation(glProgram,1,"texCoord");

	std::cout<<"Program ID"<<" "<<glProgram<<std::endl;

	glAttachShader(*glProgram, glShaderV);
	glAttachShader(*glProgram, glShaderF);

	glLinkProgram(*glProgram);
	glUseProgram(*glProgram);
	
	int  vlength,    flength,    plength,link;
	char vlog[2048], flog[2048], plog[2048];
	glGetShaderInfoLog(glShaderV, 2048, &vlength, vlog);
	glGetShaderInfoLog(glShaderF, 2048, &flength, flog);
	glGetProgramInfoLog(*glProgram, 2048, &flength, plog);
	glGetProgramiv(*glProgram,GL_LINK_STATUS,&link);
	if(!link)
		std::cout<<"link error"<<std::endl;

	std::cout << vlog << std::endl << std::endl << flog << std::endl << std::endl << plog << std::endl << std::endl;
}


std::string readShaderCode(const char* file)
{
	std::ifstream input(file);
	if(!input.good())
	{
		std::cout<<"File failed to Load"<<file<<std::endl;
		exit(1);
	}
	return std::string(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>());
}

void createVBO()
{
	 const int count  = 4;
	 _vertex_ vertices[count];
	 vertices[0].x = 0.0;   vertices[0].y = 0.0;    vertices[0].z = 0.0;	vertices[0].tx = 0.0;   vertices[0].ty = 1.0;   
	 vertices[1].x = width; vertices[1].y = 0.0;    vertices[1].z = 0.0;	vertices[1].tx = 1.0;	vertices[1].ty = 1.0;   
	 vertices[2].x = width; vertices[2].y = height; vertices[2].z = 0.0;	vertices[2].tx = 1.0;	vertices[2].ty = 0.0;
	 vertices[3].x = 0.0;   vertices[3].y = height; vertices[3].z = 0.0;	vertices[3].tx = 0.0;   vertices[3].ty = 0.0;

	/* _vertex_ tex[count];
	 tex[0].x = 0.0;  tex[0].y = 1.0;   tex[0].z = 0.0;
	 tex[1].x = 1.0;  tex[1].y = 0.0;   tex[1].z = 0.0;
	 tex[2].x = 1.0;  tex[2].y = 1.0;   tex[2].z = 0.0;
	 tex[3].x = 0.0;  tex[3].y = 0.0;   tex[3].z = 0.0;*/

	 ///////////Vertex and texture  buffer//////////////////
	 
	 glGenBuffers(1,&vertexID);
	 glBindBuffer(GL_ARRAY_BUFFER,vertexID);
	 glBufferData(GL_ARRAY_BUFFER,sizeof(_vertex_)*count,vertices,GL_DYNAMIC_DRAW);
	 
	 /*glGenBuffers(1,&texID);
	 glBindBuffer(GL_ARRAY_BUFFER,texID);
	 glBufferData(GL_ARRAY_BUFFER,sizeof(_vertex_)*count,tex,GL_DYNAMIC_DRAW);*/

	 ///////////Index buffer///////////////////
	 const int idxcount = 6;
	 unsigned short indices[idxcount] = {0,1,2,2,3,0};
	 glGenBuffers(1,&indexID);
	 glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,indexID);
	 glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(unsigned short)*idxcount,indices,GL_STATIC_DRAW);
		std::cout<<"index id "<<indexID<<std::endl;

}



void display()
{
	//launchKernel(width,height,frame_buffer_device,frame_buffer_host,samples_buffer,rand_,rand_bounce,objects_buffer,4);
	 glViewport(0,0,width,height);
	 glClearColor(1.0,0.0,0.0,1.0);
	 glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	 glBindBuffer(GL_ARRAY_BUFFER,vertexID);
	 glEnableVertexAttribArray(vertexLoc);
	 glVertexAttribPointer(vertexLoc,3,GL_FLOAT,GL_FALSE,sizeof(_vertex_),0);

	 glEnableVertexAttribArray(TexCoord_loc);
	 glVertexAttribPointer(TexCoord_loc,2,GL_FLOAT,GL_FALSE,sizeof(_vertex_),(char*)+12);
	 
	 glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,indexID);

	  ////////texture sample//////
	 //glActiveTexture(GL_TEXTURE0);
	 //glBindTexture(GL_TEXTURE_2D,tex_loc);

	 glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,0);
	 glUseProgram(program);
	 glutSwapBuffers();

}

void reshape(int w ,int h)
{
	glViewport(0,0,w,h);
	width = w;
	height = h;
}

void printFile(unsigned char*& buffer)
{
	FILE* s;
	s = fopen("output.txt","w");
	if(s==NULL)
		std::cout<<"File not opened"<<std::endl;
	for(int i=0;i<100;i++){
		fprintf(s,"\n");
		for(int j=0;j<100;j++)
		{
			long index = i*100*3 + j*3;
			fprintf(s,"%d  ,%d  ,%d   ",buffer[index+0],buffer[index+1],buffer[index+2]);
		}
	}
	fclose(s);
}

void serRayTrace(float width,float height,float *rand_host,unsigned char* frame_host)
{
	for(int j=0;j<height;j++)
	{
		for(int i=0;i<width;i++)
		{

			int index = j*width*3 + i*3;
			int _index = j*width + i;

			vector3 samples;

			for(int k=0;k<4;k++)
			{
				float u1 = rand_host[k*640*320 + 2*_index + 0];
				float u2 = rand_host[k*640*320 + 2*_index + 1];
				float r1 = 2 * 3.14 * u1;
				float r2 = sqrt(1 - u2);
				float r3 = sqrt(u2);
				vector3 offset = vector3(cos(r1)*r2, sin(r1)*r2, r3) * 0.5;

				if (k == 0) offset = offset + vector3( 0.0, 0.0, 0.0);
				if (k == 1) offset = offset + vector3( 0.0, 0.5, 0.0);
				if (k == 2) offset = offset + vector3( 0.5, 0.0, 0.0);
				if (k == 3) offset = offset + vector3( 0.5, 0.5, 0.0);

				__ray ray = {vector3(0.0,0.0,0.0),vector3((i-width/2),(-j+height/2),width) + offset};

				vector3 sample = newRayTrace(ray,_obj,rand_host,(k*640*320*20 + _index*20), width*height*8,length);

				samples = samples + (sample * 0.25);
			}

			frame_host[index + 0] = (samples.x*255)>255?255:(unsigned char)(samples.x*255);
			frame_host[index + 1] = (samples.y*255)>255?255:(unsigned char)(samples.y*255);
			frame_host[index + 2] = (samples.z*255)>255?255:(unsigned char)(samples.z*255);

			std::cout<<"Done till Pixel : "<<" "<<samples.x<<" "<<samples.y<<" "<<samples.z<<" "<<j*width + i<<std::endl;

		}
	}
}


 vector3 newRayTrace(__ray ray,__object* object,float*rand_bounce,int index,int size,int count)
{
	float epsilon = 0.000001;
	__intersection intersect;
	__intersection isect;

	vector3 __a[10],__b[10],sample;

	for (int l = 0; l < 10; l++) {
		__a[l].x = __a[l].y = __a[l].z = 0;
		__b[l].x = __b[l].y = __b[l].z = 0;
	}

	intersect.intersects = false;
	int which = -1,m=10;

	for(int l=0;l<m;l++)
	{
		ray.direction = ray.direction.unit();

		//std::cout<<ray.direction.x<<" "<<ray.direction.y<<" "<<ray.direction.z<<std::endl;

		for(int k=0;k<count;k++)
		{
			if(object[k].type==SPHERE)
			{
			double a  = ray.direction*ray.direction;
			double b = (ray.direction * ray.origin - ray.direction * object[k].center) * 2.0;
			double c = ray.origin * ray.origin + object[k].center * object[k].center - ray.origin * object[k].center * 2.0 - object[k].radius * object[k].radius;
			//float c = ((ray.origin - object[k].center)*(ray.origin - object[k].center)) - (object[k].radius*object[k].radius);

			//std::cout<<"values "<<a<<" "<<b<<" "<<c<<std::endl;


			if(a<epsilon)
				std::cout<<"Hey  what happened"<<std::endl;

			double det = (b*b) -(4.0*a*c);
			if(det<epsilon)continue;
			if(det!=det)continue;

			//std::cout<<" Determinant "<<det<<std::endl;

			double t0 = (-b + sqrt(det))/2*a;
			double t1 = (-b - sqrt(det))/2*a;
			if (t0 < epsilon && t1 < epsilon) continue;

			isect.intersects = true;
			isect.t = t0 < epsilon ? t1 : (t1 < epsilon ? t0 : (t0 < t1 ? t0 : t1));
			if(isect.t!=isect.t)continue;
			//std::cout<<"The t value is  "<<isect.t<<std::endl;

			isect.ray.origin = ray.origin + ray.direction*isect.t;
			isect.normal = (isect.ray.origin - object[k].center).unit();
			}

			else if (object[k].type == PLANE) {

				double den = ray.direction * object[k].normal;
				if (fabs(den) < epsilon) continue;

				vector3 temp = object[k].point - ray.origin;
				double num = temp * object[k].normal;
				double num_den = num/den;
				if (num_den < epsilon) continue;

				isect.intersects = true;
				isect.t = num_den;

				isect.ray.origin = ray.origin + ray.direction * isect.t;

				isect.normal = object[k].normal.unit();

			}

			if (isect.intersects) {
				if (!intersect.intersects || isect.t < intersect.t) {
					intersect = isect;
					which = k;
				}
			}
		
	}

		if (intersect.intersects) {

			__a[l] = object[which].emission;
			__b[l] = object[which].color;
		
			intersect.ray.direction = (ray.direction - intersect.normal * (ray.direction * intersect.normal * 2.0)).unit();

			/////////
			if (object[which].material == DIFFUSE) { //DIFFUSE

				vector3 w = intersect.normal;

				// cosine weighted sampling
//				float u1 = rand_device[index+0];
//				float u2 = rand_device[index+1];
				float u1 = erand();
				float u2 = erand();

				float r1 = 2.0 * 3.14 * u1;
				float r2 = sqrt(1 - u2);
				float r3 = sqrt(u2);

				vector3 u(0,0,0);
				if      (fabs(w.x) < fabs(w.y) && fabs(w.x) < fabs(w.z)) u.x = 1;
				else if (fabs(w.y) < fabs(w.x) && fabs(w.y) < fabs(w.z)) u.y = 1;
				else u.z = 1;

				u = u.cross(w).unit();
				vector3 v = w.cross(u).unit();
					 u = v.cross(w).unit();
				vector3 d = ( u * (cos(r1) * r2) + v * (sin(r1) * r2) + w * r3 ) .unit();

				intersect.ray.direction = d;

				ray = intersect.ray;
				
			} else if (object[which].material == SPECULAR) { //SPECULAR

				ray = intersect.ray;
				
			} else if (object[which].material == REFRACTIVE) { //REFRACTIVE

				bool into = ray.direction * intersect.normal < 0; // entering the medium

				float n1n2 = into ? (1.0/1.5) : (1.5/1.0);
				vector3 n  = into ? intersect.normal : (intersect.normal * -1);
				vector3 r = ray.direction;

				float n1n22 = n1n2 * n1n2;
				float rn   = r * n;
				float rn2  = rn * rn;
				
				float a = 1 - n1n22 * (1 - rn2);
				if (a >= 0) {
					ray.origin = intersect.ray.origin;
					ray.direction = r * n1n2 - n * (n1n2 * rn + sqrt(a));
					//std::cout<<"its refractive "<<ray.direction.x<<" "<<ray.direction.y<<" "<<ray.direction.z<<std::endl;
				} else ray = intersect.ray; // total internal reflection

			}
			
		} else break;
	}

	sample = __a[m - 1];
	for (int l = m-2; l >= 0; l--) sample = __a[l] + __b[l].h(sample);

	return sample;
}

double erand()
{
	return ((double) rand() / (RAND_MAX));
}
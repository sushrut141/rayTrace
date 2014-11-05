#include "modifiers.h"
#include <math.h>
#include <cstdlib>



double vector3::operator*(vector3& a){
	return  this->x*a.x + this->y*a.y  + this->z*a.z;
}

vector3 vector3::operator+(vector3& a){
	return vector3(this->x+a.x,this->y+a.y,this->z+a.z);
}

vector3 vector3::operator-(vector3& a){
	return vector3(this->x-a.x,this->y-a.y,this->z-a.z);
}

double vector3::dot(vector3& a){
	return this->x*a.x + this->y*a.y + this->z*a.z;
}

double vector3::mag(){
	return sqrt(this->x*this->x + this->y*this->y + this->z*this->z);
}

vector3 vector3::operator*(double a)
{
	return vector3(this->x*a,this->y*a,this->z*a);
}

vector3 vector3::unit()
{
	double a = this->mag();
	return vector3((this->x)/a,(this->y)/a,(this->z)/a);
}

vector3 vector3::h(vector3& a)
{
	return vector3(this->x*a.x,this->y*a.y,this->z*a.z);
}



vector3 vector3::cross(vector3& v) {
	return vector3(this->y*v.z - this->z*v.y, this->z*v.x - this->x*v.z, this->x*v.y - this->y*v.x);
}



cMatrix::cMatrix(const int m, const int n) : m(m), n(n) {
	A = new double[m * n];
	this->eye();
	
}

cMatrix::cMatrix(const int m, const int n, const double A[]) : m(m), n(n) {
	this->A = new double[m * n];
	for (int i = 0; i < m * n; i++) this->A[i] = A[i];
}

cMatrix& cMatrix::eye() {
	for (int j = 0; j < m; j++)
		for (int i = 0; i < n; i++)
			A[j * n + i] = i == j ? 1.0 : 0.0;
	return *this;
}

vector3 cMatrix::mult(const vector3& v) {
	vector3 r;
	if (m == 4 && n == 4) {
		r.x = this->A[0] * v.x + this->A[1] * v.y + this->A[2]  * v.z + this->A[3]  * 1;
		r.y = this->A[4] * v.x + this->A[5] * v.y + this->A[6]  * v.z + this->A[7]  * 1;
		r.z = this->A[8] * v.x + this->A[9] * v.y + this->A[10] * v.z + this->A[11] * 1;
	}
	return r;
}

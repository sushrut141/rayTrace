#ifndef MODIFIERS_H
#define MODIFIERS_H

class vector3
{
public:
	double x,y,z;

	vector3():x(0.0),y(0.0),z(0.0){};
	vector3(double a,double b,double c):x(a),y(b),z(c){}

	vector3 operator+(vector3& a);
	double operator*(vector3& a);           
	vector3 operator-(vector3& a);
	vector3 operator*(double a);
	
	double dot(vector3& a);
	vector3 cross(vector3& a);
	vector3 h(vector3& a);
	double mag();
	vector3 unit();

};

class cMatrix 
{
  private:
  protected:
  public:
	double *A;
	int m, n;

	cMatrix(const int m, const int n);
	cMatrix(const int m, const int n, const double A[]);
	cMatrix& eye();
	vector3 mult(const vector3& v);
};


#endif
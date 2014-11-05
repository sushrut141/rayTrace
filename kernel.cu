#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <helper_cuda.h>
#include <helper_functions.h>


enum geometry  { NONE, PLANE, SPHERE };
enum material{DIFFUSE,SPECULAR,REFRACTIVE};


class __vector
{
public:
	float x,y,z;
	__device__ __vector();
	__device__ __vector(float a,float b,float c);

	__device__ float dot(__vector a);
	__device__ __vector operator+(__vector a);
	__device__ __vector operator-(__vector a);
	__device__ float operator*(__vector a);
	__device__ __vector operator*(float s);
	__device__ __vector cross(__vector w);
	__device__ __vector h(__vector w);

	__device__ float mag();
	__device__ __vector unit();
};

__device__ __vector::__vector():x(0.0),y(0.0),z(0.0){}
__device__ __vector::__vector(float a,float b,float c):x(a),y(b),z(c){}

__device__ float __vector::dot(__vector a){ return this->x*a.x + this->y*a.y + this->z*a.z;}
__device__ __vector __vector::operator+(__vector a){return __vector(this->x+a.x,this->y+a.y,this->z+a.z);}
__device__ float __vector::operator*(__vector a){return this->x*a.x + this->y*a.y + this->z*a.z;}
__device__ __vector __vector::operator*(float s){return __vector(this->x*s,this->y*s,this->z*z);}
__device__ __vector __vector::operator-(__vector a){return __vector(this->x-a.x,this->y-a.y,this->z-a.z);}
__device__ __vector __vector::cross(__vector w) { return __vector(this->y * w.z - this->z * w.y, this->z * w.x - this->x * w.z, this->x * w.y - this->y * w.x); }
__device__ __vector __vector::h(__vector w){return __vector(this->x*w.x,this->y*w.y,this->z*w.z);}

__device__ float __vector::mag(){ return (this->x*this->x + this->y*this->y + this->z*this->z);}
__device__ __vector __vector::unit(){
	float mag = sqrt(this->x*this->x + this->y*this->y + this->z*this->z);
	return __vector(this->x/mag,this->y/mag,this->z/mag);
}


struct __object
{
	int type;
	__vector center;
	__vector color;
	__vector emission;
	__vector normal;
	__vector point;
	double radius;
	int material;
};

struct __ray
{
	__vector origin;
	__vector direction;
};


struct __intersection
{
	bool intersects;
	float t;

	__ray ray;
	__vector normal;
};
//extern __shared__ __vector __a[10],__b[10],sample;


__device__ __vector rayTrace(__ray ray,__object* object,float*rand_bounce,int index,int size,int count,__vector* __a,__vector* __b)
{
	float epsilon = 0.000001;
	__intersection intersect;
	__intersection isect;

	//__vector __a[10],__b[10],sample;
	__vector sample;

	for (int l = 0; l < 10; l++) {
		__a[l].x = __a[l].y = __a[l].z = 0;
		__b[l].x = __b[l].y = __b[l].z = 0;
	}

	intersect.intersects = false;
	int which = -1,m=10;

	for(int l=0;l<m;l++)
	{
		ray.direction = ray.direction.unit();

		for(int k=0;k<count;k++)
		{
			if(object[k].type==SPHERE)
			{
			double a  = ray.direction*ray.direction;
			double b = (ray.direction * ray.origin - ray.direction * object[k].center) * 2.0;
			double c = ray.origin * ray.origin + object[k].center * object[k].center - ray.origin * object[k].center * 2.0 - object[k].radius * object[k].radius;
			//float c = ((ray.origin - object[k].center)*(ray.origin - object[k].center)) - (object[k].radius*object[k].radius);

			double det = (b*b) -(4.0*a*c);
			if(det<epsilon)continue;

			double t0 = (-b + sqrt(det))/2*a;
			double t1 = (-b - sqrt(det))/2*a;
			if (t0 < epsilon && t1 < epsilon) continue;

			isect.intersects = true;
			isect.t = t0 < epsilon ? t1 : (t1 < epsilon ? t0 : (t0 < t1 ? t0 : t1));

			isect.ray.origin = ray.origin + ray.direction*isect.t;
			isect.normal = (isect.ray.origin - object[k].center).unit();
			}

			else if (object[k].type == PLANE) {

				double den = ray.direction * object[k].normal;
				if (fabs(den) < epsilon) continue;

				__vector temp = object[k].point - ray.origin;
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

				__vector w = intersect.normal;

				// cosine weighted sampling
//				float u1 = rand_device[index+0];
//				float u2 = rand_device[index+1];
				float u1 = rand_bounce[index + 2*l + 0];
				float u2 = rand_bounce[index + 2*l + 1];

				float r1 = 2 * 3.14 * u1;
				float r2 = sqrt(1 - u2);
				float r3 = sqrt(u2);

				__vector u(0,0,0);
				if      (fabs(w.x) < fabs(w.y) && fabs(w.x) < fabs(w.z)) u.x = 1;
				else if (fabs(w.y) < fabs(w.x) && fabs(w.y) < fabs(w.z)) u.y = 1;
				else u.z = 1;

				u = u.cross(w).unit();
				__vector v = w.cross(u).unit();
					 u = v.cross(w).unit();
				__vector d = ( u * (cos(r1) * r2) + v * (sin(r1) * r2) + w * r3 ) .unit();

				intersect.ray.direction = d;

				ray = intersect.ray;
				
			} else if (object[which].material == SPECULAR) { //SPECULAR

				ray = intersect.ray;
				
			} else if (object[which].material == REFRACTIVE) { //REFRACTIVE

				bool into = ray.direction * intersect.normal < 0; // entering the medium

				float n1n2 = into ? (1.0/1.5) : (1.5/1.0);
				__vector n  = into ? intersect.normal : (intersect.normal * -1);
				__vector r = ray.direction;
				
				float n1n22 = n1n2 * n1n2;
				float rn   = r * n;
				float rn2  = rn * rn;
				
				float a = 1 - n1n22 * (1 - rn2);
				if (a >= 0) {
					ray.origin = intersect.ray.origin;
					ray.direction = r * n1n2 - n * (n1n2 * rn + sqrt(a));
				} else ray = intersect.ray; // total internal reflection

			}
			
		} else break;
	}

	sample = __a[m - 1];
	for (int l = m-2; l >= 0; l--) sample = __a[l] + __b[l].h(sample);

	return sample;
}


__global__ void kernel(unsigned char* frame_device,double* samples_buffer,float*rand_bounce,__object* _obj,int width,int height,int length)
{
	int tx = threadIdx.x + blockIdx.x*blockDim.x;
	int ty = threadIdx.y + blockIdx.y*blockDim.y;

	int index = ty*width*3 + tx*3;
	int _index = ty*width + tx;

	__vector samples;

	__shared__ __vector __a[10],__b[10];


	for (int i = 0; i < 4; i++) {
		
		float u1 = rand_bounce[i*640*320 + 2*_index + 0];
		float u2 = rand_bounce[i*640*320 + 2*_index + 1];
		float r1 = 2 * 3.14 * u1;
		float r2 = sqrt(1 - u2);
		float r3 = sqrt(u2);
		__vector offset = __vector(cos(r1)*r2, sin(r1)*r2, r3) * 0.5;

		if (i == 0) offset = offset + __vector( 0.0, 0.0, 0.0);
		if (i == 1) offset = offset + __vector( 0.0, 0.5, 0.0);
		if (i == 2) offset = offset + __vector( 0.5, 0.0, 0.0);
		if (i == 3) offset = offset + __vector( 0.5, 0.5, 0.0);

		__ray ray = {__vector(0.0,0.0,0.0),__vector((tx-width/2.0),(-ty+height/2.0),width) + offset};

		__vector sample = rayTrace(ray,_obj,rand_bounce,(i*640*320*20 + _index*20), width*height*8,length,__a,__b);


		samples = samples + (sample * 0.25);
		__syncthreads();
	}
	
	/*samples_buffer[index + 0] = samples.x;
	samples_buffer[index + 1] = samples.y;
	samples_buffer[index + 2] = samples.z;
*/
	frame_device[index + 0] = (samples.x*255)>255?255:(unsigned char)(samples_buffer[index+0]*255);
	frame_device[index + 1] = (samples.y*255)>255?255:(unsigned char)(samples_buffer[index+1]*255);
	frame_device[index + 2] = (samples.z*255)>255?255:(unsigned char)(samples_buffer[index+2]*255);

}



extern "C" bool initRayTracer(int width,int height,unsigned char*& frame_host,unsigned char*& frame_device,double*& sample_buffer,__object*& objects_buffer,int count,__object*& _obj)
{
	cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize,sizeof(float)*640*320*8);
	if(err!=cudaSuccess)	
		printf("%s\n",cudaGetErrorString(err));
	size_t heap;
	cudaDeviceGetLimit(&heap,cudaLimitMallocHeapSize);
	printf("%d\n",heap);


	//cudaDeviceReset();
	size_t heap_size;
	//cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
	

	int size = sizeof(unsigned char)*width*height*3;
	frame_host = (unsigned char*)malloc(size);

	checkCudaErrors(cudaMalloc((void**)&frame_device,size));

	int __size = sizeof(double)*width*height*3;
	cudaMalloc((void**)&sample_buffer,__size);

	int _size = sizeof(__object)*count;
	cudaMalloc((void**)&objects_buffer,_size);
	checkCudaErrors(cudaMemcpy(objects_buffer,_obj,_size,cudaMemcpyHostToDevice));

	//printf("%f",rand_device[10]);
	return true;

}


extern "C" void launchKernel(int width,int height,unsigned char*& frame_device,unsigned char*& frame_host,double*& samples_buffer,float*& rand_device,float*& rand_bounce,__object*& _obj,int count)
{
	dim3 dimGrid(width/16,height/16,1);
	dim3 dimBlock(16,16);

	//float* host = (float*)malloc(600*sizeof(float));
	kernel<<<dimGrid,dimBlock>>>(frame_device,samples_buffer,rand_bounce,_obj,width,height,count);
	

	cudaError_t err = cudaGetLastError();
	printf("Error : %s\n",cudaGetErrorString(err));
	
	
	checkCudaErrors(cudaMemcpy(frame_host,frame_device,sizeof(unsigned char)*width*height*3,cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(host,rand_bounce,60*sizeof(float),cudaMemcpyDeviceToHost));
	cudaFree(frame_device);
	cudaFree(samples_buffer);
	cudaFree(rand_device);
	cudaFree(_obj);
}



extern "C" void randGen(float*& rand_device,float*& rand_bounce)
{
	int size_r = sizeof(float)*640*320*88;
	//cudaMalloc((void**)&rand_device,size_r);
	/*curandGenerator_t gen,gen1;
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(gen,200);
	curandGenerateUniform(gen,rand_device,size_r);
	curandDestroyGenerator(gen);*/

	/*for(int i=0;i<10;i++)
		printf(" the %d th number : %f\n",i,rand_device[i]);*/

	checkCudaErrors(cudaMalloc((void**)&rand_bounce,size_r));

	curandGenerator_t gen1;
	curandCreateGenerator(&gen1,CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(gen1,300);
	curandGenerateUniform(gen1,rand_bounce,640*320*88);
	cudaThreadSynchronize();
	curandDestroyGenerator(gen1);

	cudaMemcpy(rand_device,rand_bounce,size_r,cudaMemcpyDeviceToHost);
	//for(int i=0;i<640*80;i++)
	//	printf(" the %d th number : %f\n",i,host[i]);
	//free(host);
}


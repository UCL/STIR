//
// $Id$: $Date$
//
/*!
  \file
  \ingroup buildblock

  \brief Metz and Gaussian Filter Routines for iterative filtering 

  \author Matthew Jacobson (some help by Kris Thielemans)
  \author based on C-code by Roni Lefkowitz
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
/*
   
   First Version 2 August 98
Changes
   1. Added possibility for convolution with small kernels (variable kerlen). 22 August 1998
   2. Added Gaussian filtering.

  
*/


const float ZERO_TOL= 0.000001F; //MJ 12/05/98 Made consistent with other files
#define TPI 6.28318530717958647692
#define FORWARDFFT 1
#define INVERSEFFT -1

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#define REALC(a) 2*(a)
#define IMGC(a) 2*(a)+1

#include "ImageFilter.h"

#include <iostream> 
#include <numeric>

#include "VoxelsOnCartesianGrid.h"
#include "recon_array_functions.h"

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

//MJ 05/03/2000 centimeters-->millimeters.

START_NAMESPACE_TOMO

// KT 30/05/2000 made axdir from int to int

//MJ 22/04/99  KT wants the declarations here and not in bbfilters.h

// MJ 17/12/98 added the next three functions

void build_gauss(VectorWithOffset<float>&kernel, int res,float s2,  float sampling_interval);
/*

 build Gaussian kernel according to the full width half maximum 

*/

void discrete_fourier_transform(VectorWithOffset<float>&data, unsigned int nn, int isign);



VectorWithOffset<float> build_metz(int *kerlen, int res, float N,float fwhm, float MmPerVoxel);


void extract_row(VectorWithOffset<float>&onerow,int axdir,int row_num,VoxelsOnCartesianGrid<float>& input_image);
/* 

extract a row from the input_image
   
*/

                 
 
void convolution_1D(VectorWithOffset<float>&outrow,int res,int kerlen,VectorWithOffset<float>&onerow,VectorWithOffset<float>&cfilter);

void reput_row(VoxelsOnCartesianGrid<float> &input_image,int axdir,int row_num,VectorWithOffset<float>&outrow);

/* 

put back the filtered row into the input_image 

*/
 


void separable_cartesian_3Dfilter(int kerlx,int kerlz, VoxelsOnCartesianGrid<float> &input_image,VectorWithOffset<float>&kernelX,VectorWithOffset<float>&kernelZ,VectorWithOffset<float>&onerow, VectorWithOffset<float>&outrow);

/* Perform 3D filtering on a input_image with the possiblility of 
   different filters in the XY and Z directions
   input_image: res*res*res
   kernel: res/2
   onerow: res*2
   outrow: res 
*/


void discrete_fourier_transform(VectorWithOffset<float>&data, unsigned int nn, int isign)
{
	unsigned int n,mmax,m,j,istep,i;
	double wtemp,wr,wpr,wpi,wi,theta;
	float tempr,tempi;
	n=nn << 1;
	j=1;
	for (i=1;i<n;i+=2) {
		if (j > i) {
			SWAP(data[j],data[i]);
			SWAP(data[j+1],data[i+1]);
		}
		m=n >> 1;
		while (m >= 2 && j > m) {
			j -= m;
			m >>= 1;
		}
		j += m;
	}
	mmax=2;
	while (n > mmax) {
		istep=mmax << 1;
		theta=isign*(TPI/mmax);
		wtemp=sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi=sin(theta);
		wr=1.0;
		wi=0.0;
		for (m=1;m<mmax;m+=2) {
			for (i=m;i<=n;i+=istep) {
				j=i+mmax;
				tempr=wr*data[j]-wi*data[j+1];
				tempi=wr*data[j+1]+wi*data[j];
				data[j]=data[i]-tempr;
				data[j+1]=data[i+1]-tempi;
				data[i] += tempr;
				data[i+1] += tempi;
			}
			wr=(wtemp=wr)*wpr-wi*wpi+wr;
			wi=wi*wpr+wtemp*wpi+wi;
		}
		mmax=istep;
	}
}



void build_gauss(VectorWithOffset<float>&kernel, int res,float s2,  float sampling_interval)
{

 
 float sum;
 int cutoff=0;
 int j,hres;



 hres = res/2;
 kernel[hres-1] = 1/sqrt(s2*TPI);
 sum =   kernel[hres-1];       
 kernel[res-1] = 0;
  for (j=1;(j<hres && !cutoff);j++){
     kernel[hres-j-1] = kernel[hres-1]*(double ) exp(-0.5*(j*sampling_interval)*(j*sampling_interval)/s2);
   kernel[hres+j-1] = kernel [hres-j-1];
   sum +=  2.0 * kernel[hres-j-1];
   if (kernel[hres-j-1]  <kernel[hres-1]*ZERO_TOL) cutoff=1;

 

  }


 

/* Normalize the filter to 1 */
 for (j=0;j<res;j++) kernel[j] /= sum; 


 
           
}







//MJ 19/04/99  Used KT's solution to the shifted index problem. Also build_metz now allocates the kernel.

VectorWithOffset<float> build_metz(int *kerlen,int res ,float N,float fwhm, float MmPerVox)
 {    

  if(fwhm>0.0F){

 
      // KT 30/05/2000 dropped unsigned
     int i;
     float xreal,ximg,zabs2;                                        

//MJ 12/05/98 compute parameters relevant to DFT/IDFT

     float s2 = fwhm*fwhm/(8*log(2)); //variance in Mm

     const int n=7; //determines cut-off in both space and frequency domains
     const float sinc_length=10000.0;
     int samples_per_voxel=(int)(MmPerVox*2*sqrt(2*log(10)*n/s2)/TPI +1);
     const float sampling_interval=MmPerVox/samples_per_voxel;
     float stretch= (samples_per_voxel>1)?sinc_length:0.0;

     int Res=(int)(log((sqrt(8*n*log(10)*s2)+stretch)/sampling_interval)/log(2)+1);
     Res=(int) pow(2.0,(double) Res); //MJ 12/05/98 made adaptive 



    cerr<<endl<<"Variance: "<< s2<<endl; 
    cerr<<"Voxel dimension (in mm): "<< MmPerVox<<endl;  
    cerr<<"Samples per voxel: "<< samples_per_voxel<<endl;
    cerr<<"Sampling interval (in mm): "<< sampling_interval<<endl;
    cerr<<"FFT vector length: "<<Res<<endl;  
 
     
/* allocate memory to metz arrays */
     VectorWithOffset<float> filter(Res);

     //MJ 05/03/2000 padded 1 more element to fftdata and pre-increment
     //The former technique was illegal.
     VectorWithOffset<float> fftdata(0,2*Res);
     for (i=0;i<Res ;i++ ) filter[i]=0.0;
     for (i=0;i<2*Res ; i++ ) fftdata[i]=0.0;     
   
     
/* build gaussian */

     build_gauss(filter,Res,s2,sampling_interval);



/* Build the fft array, odd coefficients are the imaginary part */

     for (i=0;i<=Res-(Res/2);i++) {
                   fftdata[REALC(i)]=filter[Res/2-1+i];
                   fftdata[IMGC(i)] = 0.0;
                   }

     for (i=1;i<(Res/2);i++) {
                   fftdata[REALC(Res-(Res/2)+i)]=filter[i-1];
                   fftdata[IMGC(Res-(Res/2)+i)] = 0.0;
                   }


/* FFT to frequency space */
    fftdata.set_offset(1);
     discrete_fourier_transform(fftdata/*-1*/,Res,FORWARDFFT); 
    fftdata.set_offset(0);
   


/* Build Metz */                       
     N++;


     int cutoff=(int) (sampling_interval*Res/(2*MmPerVox));
     //cerr<<endl<<"The cutoff was at: "<<cutoff<<endl;


     for (i=0;i<Res;i++) {


             xreal = fftdata[REALC(i)];
             ximg  = fftdata[IMGC(i)]; 
             zabs2= xreal*xreal+ximg*ximg;
	     filter[i]=0.0; // use this loop to clear the array for later


	     //MJ 26/05/99 cut off the filter
	     if(stretch>0.0){
	       // cerr<<endl<<"truncating"<<endl;
	       if(i>cutoff && Res-i>cutoff) zabs2=0;

	     }

	     if (zabs2>1) zabs2=(float) (1-ZERO_TOL);
             if (zabs2>0) {
	       // if (zabs2>=1) cerr<<endl<<"zabs2 is "<<zabs2<<" and N is "<<N<<endl;
               fftdata[REALC(i)]=(1-pow((1-zabs2),N))*(xreal/zabs2);
               fftdata[IMGC(i)]=(1-pow((1-zabs2),N))*(-ximg/zabs2);
               }
               else {
               
                  fftdata[REALC(i)]= 0.0;
                  fftdata[IMGC(i)]= 0.0;
               }
                     
      }
/* return to the spatial space */               
 
     fftdata.set_offset(1);
     discrete_fourier_transform(fftdata/*-1*/,Res,INVERSEFFT); 
     fftdata.set_offset(0);
    

 



/* collect the results, normalize*/

      for (i=0;i<=Res/2;i++) {
	if (i%samples_per_voxel==0){
    int j=i/samples_per_voxel;
    filter[j] = (fftdata[REALC(i)]*MmPerVox)/(Res*sampling_interval);
	}

      }



//MJ 17/12/98 added step to undo zero padding (requested by RL)

 *kerlen=Res; 

 for (i=Res-1;i>=0;i--){
   if (fabs((double) filter[i])>=(0.0001)*filter[0]) break;
   else (*kerlen)--;

 }



 if ((*kerlen)>res/2){
   *kerlen=res/2;

 }


 VectorWithOffset<float> kernel(*kerlen);//=new float[(*kerlen)];

 for (i=0;i<(*kerlen);i++) kernel[i]=filter[i];
  
 return kernel;
  }

  else{
    VectorWithOffset<float> kernel(1);//=new float[1];
    //*kernel=1.0F;
    //*kerlen=1L;
    kernel[0] = 1.F;
    *kerlen=1L;

 return kernel;
  }



}


void extract_row(VectorWithOffset<float>&onerow,int axdir,int row_num,VoxelsOnCartesianGrid<float>& input_image){


  int zs,ys,xs, ze,ye,xe;
 

  zs=input_image.get_min_z();
  ys=input_image.get_min_y();
  xs=input_image.get_min_x(); 
  
  ze=input_image.get_max_z();  
  ye=input_image.get_max_y(); 
  xe=input_image.get_max_x();


  if(axdir==2){
    
    int  res=input_image.get_x_size();
    onerow.fill(0);    
    int hres=res/2;

    int z_0=row_num/res;
    int y_0=row_num-z_0*res;

    for (int x=xs; x<= xe; x++){
     
      onerow[hres+x-xs]= input_image[zs+z_0][ys+y_0][x];

    }

  }

  else if(axdir==1){
    
    int  res=input_image.get_y_size();
    onerow.fill(0);    
    int hres=res/2;

    int z_0=row_num/res;
    int x_0=row_num-z_0*res;

    for (int y=ys; y<= ye; y++){

      onerow[hres+y-ys]=input_image[zs+z_0][y][xs+x_0];

    }

  }

  else if(axdir==0){
    
    int  res=input_image.get_z_size();
    onerow.fill(0);    
    int hres=res/2;

    int y_0=row_num/input_image.get_y_size();
    int x_0=row_num-y_0*input_image.get_y_size();

    for (int z=zs; z<= ze; z++){

      onerow[hres+z-zs]=input_image[z][ys+y_0][xs+x_0];

    }

  }


  else  cerr<<endl<<"Filters: debug message : extract_row"<<endl;

 
}
         
                  
                  
                  
void convolution_1D(VectorWithOffset<float>&outrow,int res,int kerlen,VectorWithOffset<float>&onerow,VectorWithOffset<float>&cfilter){

  int i,j,hres;
  hres =res/2;                  
  if (kerlen>hres) kerlen = hres;
  /* convolve the filter (res/2) with the padded row (res*2) 
     and put in outrow (res)
  */
  for (i=0; i<res; i++) {                                            
    outrow[i] = cfilter[0]*onerow[hres+i];
    for (j=1; j<kerlen; j++) outrow[i] += cfilter[j]*(onerow[hres+i-j]+onerow[hres+i+j]);
    /*  if (outrow[i]<0) outrow[i]=ZERO_TOL;*/

  }
}  


void reput_row(VoxelsOnCartesianGrid<float> &input_image,int axdir,int row_num,VectorWithOffset<float>&outrow){


  int zs,ys,xs, ze,ye,xe;
 

  zs=input_image.get_min_z();
  ys=input_image.get_min_y();
  xs=input_image.get_min_x(); 
  
  ze=input_image.get_max_z();  
  ye=input_image.get_max_y(); 
  xe=input_image.get_max_x();


  if(axdir==2){
    
    int  res=input_image.get_x_size();
    //int hres=res/2; 
    
    //MJ 19/04/99 doesn't look like we ever used hres here

    int z_0=row_num/res;
    int y_0=row_num-z_0*res;

    for (int x=xs; x<= xe; x++){

      input_image[zs+z_0][ys+y_0][x]=outrow[x-xs];

    }

  }

  else if(axdir==1){
    
    int  res=input_image.get_y_size();
    //int hres=res/2;

    int z_0=row_num/res;
    int x_0=row_num-z_0*res;

    for (int y=ys; y<= ye; y++){

      input_image[zs+z_0][y][xs+x_0]=outrow[y-ys];

    }

  }

  else if(axdir==0){
    
    //int  res=input_image.get_z_size();
    //int hres=res/2;

    int y_0=row_num/input_image.get_y_size();
    int x_0=row_num-y_0*input_image.get_y_size();


    for (int z=zs; z<= ze; z++){

      input_image[z][ys+y_0][xs+x_0]=outrow[z-zs];

    }

  }


  else  cerr<<endl<<"Filters: debug message : reput_row"<<endl;
  
}
 
 

                 

void separable_cartesian_3Dfilter(int kerlx,int kerlz, VoxelsOnCartesianGrid<float> &input_image,VectorWithOffset<float>&kernelX,VectorWithOffset<float>&kernelZ,VectorWithOffset<float>&onerow, VectorWithOffset<float>&outrow){

  int axdir;
  int i,pres;

  int resx=input_image.get_x_size();
  int resz=input_image.get_z_size();

  /* filter in the X, Y directions */

  axdir = 2;
  pres = resx*resz;
  printf ("Filters:filtering in direction %d\n",axdir);
  for (i=0;i<pres;i++){
    extract_row(onerow,axdir,i,input_image);
    convolution_1D(outrow,resx,kerlx,onerow,kernelX);
    reput_row(input_image,axdir,i,outrow); 

 

  }
      
  axdir = 1; 
  printf ("Filters:filtering in direction %d\n",axdir);
  for (i=0;i<pres;i++){
    extract_row(onerow,axdir,i,input_image);
    convolution_1D(outrow,resx,kerlx,onerow,kernelX);
    reput_row(input_image,axdir,i,outrow);
  }
      
  axdir = 0;
  pres=resx*resx;
  printf ("Filters:filtering in direction %d\n",axdir);
  for (i=0;i<pres;i++){
    extract_row(onerow,axdir,i,input_image);
    convolution_1D(outrow,resz,kerlz,onerow,kernelZ);
    reput_row(input_image,axdir,i,outrow); 
 

  }      

  // TODO do this only conditionally
    truncate_min_to_small_positive_value(input_image);
    cerr<<endl; //MJ 06/03/2000 added 

  /* direction 0 : Z direction */
  /* direction 1 : Y direction */
  /* direction 2 : X direction */

}                 



//Constructor-like

//MJ 05/03/2000 got rid of scanner dependence
void ImageFilter::build(const DiscretisedDensity<3,float>& representative_density,double fwhmx_dir,double fwhmz_dir,float Nx_dir,float Nz_dir)
{

  if (kernels_built)
    {
    error("Filter already built\n");    
    }
  const VoxelsOnCartesianGrid<float>& representative_image =
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(representative_density);

  fwhmx=fwhmx_dir;
  fwhmz=fwhmz_dir;
  Nx=Nx_dir;
  Nz=Nz_dir;

  //Note, does not work with zooming
  //MJ 05/03/2000 Now it does!!
  int Resx=representative_image.get_x_size();
  int Resz=representative_image.get_z_size();

  onerow.grow(0,2*max(Resx,Resz)-1);
  outrow.grow(0,max(Resx,Resz)-1);

   kernelX=build_metz(&kerlx,Resx,Nx,fwhmx,representative_image.get_voxel_size().x());

   for (int i=0;i<kerlx;i++)
   if(Nx>0.0)  printf ("X-dir Metz[%d]=%f\n",i,kernelX[i]);   
   else printf ("X-dir Gauss[%d]=%f\n",i,kernelX[i]);

   cerr<<endl;

   kernelZ=build_metz(&kerlz,Resz,Nz,fwhmz,representative_image.get_voxel_size().z());  

   for (int i=0;i<kerlz;i++)
   if(Nz>0.0)  printf ("Z-dir Metz[%d]=%f\n",i,kernelZ[i]);   
   else printf ("Z-dir Gauss[%d]=%f\n",i,kernelZ[i]);
    
   cerr<<endl;

   kernels_built=true; //kernels successfully built



}

ImageFilter::ImageFilter(){

kernels_built=false;
kerlx=kerlz=0;

}


ImageFilter::~ImageFilter()
{}




void ImageFilter::apply(DiscretisedDensity<3,float>& input_image){

  //MJ 03/05/2000
  if(kerlx==0 || kerlz==0) {error("Filter operation attempted with an unconstructed ImageFilter object \n");}

  VoxelsOnCartesianGrid<float>& input_image_vox =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(input_image);
  separable_cartesian_3Dfilter(kerlx,kerlz,input_image_vox,kernelX,kernelZ,onerow,outrow);


}



END_NAMESPACE_TOMO

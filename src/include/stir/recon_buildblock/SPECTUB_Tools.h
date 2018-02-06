/*
    Copyright (c) 2013, Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. 
    Copyright (c) 2013, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details

  \author Carles Falcon
*/

#ifndef _WM_SPECTUB_H
#define _WM_SPECTUB_H

#include <iostream>
#include <vector>

#include <string>

namespace SPECTUB {

//::: srtuctures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

//! collimator parameters structure
typedef struct
{
 	int num;      // number of collimator (see weight_64.cpp for options) 
	bool do_fb;   // true: fanbeam collimator || false: parallel collimator   
	
	//... parallel collimator parameters .....................

	float A;      // linear factor for dependency of sigma on distance: sigma=A*dist+B (parallel, fanbeam-vertical)
    float B;	  // independent factor for dependency of sigma on distance: sigma=A*dist+B (parallel, fanbeam-vertical)
		
	//... fanbeam collimator parameters ......................
	
	float F;      // Focal length (fanbeam)
	float L;      // collimator to detector distance (?) (fanbeam)	
	float A_h;    // linear factor for dependency of sigma on distance (fanbeam horizontal)
	float A_v;    // linear factor for dependency of sigma on distance (fanbeam horizontal)
	float D;      // collimator thicknes?? (fanbeam)
	float w;      // collimator thickness (?) (fanbeam)
	float insgm;  // intrinsic sigma (cristal resolution) (fanbeam)
	
} collim_type;

//! structure for bin information
typedef struct
{
	int Nrow;     // number of rows
	int Ncol;     // number of columns
	int Nsli;     // number of slices
	int Npix;     // number of pixels (axial planes)
	int Nvox;	  // number of voxels (the whole volume)
		
	int first_sl; // first slice to reconstruct (0->Nslic-1)
	int last_sl;  // last slice to reconstruct + 1 (end of the 'for' loop) (1->Nslic) 
	
	float Nrowd2; // half of Nrow
	float Ncold2; // half of Ncol
	float Nslid2; // half of Nsli
	
	float Xcmd2;  // Half of the size of the volume, dimension x (cm);
	float Ycmd2;  // Half of the size of the volume, dimension y (cm);
	float Zcmd2;  // Half of the size of the volume, dimension z (cm);
	
	float szcm;   // voxel size (side length in cm)
	float thcm;   // voxel thickness (cm)
	
	float x0;     // x coordinate (cm, ref center of volume) of the first voxel
	float y0;	  // y coordinate (cm, ref center of volume) of the first voxel
	float z0;     // z coordinate (cm, ref center of volume) of the first voxel
	
	float *val;   // array of values
	
} volume_type;


//! structure for projection information
typedef struct
{
	int   Nbin;     // length of the detection line in bins (number of bins per line)
	float lngcm;	// length of the detection line in cm.	
	float szcm;     // bin size in cm
	
	int   Nsli;		// number of slices
	float thcm;     // slice thickness in cm
	
	int   Nang;     // number of projection angles
	float ang0;     // initial projection angle. degrees from upper detection plane (parallel to table). Negative for CW rotacions (see manual) 
	float incr;     // angle increment between two consecutive projection angles. Degrees. Negative for CW, Positive for CCW
	
	int NOS;		// number of subsets in which to split the matrix
	int NangOS;     // Number of angles in each subset = Nang/NOS
	int Nbp;        // number of bins in a 2D projection=lng*Nsli
	int Nbt;		// total number of bins= Nbp*Nang
	int NbOS;       // total number of bins per subset= Nbp*NangOS = Nbt/NOS
	int *order;		// order of the angles of projection (array formed by indexs of angles belonging to consecutive subsets)
	
	float Nbind2;   // length of the detection line (in bins) divided by 2
	float lngcmd2;  // length of the detection line in cm divided by 2	
	float Nslid2;   // number of slices divided by 2
	
} proj_type;

//! complementary information (matrix header)
typedef struct  
{
	int subset_ind;   // subset index of this matrix (-1: all subsets in a single file)
	int *index;       // included angles into this subset (index. Multiply by increm to get corresponding angles in degrees)

	float *Rrad;      // Rotation radius (one value for each projection angle)
	float min_w;      // minimum weight to be taken into account
	float psfres;     // spatial resolution of continous distributions in PSF calculation 
	float maxsigm;    // maximum number of sigmas in PSF calculation
	
	bool fixed_Rrad;  // true: fixed radius of projection || false: variable radius of projection 
	bool do_psf;      // true: to correct for PSF         || false: do not correct for PSF
	bool do_psf_3d;   // true: 3d correction for PSF      || false: 2d correction for PSF
	bool predef_col;  // true: predefined collimator      || false: user defined PSF parametres  
	bool do_att;      // true: to correct for attenuation || false: do not correct for attenuation
	bool do_full_att; // true: diff att for each PSF bin  || false: the whole PSF has the same att. factor (central line)
	bool do_msk;      // true: weights just inside msk    || false: weights for the whole FOV
	bool do_msk_slc;  // true: weights for several slices || false: weights for all slices
	bool do_msk_cyl;  // true: to use cylinder as a mask  || false: not to use cylinder as a mask
	bool do_msk_att;  // true: to use att map as a mask   || false: not to use att map as a mask
    bool do_msk_file; // true: explicit mask              || false: not to use explicit mask 
	
	std::string att_fn;    // attenuation map filename	
	std::string msk_fn;    // explicit mask filename	
	std::string col_fn;    // collimator parameters filename
	std::string Rrad_fn;   // rotation radius file name
	
	volume_type vol;  //
	proj_type prj;    //
	collim_type COL;  // collimator structure (see weight3d_64b.cpp for options)
	
} wmh_type;

//! weight_mat structure definition. Structure for reading weight matrix
typedef struct  
{
    //weight matrix dimensions
	
	int ne;     //nonzero elements 
	int NbOS;   //dimension 1 (rows) of the weight matrix (NbOS or NBt)
	int Nvox;   //dimension 2 (columns) of the weight matrix (Nvox)

	//weight matrix values
	
	float *ar;  //array of nonzero elements of weight matrix (by rows)
	int *ja;    //array of the column index of the above elements
	int *ia;    //array containing the indexes of the previous vector where a row change happens
	
	bool do_save_wmh; //to save or not to save weight_mat header info into weight_mat file

} wm_type;

//! weight_mat_da structure definition. Structure for generating weight matrix
typedef struct   
{
	int NbOS;          // dimension 1 (rows) of the weight matrix (NbOS or NBt)
	int Nvox;          // dimension 2 (columns) of the weight matrix (Nvox)
	float **val;       // double array to store weights (index of the projection element, number of weight for that element)
	int **col;         // double array to store column indexs of the above element  (index of the projection element, number of weight for that element)
	int *ne;           // array indicating how many elements has been stored for each element of projection
	
	//... filename .............................................
	
	std::string fn;      // matrix base name (filename without extension index)
	std::string OSfn;    // matrix filename
	std::string fn_hdr;  // matrix header file name
	
	//... indexs for STIR format ...............................
	
	int *na, *nb, *ns;        //indexs for projection elements (angle, bin, slice respec.)
	short int *nx, *ny, *nz;  //indexs for image elements (x,y,z) 
	
	//... format ...............................................
	
	bool do_save_wmh;  // to save or not to save weight_mat header info into weight_mat file
	bool do_save_STIR; // to save weight matrix with STIR format
		
} wm_da_type;

//! structure for distribution function information
typedef struct
{
	int lng;		// length (in discretization intervals) (odd number)
	int lngd2;      // half of the length (in discretization intervals) (lng-1)/2
	float res;      // spatial resolution of distfunc (discretization interval)
	float *val;     // array of values
	float *acu;     // distribution function values (cumulative sum)

} discrf_type;

//! structure for PSF information
typedef struct
{
	int maxszb;		// maximum size in bins (for allocation purposes)
    
	int di;         // discretization interval (to reduce spatial resolution to bin resolution). (int: #points)
	int *ind;        // projection indexs for the bins of the PSF (horizontal)
 	int Nib;        // actual number of bins forming the PSF (length of PSF in bins)
 	
	float sgmcm;    // sigma of the PSF in cm
	float lngcm;    // length of PSF (in cm)
	float lngcmd2;  // half of the length of PSF (in cm)
	float *val;     // array of values
	float efres;    // effective resolution (psfres rescaled to real psf length)
	
} psf1d_type;

//! structure for distribution function information
typedef struct
{
	int maxszb_h;   // maximum size in bins horizontal (for allocation purposes)
    int maxszb_v;   // maximum size in bins vertical (for allocation purposes)
    int maxszb_t;   // maximum size in bins total (for allocation purposes)
	
    int *ib;        // projection indexs for the bins of the PSF (horizontal)
	int *jb;        // projection indexs for the bins of the PSF (vertical)
 	int Nib;        // actual number of bins forming the PSF (length of PSF in bins)
	
	float *val;     // array of values
	
} psf2da_type;

//! structure to store angles values, indices and ratios
typedef struct   
{
	int ind;           // index of angle considering the whole set of projections (sequential order: 0->Nang-1)
	int indOS;         // index of angle considering the subjet
	int iOS_proj;      // index of the first bin for this angle (in subset of projections)

	float cos;         // coninus of the angle
	float sin;         // sinus of the angle
	 
	// parametres for describng the trapezoidal projection of a square voxel
	
	float p;           // plateau higness
	float m;           // slope of the trapezoid
	float n;           // independent term of the slope
	int   N1;          // index of the first vertice (end of plateau) in DX units
	int   N2;          // index of the second vertice (end of the slope) in DX units
	discrf_type vxprj; // projection of a square voxel in this direction (for no PSF)	
	
	// variable rotation radius
	
	float Rrad ;       // rotation radius for this angle
	
	// first bin position and increments
	
	float xbin0;       // x coordinate for the first bin of the detection line corresponding to this angle 
	float ybin0;       // y coordinate for the first bin of the detection line corresponding to this angle 
	float incx;        // increment in x to the following bin in detection line
	float incy;        // increment in y to the following bin in detection line
	
} angle_type;


//! structure for voxel information
typedef struct
{
	float szcm;   // voxel size (side length in cm)
	float thcm;   // voxel thickness (cm)
	
	int irow;     // row index
	int icol;     // column index
	int islc;     // slice index
	int ip;       // in plane index (considering the slice as an array) of the voxel
	int iv;       // volume index (considering the volume as an array) of the voxel
	
	float x;      // x coordinate (cm, ref center of volume)
	float y;      // y coordinate (cm, ref center of volume)
	float z;      // z coordinate (cm, ref center of volume)
	float x1;     // x coordinade in rotated framework
	
	float dv2dp;  // distance from voxel to detection plane
	float costhe; // cosinus of theta (angle between focal-voxel line and line perpendicular to detection plane) (fanbeam)
	float xdc;    // distance (cm over detection line) from projected voxel to the center of the detection line
	float xd0;    // distance (cm over detection line) from projected voxel to the begin of the detection line 
	float zd0;    // distance (cm) to the lowest plane of the volume
	
} voxel_type;

//! structure for bin information
typedef struct
{	
	float szcm;   // bin size (cm)
	float szcmd2; // half of the above value
	float thcm;   // bin thickness (cm)
	float thcmd2; // bin thickness (cm)

	float x;      // x coordinate (cm, ref center of volume)
	float y;	  // y coordinate (cm, ref center of volume)
	float z;      // z coordinate (cm, ref center of volume)
	
	float szdx;   // bin size in resolution units
	float thdx;   // bin thickness in resolution units 
	
} bin_type;

//! structure for attenuation calculus
typedef struct
{
	float *dl;	// distance of attenuation path on each crossed voxel of the attenuation map 
	int *iv;    // in-plane index (considering slices of attmap as an array) of any crossed voxel of the attenuation map
	int lng;    // number of elements in the attenuation path
	int maxlng; // maximum number of elements in the attenuation path (for allocation)
	
} attpth_type;


//::: functions :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


//... functions from wmtools_SPECT.cpp .........................................


void write_wm_FC ();                               // to write double array weight matrix 

void write_wm_hdr ();                              // to write header of a matrix

void write_wm_STIR ();                             // to write matrix in STIR format


void index_calc ( int *indexs );                   // to calculate projection index order in subsets 

void read_Rrad ();								   // to read variable rotation radius from a text file (1 radius per line)

//
//void col_params ( collim_type *COL );              // to fill collimator structure
//
//void read_col_params ( collim_type *COL);          // to read collimator parameters from a file


void fill_ang ( angle_type *ang );				   // to fill angle structure

void generate_msk ( bool *msk_3d, bool *msk_2d, float *att, volume_type *vol); // to create a boolean mask for wm (no weights outside the msk)

void read_msk_file ( bool * msk );                 // to read mask from a file


void read_att_map ( float *attmap );               // to read attenuation map from a file


int  max_psf_szb ( angle_type *ang );

float calc_sigma_h ( voxel_type vox, collim_type COL);

float calc_sigma_v ( voxel_type vox, collim_type COL);


char *itoa ( int n, char *s);                      // to conver integer to ascii

void free_wm ( wm_type *f );                       // to free weight_mat

void free_wm_da ( wm_da_type *f );                 // to free weight_mat_da


void error_wmtools_SPECT(int nerr, std::string txt);    // error messages in wm_SPECT


//... functions from wm_SPECT.2.0............................

//int wm_SPECT( std::string inputFile);

// void error_wm_SPECT( int nerr, std::string txt);      //list of error messages

////void wm_inputs( std::string fileName, proj_type * prj, volume_type *vol, voxel_type *vox, bin_type *bin );
//void wm_inputs(char **argv, 
//			   int argc, 
//			   proj_type *prj,
//			   volume_type *vol,
//			   voxel_type *vox,
//	       bin_type *bin);
//
//
////void read_inputs( std::string param, proj_type * prj, volume_type *vol, voxel_type *vox, bin_type *bin );
//void read_inputs(vector<std::string> param, 
//				 proj_type *prj,
//				 volume_type *vol,
//				 voxel_type *vox,
////		 bin_type *bin);
//
//extern wmh_type wmh;           // weight matrix header. Global variable
//
//extern wm_da_type wm;          // double array weight matrix structure. Global variable
//
//extern float * Rrad;           // variable projection radius (in acquisition order)

} // namespace SPECTUB

#endif //_WM_SPECT_H

/*
 * Copyright (c) 2014,
 * Institute of Nuclear Medicine, University College of London Hospital, UCL, London, UK.
 * Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. All rights reserved.
 * This software is distributed WITHOUT ANY WARRANTY; 
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 
 \author Carles Falcon
 */

#ifndef _WM_SPECT_mph_H
#define _WM_SPECT_mph_H

#include <iostream>
#include <vector>

namespace SPECTUB_mph
{

#define NUMARG 23

#define EPSILON 1e-12
#define EOS '\0'

#define maxim(a,b) ((a)>=(b)?(a):(b))
#define minim(a,b) ((a)<=(b)?(a):(b))
#define abs(a) ((a)>=0?(a):(-a))
#define SIGN(a) (a<-EPSILON?-1:(a>EPSILON?1:0))
 
#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#define dg2rd 0.01745329251994

#define DELIMITER1 '#' //delimiter character in input parameter text file
#define DELIMITER2 '%' //delimiter character in input parameter text file

  //::: srtuctures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  //.....structure for distribution function information

  typedef struct
  {
    int max_dimx;	// maximum size in bins (for allocation purpose)
    int max_dimz;	// maximum size in bins (for allocation purpose)
    
    int dimx;       // actual number of bins forming the PSF (dimx)
    int dimz;       // actual number of bins forming the PSF (dimz)

    int ib0;        // first i index for the bins of the PSF
    int jb0;        // first j index for the bins of the PSF
    
    float lngxcmd2; // half of the lenght in cm, dimx
    float lngzcmd2; // half of the lenght in cm, dimz
    
    float xc;
    float zc;
 
    float sum;      // sum of values (for intensity normalization)
    
    float ** val;   // array of values
    //float res1;     // resolution dimx
    //float res2;     // resolution dimz
	
  } psf2d_type;

  //... collimator's holes parametre structure................................

  typedef struct
  {
    //int idet;        // index of the detel the hole projects to
    
    //... coordinates ......................
    
    float acy;       // angular coordinate of the hole in volume framework: angle (deg) (cyl)
    float acyR;      // angular coordinate of the hole in rotated framework: ang (deg) (cyl)
    
    float x1;        // x coordinate of the center of the hole (cm): rotated framework
    float y1;        // y coordinate of the center of the hole (cm): rotated framework
    float z1;        // z coordinate of the center of the hole (cm): rotated framework
    
    //... axial, edge and acceptance angles ...............................
    
    float ahx;       // x angle of the hole axis (deg)
    float ahz;       // z angle of the hole axis (deg)
    
    float aa_x;      // acceptance angle x (aperture with respect to hole axis: half of the total acceptance angle)
    float aa_z;      // acceptance angle z (aperture with respect to hole axis: half of the total acceptance angle)
    
    float egx;     // absolute angle of the hole edge (x, minor) in rotated framework
    float Egx;     // absolute angle of the hole edge (x, major) in rotated framework
    float egz;     // absolute angle of the hole edge (z, minor) in rotated framework
    float Egz;     // absolute angle of the hole edge (z, major) in rotated framework
    
    float ax_M;      // maximum angle of acceptance x
    float ax_m;      // minimum angle of acceptance x
    float az_M;      // maximum angle of acceptance z
    float az_m;      // minimum angle of acceptance z
    
    
    //... others ....................
    
    std::string shape;  // hole shape { rect, round }
    bool do_round; // true: round shape || false: rectangular shape
    float dxcm;    // horizontal size of the hole (cm): horizontal axis, diameter
    float dzcm;    // vertical size of the hole (cm): vertical axis, diameter
    
  } hole_type;

  //... structure for collimator information

  typedef struct
  {
    std::string model;      // cylindrical (cyl) or polygonal prism (pol)
    
    float rad;         // radius of cylinder containig holes (cyl) or apothem (pol)
    float L;           // collimator thickness
    float Ld2;         // half of the collimator thickness
    
    int   Nht;         // total number of holes
    std::vector <hole_type> holes;  //array hole_type structure
    
  } mphcoll_type;


  //.....structure for ring elements information

  typedef struct
  {
    int nh ;        // number of holes that project to this detel
    std::vector<int> who;  // array of indices of holes projecting to this detel
    
    float x0 ;       // cartesian coordinates of the center (unrotated system of reference): x
    float y0 ;       // cartesian coordinates of the center (unrotated system of reference): y
    float z0 ;       // cartesian coordinates of the center (unrotated system of reference): z
	
    float xbin0 ;    // x coordinate for the first bin of the detection row corresponding to this angle
    float ybin0 ;    // y coordinate for the first bin of the detection row corresponding to this angle
    float zbin0 ;    // z coordinate for the first bin of the detection row corresponding to this angle
    
    float incx ;     // increment in x to the following bin in detector row
    float incy ;     // increment in y to the following bin in detector row
    float incz ;     // increment in z to the following detector row

    float theta ;    // theta: in-plane angle radius vs x-axis. longitude (deg)
    float costh ;    // cosinus of theta
    float sinth ;    // sinus of theta
	
  } detel_type;

  //.....structure for distribution function information

  typedef struct
  {
    int dim;		// length (in discretization intervals) (odd number)
    int i_max;      // last i-index = dim -1
    float res;      // spatial resolution of distfunc (discretization interval)
    float *val;     // double array of values

  } discrf1d_type;

  //.....structure for distribution function information

  typedef struct
  {
    int dim;		// length (in discretization intervals) (odd number)
    int i_max;      // last i-index = dim -1
    int j_max;      // last j-index = dim -1
    float res;      // spatial resolution of distfunc (discretization interval)
    float **val;    // double array of values
    
  } discrf2d_type;

  //.....structure for projection information

  typedef struct
  {
    int Nbin;       // number of bins per row
    int Nsli;	    // number of slices
    
    int Ndt;        // number of detels (detector elements)
    int Nbd;        // number of bins per detel
    int Nbt;		// total number of bins
	   
    float szcm;     // bin size in cm
    float szcmd2;   // bin size in cm divided by 2
    float thcm;     // slice thickness (cm)
    float thcmd2;   // slice thickness in cm divided by 2
   
    float crth;     // crystal thickness (cm) to correction for depth
    float crth_2;   // power 2 of the above value
    float crattcoef; // attenuation coefficient of crystal
    float max_dcr;  //maximum distance of a ray inside the crystal
	
    float FOVxcmd2; // FOVcmx divided by 2
    float FOVzcmd2; // FOVcmz divided by 2  
    
    float rad;      // ring radius
    float radc;     // extended ring radius = radius + crystal thickness

    int NOS;		// number of subsets
    int NdOS;       // Number of detels per subset = Ndt/NOS
    int NbOS;       // total number of bins per subset = Nbt/NOS
    
    int *order;		// order of the angles of projection (array formed by indexs of angles belonging to consecutive subsets)
    float sgm_i;    // sigma of intrinsic PSF (cm)

    float *val;
	
  } prj_mph_type;

  //... structure for bin information..................................

  typedef struct
  {

    int Dimx;     // number of columns
    int Dimy;     // number of rows
    int Dimz;     // number of slices

    int Npix;     // number of pixels (voxels) per axial slice
    int Nvox;	  // number of voxels (the whole volume)
    
    int first_sl; // first slice to reconstruct (0->Nslic-1)
    int last_sl;  // last slice to reconstruct + 1 (end of the 'for' loop) (1->Nslic)
	
    float FOVxcmd2;  // half of the size of the volume, dimension x (cm);
    float FOVcmyd2;  // half of the size of the volume, dimension y (cm);
    float FOVzcmd2;  // half of the size of the volume, dimension z (cm);
	
    float szcm;   // voxel size (side length in cm)
    float thcm;   // voxel thickness (cm)
	
    float x0;     // x coordinate (cm, ref center of volume) of the first voxel
    float y0;	  // y coordinate (cm, ref center of volume) of the first voxel
    float z0;     // z coordinate (cm, ref center of volume) of the first voxel
	
    float *val;   // array of values
	
  } volume_type;

  //... matrix header information ............................

  typedef struct  
  {
    int subsamp;    // bin subsampling factor for accurate PSF and convolution calculations (typically 2 to 5)
    float mn_w;       // minimum weight to be taken into account
    float highres;    // high spatial resolution of continous distributions in PSF calculation

    float Nsigm;      // number of sigmas in PSF calculation
    float mndvh2;     // squared minimum distance voxel-hole (cm). Reference for efficiency
    float ro;         // radius of the object
    
    float max_hsxcm;   // maximum hole size dimx (for allocation purposes)
    float max_hszcm;   // maximum hole size dimz (for allocation purposes)
    float max_amp;     // maximum amplification
    float tmax_aix;    // tangent of the maximum incidence angle x (for allocation purposes)
    float tmax_aiz;    // tangent of the maximum incidence angle z (for allocation purposes)
	
    bool do_psfi;     // true: correct for intrinsic PSF
    bool do_depth;    // true: correct for impact depth
    bool do_att;      // true: correct for attenuation
    bool do_full_att; // true: coef att for each bin of PSF || false: same att factor for all bins of PSF (central line)
    bool do_msk_att;  // true: to use att map as a mask
    bool do_msk_file; // true: explicit mask
    
    // internal booleans variables
    
    bool do_subsamp;
    bool do_round_cumsum;
    bool do_square_cumsum;
	
    std::string att_fn;       // attenuation map filename
    std::string msk_fn;       // explicit mask filename
    std::string detector_fn;  // ring parameter filename
    std::string collim_fn;    // collimator parameter filename
	
    volume_type vol;           // structure with information of volume
    std::vector<detel_type> detel;  // structure with detection elements information
    prj_mph_type prj;          // structure with detection rings information
    mphcoll_type collim;       // structure with the collimator information
    
  } wmh_mph_type;

  //.......weight_mat structure definition. Structure for reading weight matrix

  typedef struct  
  {
    //... weight matrix dimensions .....................
	
    int ne;    //nonzero elements
    int Nbt;   //dimension 2 (rows) of the weight matrix (NbtOS or NBt)
    int Nvox;  //dimension 1 (columns) of the weight matrix (Nvox)

    //... weight matrix values .........................
	
    float *ar;  //array of nonzero elements of weight matrix (by rows)
    int *ja;    //array of the column index of the above elements
    int *ia;    //array containing the indexes of the previous vector where a row change happens

  } wm_type;

  //.......weight_mat_da structure definition. Structure for generating weight matrix

  typedef struct   
  {
    int Nbt;          // dimension 2 (rows) of the weight matrix (NbOS or NBt)
    int Nvox;          // dimension 1 (columns) of the weight matrix (Nvox)
    float **val;       // double array to store weights (index of the projection element, number of weight for that element)
    int **col;         // double array to store column indexs of the above element  (index of the projection element, number of weight for that element)
    int *ne;           // array indicating how many elements has been stored for each element of projection
	
    //... filename .............................................
	
    std::string fn;      // matrix name
    std::string fn_hdr;  // matrix header file name
	
    //... format ...............................................
	
    bool do_save_STIR; // to save weight matrix with STIR format
	
    //... indexs for STIR format ...............................
	
    int *na, *nb, *ns;        //indexs for projection elements (angle, bin, slice respec.)
    short int *nx, *ny, *nz;  //indexs for image elements (x,y,z) 
		
  } wm_da_type;

  //.....structure for distribution function information

  typedef struct
  {
    discrf2d_type square;  // distribution for square shape hole
    discrf2d_type round;   // distribution for round shape hole
    discrf1d_type cr_att;  // exponential to correct for crystal attenuation when do_depth
    
  } pcf_type;

  //.....structure for voxel information

  typedef struct
  {
    int ix;       // column index
    int iy;       // row index
    int iz;       // slice index
    int ip;       // inplane index (slice as an array)
    int iv;       // volume index (considering the volume as an array) of the voxel
	
    float x;      // x coordinate (cm, ref center of volume)
    float y;      // y coordinate (cm, ref center of volume)
    float z;      // z coordinate (cm, ref center of volume)
    
    float x1;     // x coordinate in rotated framework (cm)
    float y1;     // y coordinate in rotated framework (cm)
	
  } voxel_type;

  //.....structure for bin information

  typedef struct
  {
    float x;      // x coordinate (cm, ref center of volume)
    float y;	  // y coordinate (cm, ref center of volume)
    float z;      // z coordinate (cm, ref center of volume)
	
  } bin_type;

  //...... structure for LOR information

  typedef struct
  {
    //... all the following distances in cm .....................

    float x1d_l;   // x coordiante of intersection lor-detection plane in rotated reference system
    float z1d_l;   // z coordiante of intersection lor-detection plane in rotated reference system
    float x1dc_l;  // x coordiante of intersection lor-detection plane + crystal in rotated reference system
    float z1dc_l;  // z coordiante of intersection lor-detection plane + crystal in rotated reference system
    
    float hsxcm_d;   // size (cm) of the shadow of the hole at detector plane (x-azis)
    float hszcm_d;   // size (cm) of the shadow of the hole at detector plane (z-axis)
    float hsxcm_d_d2;  // half of the size (cm) of the shadow of the hole at detector plane (x-azis)
    float hszcm_d_d2;  // half of the size (cm) of the shadow of the hole at detector plane (z-axis)
    
    float hsxcm_dc;  // size (cm) of the shadow of the hole at detector + crystal plane (x-axis)
    float hszcm_dc;  // size (cm) of the shadow of the hole at detector + crystal plane (z-axis)
    float hsxcm_dc_d2;  // half of the size (cm) of the shadow of the hole at detector plane (x-azis)
    float hszcm_dc_d2;  // half of the size (cm) of the shadow of the hole at detector plane (z-axis)
    
    //... others .......................................................
    
    float eff;    // effectiveness
    
  } lor_type;        //voxel-hole link


  //::: functions :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  //... functions from wmtools_SPECT.cpp .........................................

  void wm_alloc ( int * Nitems );                        // allocate wm

  void free_wm () ;                                      // delete wm


  void write_wm_FC_mph ();                               // write double array weight matrix

  void write_wm_hdr_mph ();                              // write header of a matrix

  void write_wm_STIR_mph ();                             // write matrix in STIR format


  void read_prj_params_mph ();                          // read ring parameters from a file

  void read_coll_params_mph ();                          // read collimator parameters from a file

  void which_hole();


  void fill_pcf ();                                      // fill precalculated functions

  void free_pcf ();                                      // fill precalculated functions



  void calc_cumsum ( discrf2d_type *f );

  void generate_msk_mph ( bool *msk_3d, float *att );    // create a boolean mask for wm (no weights outside the msk)

  void read_msk_file_mph ( bool * msk );                 // read mask from a file


  std::string wm_SPECT_read_value_1d ( std::ifstream * stream1, char DELIMITER );

  void wm_SPECT_read_hvalues_mph ( std::ifstream * stream1, char DELIMITER, int * nh, bool do_cyl );

  void read_att_map_mph ( float *attmap );               // read attenuation map from a file



  char *itoa ( int n, char *s);                      // to conver integer to ascii


  void error_wmtools_SPECT_mph( int nerr, int ip, std::string txt );    // error messages in wm_SPECT

}

#endif //_WM_SPECT_H

/*
    Copyright (C) 2022, Matthew Strugari
    Copyright (C) 2014, Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. All rights reserved.
    Copyright (C) 2014, 2021, University College London
    This file is part of STIR.

    This software is distributed WITHOUT ANY WARRANTY;
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    See STIR/LICENSE.txt for details

    \author Carles Falcon
    \author Matthew Strugari
*/

//system libraries
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <math.h>
#include <ctype.h>
#include <algorithm>
#include <vector>
#include <time.h>
#include "stir/info.h"
#include "stir/error.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

//user defined libraries
#include "stir/recon_buildblock/PinholeSPECTUB_Tools.h"
#include "stir/recon_buildblock/PinholeSPECTUB_Weight3d.h"

using namespace std;
using std::string;

namespace SPECTUB_mph
{

#define NUMARG 23

#define EPSILON 1e-12
#define EOS '\0'

#define maxim(a,b) ((a)>=(b)?(a):(b))
#define minim(a,b) ((a)<=(b)?(a):(b))
#define abs(a) ((a)>=0?(a):(-a))
#define SIGN(a) (a<-EPSILON?-1:(a>EPSILON?1:0))
 
//#ifndef M_PI
//#define M_PI 3.141592653589793
//#endif

//#define dg2rd 0.01745329251994
float dg2rd = boost::math::constants::pi<float>() / (float)180. ;

#define DELIMITER1 '#' //delimiter character in input parameter text file
#define DELIMITER2 '%' //delimiter character in input parameter text file

//... global variables ..............................................

extern wmh_mph_type wmh;
extern wm_da_type wm;
extern pcf_type pcf;

//=============================================================================
//=== wm_alloc =============================================================
//=============================================================================

void wm_alloc( int * Nitems)
{

    //... double array wm.val and wm.col .....................................................
	
	//if ( ( wm.val = new (nothrow) float * [ wmh.prj.NbOS ] ) == NULL ) error_wmtools_SPECT_mph( 200, wmh.prj.NbOS, "wm.val[]" );
	//if ( ( wm.col = new (nothrow) int   * [ wmh.prj.NbOS ] ) == NULL ) error_wmtools_SPECT_mph( 200, wmh.prj.NbOS, "wm.col[]" );
	
	//... array wm.ne .........................................................................
	
	//if ( ( wm.ne = new (nothrow) int [ wmh.prj.NbOS + 1 ] ) == 0 ) error_wmtools_SPECT_mph(200, wmh.prj.NbOS + 1, "wm.ne[]");

    
    //... memory allocation for wm double arrays ...................................
    
    for( int i = 0 ; i < wmh.prj.NbOS ; i++ ){
        
        if ( ( wm.val[ i ] = new (nothrow) float [ Nitems[ i ] ]) == NULL ) error_wmtools_SPECT_mph( 200, Nitems[ i ], "wm.val[][]" );
        if ( ( wm.col[ i ] = new (nothrow) int   [ Nitems[ i ] ]) == NULL ) error_wmtools_SPECT_mph( 200, Nitems[ i ], "wm.col[][]" );
    }
    
    //... to initialize wm to zero ......................
    
    for ( int i = 0 ; i < wmh.prj.NbOS ; i++ ){
        
        wm.ne[ i ] = 0;
        
        for( int j = 0 ; j < Nitems[ i ] ; j++ ){
            
            wm.val[ i ][ j ] = (float)0.;
            wm.col[ i ][ j ] = 0;
        }
    }
    wm.ne[ wmh.prj.NbOS ] = 0;
}

//=============================================================================
//=== write_wm_FC =============================================================
//=============================================================================
//*** weight matrix no longer written to file
//=============================================================================
#if 0
void write_wm_FC_mph()
{
	FILE *fid;
	
	int ia_acum = 0;
	
	if ( (fid = fopen( wm.fn.c_str(), "wb" ) ) == NULL ) error_wmtools_SPECT_mph( 31, 0, wm.fn );
	
	fwrite ( &(wm.Nbt), sizeof(int), 1, fid);  // to write number of rows of wm (NbOS)
	fwrite ( &(wm.Nvox), sizeof(int), 1, fid);  // to write number of columns of wm (Nvox)
	
	//... number of non-zero elements in the weight matrix .......
	
	int ne = 0;
	for ( int j=0 ; j < wm.Nbt ; j++ ){
		ne += wm.ne[j];
    }
	fwrite ( &ne, sizeof(int), 1, fid);         // to write number of non-zeros element in the weight matrix
	
	//... to write the array of weights (along rows) ..............
	
	for ( int i = 0 ; i < wm.Nbt ; i++ ){
		for (int j = 0 ; j < wm.ne[i] ; j++ ){
			fwrite ( &wm.val[ i ][ j ], sizeof(float), 1, fid);
		}
	}
	
	//... to write the column index of each weight (volume index of the voxel the weight is associated to) ....
	
    for ( int i = 0 ; i < wm.Nbt ; i++ ){
		for ( int j = 0 ; j < wm.ne[ i ] ; j++ ){
			fwrite ( &wm.col[ i ][ j ] ,sizeof(int) ,1 , fid);
		}
	}
	//... to write the indexs of the array of weights where a change of row happens .........
	
	for ( int i = 0 ; i < wm.Nbt ; i++ ){
		fwrite ( &ia_acum, sizeof(int), 1, fid);
		ia_acum += wm.ne[i];
	}

	//... to write the total number of saved weights ..........................
	
	fwrite ( &ia_acum, sizeof(int), 1, fid);
	
	cout << "number of non-zero elements: " << ia_acum << endl;

	fclose (fid);
}

//=============================================================================
//=== write_wm_hdr ============================================================
//=============================================================================

void write_wm_hdr_mph()
{
	ofstream stream1( wm.fn_hdr.c_str() );
	if( !stream1 ) error_wmtools_SPECT_mph( 31, 0, wm.fn_hdr );
	
	//....... image and projections characteristics.........

	stream1 << "Header for the matrix: " << wm.fn << endl;
	stream1 << "number of columns: " << wmh.vol.Dimx << endl;
	stream1 << "number of rows: " << wmh.vol.Dimy << endl;
	stream1 << "number of slices: " << wmh.vol.Dimz << endl;
	stream1 << "voxel size (cm): " << wmh.vol.szcm << endl;
	stream1 << "slice thickness (cm): " << wmh.vol.thcm << endl;
    
    stream1 << "radius of the object (cm): " << wmh.ro << endl;
    
    stream1 << "first slice to reconstruct : " << wmh.vol.first_sl + 1 << endl;
    stream1 << "last slice to reconstruct : " << wmh.vol.last_sl << endl;
 
    stream1 << "minimum weight (geometrical contribution): " << wmh.mn_w << endl;
	stream1 << "high resolution (discretization interval for cumsum): " << wmh.highres << endl;
	stream1 << "psfi subsampling factor: " << wmh.subsamp << endl;
    
    stream1 << "number of bin per row : " << wmh.prj.Nbin << endl;
    stream1 << "number of rows/slices : " << wmh.prj.Nsli << endl;
    stream1 << "number of detels : " << wmh.prj.Ndt << endl;
    
    stream1 << "detector parameters from: " << wmh.detector_fn << endl;
    stream1 << "collimator parameters from: " << wmh.collim_fn << endl;
    
	//......... psf parameters .................
	
	stream1 << "psf correction: " << wmh.do_psfi << endl;
    if ( wmh.do_psfi ){
	    stream1 << "\tnumber of sigmas in psf calculation: " << wmh.Nsigm << endl;
    }

    //......... correction for depth .................
    
	stream1 << "depth correction: " << wmh.do_depth << endl;

    //......... attenuation parameters .................

	stream1 << "attenuation correction: " << wmh.do_att << endl;
	if ( wmh.do_att ){
		if ( wmh.do_full_att ) stream1 << "\tmode: full " << endl;
		else stream1 << "\tmode: simple " << endl;
        stream1 << "\tattenuation map: " << wmh.att_fn << endl;
	}
    
	//......... masking ....................................
	
    if ( wmh.do_msk_att )  stream1 << "mask type: att" << endl;
    else{
        if ( wmh.do_msk_file ){
            stream1 << "mask type: file" << endl;
            stream1	<< "\tmask file name: " << wmh.msk_fn << endl;
        }
        else stream1 << "mask type: no" << endl;
    }
    
    //... matrix format .....................................
    
    if ( wm.do_save_STIR ) stream1 << "format: STIR" << endl;
    else stream1<< "format: FC " << endl;
    
	stream1.close();
}

//=============================================================================
//=== write_wm_STIR ===========================================================
//=============================================================================

void write_wm_STIR_mph()
{
	int seg_num = 0;             // segment number for STIR matrix (always zero)
	FILE *fid;
	
	if ( ( fid = fopen( wm.fn.c_str() , "wb" )) == NULL ) error_wmtools_SPECT_mph( 31, 0, wm.fn );
	
	//...loop for matrix elements: projection index ..................
	
	for( int j = 0 ; j < wm.Nbt ; j++ ){
		
		//... to write projection indices and number of elements .......
		
		fwrite( &seg_num, sizeof(int), 1, fid);
		fwrite( &wm.na [ j ], sizeof(int), 1, fid);
		fwrite( &wm.ns [ j ], sizeof(int), 1, fid);
		fwrite( &wm.nb [ j ], sizeof(int), 1, fid);
		fwrite( &wm.ne [ j ], sizeof(int), 1, fid);
		
		//... loop for matrix elements: image indexs..................
		
		for ( int i = 0 ; i < wm.ne[ j ] ; i++ ){
			
			fwrite( &wm.nz[ wm.col[ j ][ i ] ], sizeof(short int), 1, fid);
			fwrite( &wm.ny[ wm.col[ j ][ i ] ], sizeof(short int), 1, fid);
			fwrite( &wm.nx[ wm.col[ j ][ i ] ], sizeof(short int), 1, fid);
			fwrite( &wm.val[ j ][ i ], sizeof(float),1,fid);
		}   
	}	
	fclose( fid );
}
#endif

//=============================================================================
//=== precalculated functions ===============================================
//==============================================================================

void fill_pcf()
{
    
    //... distribution function for a round shape hole .................
    
    if ( wmh.do_round_cumsum ){
        
        float lngcmd2 = (float) 0.5 ;
        
        float d1, d2_2, d2;
        
        pcf.round.res = wmh.highres;
        int dimd2     = (int) floorf ( lngcmd2 / pcf.round.res ) + 2 ; // add 2 to have at least one column of zeros as margin
        pcf.round.dim = dimd2 * 2 ;                                    // length of the density function (in resolution elements). even number
        lngcmd2 += (float)2. * pcf.round.res ;                         // center of the function
        
        pcf.round.val   = new float * [ pcf.round.dim ]; // density function allocation
        
        for (int j = 0 ; j < pcf.round.dim ; j++){
            
            pcf.round.val[ j ] = new float [ pcf.round.dim ];
            
            d2   = (float) j * pcf.round.res - lngcmd2 ;
            d2_2 = d2 * d2 ;
            
            for ( int i = 0 ; i < pcf.round.dim; i++) {
                
                d1 = (float) i * pcf.round.res - lngcmd2 ;
                
                if ( sqrtf ( d2_2 + d1 * d1 ) <= (float)0.5 ) pcf.round.val[ j ][ i ] = (float) 1.;
                else  pcf.round.val[ j ][ i ] = (float) 0.;
            }
        }
        
        calc_cumsum( &pcf.round );

        //cout << "\n\tLength of pcf.round density function: " << pcf.round.dim << endl;
    }
    
    //... distribution function for a square shape hole ...................
    
    if ( wmh.do_square_cumsum ){
        
        float lngcmd2   = (float) 0.5 ;
        
        float d1, d2;
        
        pcf.square.res = wmh.highres;
        
        int dimd2     = (int) floorf ( lngcmd2 / pcf.square.res ) + 2 ; // add 2 to have at least one column of zeros as margin
        pcf.square.dim = dimd2 * 2 ;                                    // length of the density function (in resolution elements). even number
        lngcmd2 += (float)2. * pcf.square.res ;                         // center of the function
        
        pcf.square.val   = new float * [ pcf.square.dim ]; // density function allocation
        
        for ( int j = 0 ; j < pcf.square.dim ; j++ ){
            
            pcf.square.val[ j ] = new float [ pcf.square.dim ];
            
            d2 = lngcmd2 - (float) j * pcf.square.res ;
            
            for ( int i = 0 ; i < pcf.square.dim ; i++ ){
                
                if ( fabs( d2 ) > (float)0.5 ) pcf.square.val[ j ][ i ] = (float)0. ;
                else{
                    d1 = lngcmd2 - (float) i * pcf.square.res ;
                    if ( fabs( d1 ) > (float)0.5 ) pcf.square.val[ j ][ i ] = (float)0. ;
                    else pcf.square.val[ j ][ i ] = (float) 1.;
                }
            }
        }
        
        calc_cumsum( &pcf.square );
        
        //cout << "\n\tLength of pcf.square density function: " << pcf.square.dim << endl;
    }
    
    if ( wmh.do_depth ){
        
        pcf.cr_att.dim = (int) floorf( wmh.prj.max_dcr / wmh.highres ) ;
        
        pcf.cr_att.i_max = pcf.cr_att.dim - 1 ;
        
        pcf.cr_att.val = new float [ pcf.cr_att.dim ] ;
        
        float stp = wmh.highres * wmh.prj.crattcoef ;
        
        for ( int i = 0 ; i < pcf.cr_att.dim ; i ++ ) pcf.cr_att.val[ i ] = expf( - (float)i * stp );
        
        //cout << "\n\tLength of exponential to correct for crystal attenuation when do_depth: " << pcf.cr_att.dim << endl;
    }
    
}

//==========================================================================
//=== calc_round_cumsum ===================================================
//==========================================================================

void calc_cumsum ( discrf2d_type *f )
{

    //... cumulative sum by columns ...........................
    
    for ( int j = 0 ; j < f->dim ; j++ ){
        
        for ( int i = 1 ; i < f->dim ; i++ ){
            
            f->val[ j ][ i ] = f->val[ j ][ i ] + f->val[ j ][ i - 1 ];
        }
    }
    
    //... cumulative sum by rows ...............................
    
    for ( int j = 1 ; j < f->dim ; j++ ){
        
        for ( int i = 0 ; i < f->dim ; i++ ){
            
            f->val[ j ][ i ] = f->val[ j ][ i ] + f->val[ j - 1 ][ i ];
        }
    }
    
    //... normalization to one .................................
    
    float vmax = f->val[ f->dim - 1 ][ f->dim - 1 ];
    
    for ( int j = 0 ; j < f->dim ; j++ ){
        
        for ( int i = 0 ; i < f->dim ; i++ ){
            
            f->val[ j ][ i ] /= vmax;
        }
    }
    
    f->i_max = f->j_max = f->dim -1 ;
}

//==============================================================================
//=== read proj params mph ===============================================
//==============================================================================
/*
void read_prj_params_mph()
{
	string token;
    detel_type d;
    
    char DELIMITER = ':';
    
    ifstream stream1;
    stream1.open( wmh.detector_fn.c_str() );
    if( !stream1 ) error_wmtools_SPECT_mph( 122, 0, wmh.detector_fn );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    int Nring = atoi ( token.c_str() );
    
    if ( Nring <= 0 ) error_wmtools_SPECT_mph(222, Nring, "Nring");
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.rad = atof ( token.c_str() );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    float FOVcmx = atof ( token.c_str() );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    float FOVcmz = atof ( token.c_str() );

    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.Nbin = atoi ( token.c_str() );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.Nsli = atoi ( token.c_str() );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.sgm_i = atof ( token.c_str() );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.crth = atof ( token.c_str() );
   
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.crattcoef = atof ( token.c_str() );
    
    //... check for parameters ...........................................
    
    if ( wmh.prj.Nbin <= 0 ) error_wmtools_SPECT_mph( 190, wmh.prj.Nbin ,"Nbin < 1" );
    if ( wmh.prj.Nsli <= 0 ) error_wmtools_SPECT_mph( 190, wmh.prj.Nsli ,"Nsli < 1" );
    
    if ( FOVcmx <= 0.) error_wmtools_SPECT_mph( 190, FOVcmx ,"FOVcmx non positive" );
    if ( FOVcmz <= 0.) error_wmtools_SPECT_mph( 190, FOVcmz ,"FOVcmz non positive" );
    
    if ( wmh.prj.rad  <= 0.) error_wmtools_SPECT_mph( 190, wmh.prj.rad  ,"Drad non positive" );
    if ( wmh.prj.sgm_i < 0.) error_wmtools_SPECT_mph( 190, wmh.prj.sgm_i ,"PSF int: sigma non positive" );
    
    //... derived variables .......................
    
    wmh.prj.Nbd    = wmh.prj.Nsli * wmh.prj.Nbin;
        
    wmh.prj.FOVxcmd2 = FOVcmx / (float) 2.;
    wmh.prj.FOVzcmd2 = FOVcmz / (float) 2.;
    
    wmh.prj.szcm =  FOVcmx /  (float) wmh.prj.Nbin ;
    wmh.prj.thcm =  FOVcmz /  (float) wmh.prj.Nsli ;
    
    wmh.prj.radc   = wmh.prj.rad + wmh.prj.crth ;
    wmh.prj.szcmd2 = wmh.prj.szcm / (float) 2.;
    wmh.prj.thcmd2 = wmh.prj.thcm / (float) 2. ;
    wmh.prj.crth_2 = wmh.prj.crth * wmh.prj.crth ;
    
    if ( !wmh.do_depth ) wmh.prj.rad += wmh.prj.crth / (float)2.; // setting detection plane at half of the crystal thickness
    
    //... print out values (to comment or remove)..............................
    
    cout << "\n\tNumber of rings: " << Nring << endl;
    cout << "\tRadius (cm): " << wmh.prj.rad << endl;
    cout << "\tFOVcmx (cm): " << FOVcmx << endl;
    cout << "\tFOVcmz (cm): " << FOVcmz << endl;
    cout << "\tNumber of bins: " << wmh.prj.Nbin << endl;
    cout << "\tNumber of slices: " << wmh.prj.Nsli << endl;
    cout << "\tBin size (cm): " << wmh.prj.szcm << endl;
    cout << "\tSlice thickness (cm): " << wmh.prj.thcm << endl;
    cout << "\tIntrinsic PSF sigma (cm): " << wmh.prj.sgm_i << endl;
    cout << "\tCrystal thickness (cm): " << wmh.prj.crth << endl;
    
    //... for each ring ..............................
    
    wmh.prj.Ndt    = 0 ;
    
    for ( int i = 0 ; i < Nring ; i++ ){
        
        token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
        int Nang = atoi ( token.c_str() );
        if ( Nang <= 0 ) error_wmtools_SPECT_mph( 190, Nang ,"Nang < 1" );
        
        token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
        float ang0 = atof ( token.c_str() );
        
        token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
        float incr = atof ( token.c_str() );
        
        token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
        d.z0 = atof ( token.c_str() );
        
        d.nh = 0;
        
        for ( int j= 0 ; j < Nang ; j++ ){
            
            //... angles and ratios ................................................
            
            d.theta = ( ang0 + (float)j * incr ) * dg2rd;   // projection angle in radians
            d.costh = cosf( d.theta );	                    // cosinus of the angle
            d.sinth = sinf( d.theta );	                    // sinus of the angle
            
            //... cartesian coordinates of the center of the detector element .............
            
            d.x0 = wmh.prj.rad * d.costh;
            d.y0 = wmh.prj.rad * d.sinth;
            
            //... coordinates of the first bin of each projection and increments for consecutive bins ....
            
            if(wmh.do_att){
                
                d.incx  = wmh.prj.szcm * d.costh;
                d.incy  = wmh.prj.szcm * d.sinth;
                d.incz  = wmh.prj.thcm;
                
                d.xbin0 = -wmh.prj.rad * d.sinth - ( wmh.prj.FOVxcmd2 + wmh.prj.szcm * (float)0.5 ) * d.costh ;
                d.ybin0 =  wmh.prj.rad * d.costh - ( wmh.prj.FOVxcmd2 + wmh.prj.szcm * (float)0.5 ) * d.sinth ;
                d.zbin0 =  d.z0 - wmh.prj.FOVzcmd2 + wmh.prj.thcmd2 ;
            }
            wmh.detel.push_back( d );
        }
      
        //... update of wmh cumulative values .....................................
        
        wmh.prj.Ndt += Nang ;
        
        //... print out values (to comment or remove)..............................
        
        cout << "\n\tDetector ring: " << i << endl;
        cout << "\tNumber of angles: " << Nang << endl;
        cout << "\tang0: " << ang0 << endl;
        cout << "\tincr: " << incr << endl;
        cout << "\tz0: " << d.z0 << endl;
        cout << "\tNumber of holes per detel: " << d.nh << endl;
    }
    
    //... fill detel .....................................................
    
    stream1.close();
    
    wmh.prj.Nbt  = wmh.prj.Nbd * wmh.prj.Ndt ;
    
    cout << "\n\tTotal number of detels: " << wmh.prj.Ndt << endl;
    cout << "\tTotal number of bins: " << wmh.prj.Nbt << endl;
    
    return;
}*/


void read_prj_params_mph()
{
	string token;
    detel_type d;
    std::stringstream info_stream;
    
    char DELIMITER = ':';
    
    ifstream stream1;
    stream1.open( wmh.detector_fn.c_str() );
    if( !stream1 ) error_wmtools_SPECT_mph( 122, 0, wmh.detector_fn );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    int Nring = atoi ( token.c_str() );
    
    if ( Nring <= 0 ) error_wmtools_SPECT_mph(222, Nring, "Nring");
    /*
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    float FOVcmx = atof ( token.c_str() );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    float FOVcmz = atof ( token.c_str() );

    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.Nbin = atoi ( token.c_str() );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.Nsli = atoi ( token.c_str() );
    */
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.sgm_i = atof ( token.c_str() );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.crth = atof ( token.c_str() );
   
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.prj.crattcoef = atof ( token.c_str() );
    
    //... check for parameters ...........................................

    if ( wmh.prj.Nbin <= 0 ) error_wmtools_SPECT_mph( 190, wmh.prj.Nbin ,"Nbin < 1" );
    if ( wmh.prj.Nsli <= 0 ) error_wmtools_SPECT_mph( 190, wmh.prj.Nsli ,"Nsli < 1" );
    
    if ( wmh.prj.szcm <= 0.) error_wmtools_SPECT_mph( 190, wmh.prj.szcm ,"szcm non positive" );
    if ( wmh.prj.thcm <= 0.) error_wmtools_SPECT_mph( 190, wmh.prj.thcm ,"thcm non positive" );
    
    if ( wmh.prj.rad  <= 0.) error_wmtools_SPECT_mph( 190, wmh.prj.rad  ,"Drad non positive" );
    if ( wmh.prj.sgm_i < 0.) error_wmtools_SPECT_mph( 190, wmh.prj.sgm_i ,"PSF int: sigma non positive" );

    //... derived variables .......................
    
    wmh.prj.radc   = wmh.prj.rad + wmh.prj.crth ;
    wmh.prj.crth_2 = wmh.prj.crth * wmh.prj.crth ;
    
    if ( !wmh.do_depth ) wmh.prj.rad += wmh.prj.crth / (float)2.; // setting detection plane at half of the crystal thickness
    
    //... print out values (to comment or remove)..............................
    
    info_stream << "Projection parameters" << endl;
    info_stream << "Number of rings: " << Nring << endl;
    info_stream << "Radius (cm): " << wmh.prj.rad << endl;
    info_stream << "FOVcmx (cm): " << wmh.prj.FOVxcmd2*2. << endl;
    info_stream << "FOVcmz (cm): " << wmh.prj.FOVzcmd2*2. << endl;
    info_stream << "Number of bins: " << wmh.prj.Nbin << endl;
    info_stream << "Number of slices: " << wmh.prj.Nsli << endl;
    info_stream << "Bin size (cm): " << wmh.prj.szcm << endl;
    info_stream << "Slice thickness (cm): " << wmh.prj.thcm << endl;
    info_stream << "Intrinsic PSF sigma (cm): " << wmh.prj.sgm_i << endl;
    info_stream << "Crystal thickness (cm): " << wmh.prj.crth << endl;
    
    //... for each ring ..............................
    
    wmh.prj.Ndt    = 0 ;
    
    for ( int i = 0 ; i < Nring ; i++ ){
        
        token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
        int Nang = atoi ( token.c_str() );
        if ( Nang <= 0 ) error_wmtools_SPECT_mph( 190, Nang ,"Nang < 1" );
        
        token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
        float ang0 = atof ( token.c_str() );
        
        token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
        float incr = atof ( token.c_str() );
        
        token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
        d.z0 = atof ( token.c_str() );
        
        d.nh = 0;
        
        for ( int j= 0 ; j < Nang ; j++ ){
            
            //... angles and ratios ................................................
            
            d.theta = ( ang0 + (float)j * incr ) * dg2rd;   // projection angle in radians
            d.costh = cosf( d.theta );	                    // cosinus of the angle
            d.sinth = sinf( d.theta );	                    // sinus of the angle
            
            //... cartesian coordinates of the center of the detector element .............
            
            d.x0 = wmh.prj.rad * d.costh;
            d.y0 = wmh.prj.rad * d.sinth;
            
            //... coordinates of the first bin of each projection and increments for consecutive bins ....
            
            if(wmh.do_att){
                
                d.incx  = wmh.prj.szcm * d.costh;
                d.incy  = wmh.prj.szcm * d.sinth;
                d.incz  = wmh.prj.thcm;
                
                d.xbin0 = -wmh.prj.rad * d.sinth - ( wmh.prj.FOVxcmd2 + wmh.prj.szcm * (float)0.5 ) * d.costh ;
                d.ybin0 =  wmh.prj.rad * d.costh - ( wmh.prj.FOVxcmd2 + wmh.prj.szcm * (float)0.5 ) * d.sinth ;
                d.zbin0 =  d.z0 - wmh.prj.FOVzcmd2 + wmh.prj.thcmd2 ;
            }
            wmh.detel.push_back( d );
        }
      
        //... update of wmh cumulative values .....................................
        
        wmh.prj.Ndt += Nang ;
        
        //... print out values (to comment or remove)..............................
        
        info_stream << "\nDetector ring: " << i << endl;
        info_stream << "Number of angles: " << Nang << endl;
        info_stream << "ang0: " << ang0 << endl;
        info_stream << "incr: " << incr << endl;
        info_stream << "z0: " << d.z0 << endl;
        info_stream << "Number of holes per detel: " << d.nh << endl;
    }
    
    //... fill detel .....................................................
    
    stream1.close();
    
    wmh.prj.Nbt  = wmh.prj.Nbd * wmh.prj.Ndt ;
    
    info_stream << "\nTotal number of detels: " << wmh.prj.Ndt << endl;
    info_stream << "Total number of bins: " << wmh.prj.Nbt << endl;
    
    stir::info(info_stream.str());
    
    return;
}



///=============================================================================
//=== read collimator params mph ===============================================
//==============================================================================

void read_coll_params_mph( )
{
	string token;
    vector<string> param;
    std::stringstream info_stream;
    
    char DELIMITER = ':';
	
    ifstream stream1;
    stream1.open( wmh.collim_fn.c_str() );
    
	if( !stream1 ) error_wmtools_SPECT_mph( 122, 0, wmh.collim_fn );
    
    wmh.collim.model = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
   
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.collim.rad = atof ( token.c_str() );
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.collim.L   = atof ( token.c_str() );
    wmh.collim.Ld2 =  wmh.collim.L / (float)2. ;
    
    token = wm_SPECT_read_value_1d ( &stream1, DELIMITER );
    wmh.collim.Nht = atoi ( token.c_str() );
    
//    wmh.collim.holes = new hole_type [ wmh.collim.Nht ];
    
    int nh = 0;
    if ( wmh.collim.model == "cyl" ) wm_SPECT_read_hvalues_mph( &stream1, DELIMITER, &nh, true );
    else{
        if ( wmh.collim.model == "pol" ) wm_SPECT_read_hvalues_mph( &stream1, DELIMITER, &nh, false );
        else error_wmtools_SPECT_mph ( 334, 0, wmh.collim.model );
    }
    
    if ( nh != wmh.collim.Nht ) error_wmtools_SPECT_mph( 150, nh, "" );
    
    //... check for parameters ...........................................
    
    
    if ( wmh.collim.rad <= 0. ) error_wmtools_SPECT_mph( 190, wmh.collim.rad ,"Collimator radius non positive" );
    
    if ( wmh.collim.Nht <= 0 ) error_wmtools_SPECT_mph( 190, wmh.collim.Nht ,"Number of Holes < 1" );
    
    //... print out values (to comment or remove)..............................
    
    stream1.close();
    
    info_stream << "Collimator parameters" << endl;
    info_stream << "\nCollimator model: " << wmh.collim.model << endl;
	info_stream << "Collimator rad: " << wmh.collim.rad << endl;
	info_stream << "Number of holes: " << wmh.collim.Nht << endl;
    
    stir::info(info_stream.str());

    return;
}

//=====================================================================
//======== wm_SPECT_read_hvalues_mph ==============================
//=====================================================================

void wm_SPECT_read_hvalues_mph( ifstream * stream1 , char DELIMITER, int * nh, bool do_cyl )
{
    
    size_t pos1, pos2, pos3;
    string line, token;
    hole_type h;
    
    float max_hsxcm  = (float) 0.;
    float max_hszcm  = (float) 0.;
    
    float max_aix  = (float) 0.;
    float max_aiz  = (float) 0.;

    *nh = 0 ;
    
    while ( getline ( *stream1, line ) ){
        
        pos1 = line.find( DELIMITER );
        
        if ( pos1 == string::npos ) continue;
        
        //... detel index ...................
        
        pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
        pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
        if ( pos2 == string::npos || pos3 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "idet" );
        token = line.substr( pos2 , pos3 - pos2 );
        int idet = atoi( token.c_str() ) - 1 ;
        wmh.detel[ idet ].who.push_back( *nh );
        wmh.detel[ idet ].nh ++;
        pos1  = pos3;
        
        //... second parameter ...........................
        
        if ( do_cyl ){
            
            //... angle ...........................
            
            pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
            pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
            if ( pos2 == string::npos || pos3 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "angle(deg)" );
            token = line.substr( pos2 , pos3 - pos2 );
            h.acy = atof( token.c_str() ) * dg2rd ;
            pos1  = pos3;
        }
        else{
            
            //... x position ...........................
            
            pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
            pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
            if ( pos2 == string::npos || pos3 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "x(cm)" );
            token = line.substr( pos2 , pos3 - pos2 );
            h.x1 = atof( token.c_str() );
            pos1  = pos3;
        }
    
        //... y position (along collimator wall. 0 for centrer of collimator wall) ...................
        
        pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
        pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
        if ( pos2 == string::npos || pos3 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "y(cm)" );
        token = line.substr( pos2 , pos3 - pos2 );
        float yd = atof( token.c_str() );
        pos1  = pos3;
        
        //... z position ...................
        
        pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
        pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
        if ( pos2 == string::npos || pos3 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "z(cm)" );
        token = line.substr( pos2 , pos3 - pos2 );
        h.z1 = atof( token.c_str() );
        pos1  = pos3;
        
        //... shape .............................
        
        pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
        pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
        if ( pos2 == string::npos || pos3 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "shape" );
        token = line.substr( pos2 , pos3 - pos2 );
        if ( token.compare( "rect") != 0 && token.compare( "round" ) != 0 ) error_wmtools_SPECT_mph ( 444, *nh, "");
        h.shape = token.c_str();
        pos1  = pos3;
        
        if ( token.compare( "rect") == 0 ){
            wmh.do_square_cumsum = true;
            h.do_round = false;
        }
        else{
            wmh.do_round_cumsum = true;
            h.do_round = true;
        }
        
        //... dimension x cm .......................
        
        pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
        pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
        if ( pos2 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "dxcm" );
        token = line.substr( pos2 , pos3 - pos2 );
        h.dxcm = atof( token.c_str() );
        if ( h.dxcm > max_hsxcm ) max_hsxcm = h.dxcm;
        pos1  = pos3;
        
        //... dimension z cm .......................
        
        pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
        pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
        if ( pos2 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "dzcm" );
        token = line.substr( pos2 , pos3 - pos2 );
        h.dzcm = atof( token.c_str() );
        if ( h.dzcm > max_hszcm ) max_hszcm = h.dzcm;
        pos1  = pos3;
        
        //... hole axial angle x .......................
        
        pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
        pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
        if ( pos2 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "ahx" );
        token = line.substr( pos2 , pos3 - pos2 );
        h.ahx = atof( token.c_str() ) * dg2rd ;
        pos1  = pos3;
     
        //... hole axial angle z .......................
        
        pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
        pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
        if ( pos2 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "ahz" );
        token = line.substr( pos2 , pos3 - pos2 );
        h.ahz = atof( token.c_str() ) * dg2rd ;
        pos1  = pos3;
        
        //... x acceptance angle ........................
        
        pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
        pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
        if ( pos2 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "aa_x" );
        token = line.substr( pos2 , pos3 - pos2 );
        h.aa_x = atof( token.c_str() ) * dg2rd ;
        pos1  = pos3;
        
        //... z acceptance angle ........................
        
        pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
        pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
        if ( pos2 == string::npos ) error_wmtools_SPECT_mph ( 333, *nh, "aa_z" );
        token = line.substr( pos2 , pos3 - pos2 );
        h.aa_z = atof( token.c_str() ) * dg2rd ;
        
        //... derived variables ........................................
        
        if( do_cyl ) {
            h.acyR = h.acy - wmh.detel[ idet ].theta ;
            h.x1   = ( wmh.collim.rad  + yd ) * sinf( h.acyR );
            h.y1   = ( wmh.collim.rad  + yd ) * cosf( h.acyR ) ;
        }
        else{
             h.y1  = wmh.collim.rad + yd ;
        }
        
        h.z1  = wmh.detel[ idet ].z0 + h.z1 ;
        
        //... edge slope x,z minor and major .......................
        
        h.Egx = h.ahx + h.aa_x ;
        h.egx = h.ahx - h.aa_x ;
        h.Egz = h.ahz + h.aa_z ;
        h.egz = h.ahz - h.aa_z ;
        
        //... angles max and min ..........................................
        
        h.ax_M = h.Egx ;
        h.ax_m = h.egx ;
        h.az_M = h.Egz ;
        h.az_m = h.egz ;
        
        //... incidence angle maximum, for PSF allocation when correction for depth...............
        
        if ( fabs ( h.ax_m ) > max_aix  ) max_aix = h.ax_m ;
        if ( fabs ( h.ax_M ) > max_aix  ) max_aix = h.ax_M ;
        
        if ( fabs ( h.az_m ) > max_aiz  ) max_aiz = h.az_m ;
        if ( fabs ( h.az_M ) > max_aiz  ) max_aiz = h.az_M ;
        
        wmh.collim.holes.push_back ( h );
        
        *nh = *nh + 1;
    }
    
    //... maximum hole dimensions and incidence angles .................
    
    wmh.max_hsxcm = max_hsxcm;
    wmh.max_hszcm = max_hszcm;
    
    wmh.tmax_aix = tanf( max_aix );
    wmh.tmax_aiz = tanf( max_aiz );
    
    wmh.prj.max_dcr = (float)1.2 * wmh.prj.crth / cosf( max ( max_aix , max_aiz ) ) ;
}

//=============================================================================
//=== generate_msk_mph ========================================================
//=============================================================================

void generate_msk_mph ( bool *msk_3d, float *attmap )
{
    
//    bool do_save_resulting_msk = true;
	
	//... to create mask from attenuation map ..................
	
	if ( wmh.do_msk_att ){                                   
		for ( int i = 0 ; i < wmh.vol.Nvox ; i++ ){
			msk_3d[ i ] = ( attmap[ i ] > EPSILON );                
		}
	}
	else {
		//... to read a mask from a (int) file ....................

        if ( wmh.do_msk_file ) stir::error("Mask incorrectly read from file."); //read_msk_file_mph( msk_3d );  // STIR implementation never calls this to avoid using read_msk_file_mph
		
		else {
            
            //... to create a cylindrical mask......................
            
			float xi2, yi2;
 
			float Rmax2 = wmh.ro * wmh.ro; // Maximum allowed radius (distance from volume centre)
            
			for ( int j = 0 , ip = 0 ; j < wmh.vol.Dimy ; j++ ){
				
				yi2 = ( (float)j + (float)0.5 ) * wmh.vol.szcm - wmh.vol.FOVcmyd2 ;
				yi2 *= yi2;
				
				for ( int i = 0 ; i < wmh.vol.Dimx ; i++ , ip++ ){

					xi2  = ( (float)i + (float)0.5 ) * wmh.vol.szcm - wmh.vol.FOVxcmd2 ;
					xi2 *= xi2;
					
					if ( ( xi2 + yi2 ) > Rmax2 ){
                        
						for ( int k = 0 ; k < wmh.vol.Dimz ; k ++ ) msk_3d[ ip + k * wmh.vol.Npix ] = false;
					}
                    else {
                        for ( int k = 0 ; k < wmh.vol.Dimz ; k ++ ) msk_3d[ ip + k * wmh.vol.Npix ] = true;
                    }
				}
			}
		}
	}
}

//=============================================================================
//=== read_mask file ==========================================================
//=============================================================================
#if 0
void read_msk_file_mph( bool *msk )
{
	FILE *fid;
	int *aux;
	
	aux = new int [ wmh.vol.Nvox ];
	
	if ( (fid = fopen( wmh.msk_fn.c_str() , "rb")) == NULL) error_wmtools_SPECT_mph( 126, 26, wmh.msk_fn);
	fread( aux, sizeof(int), wmh.vol.Nvox, fid);
	fclose(fid);
	
	for (int i = 0 ; i < wmh.vol.Nvox ; i ++ )	msk[ i ] = ( aux[ i ] != 0 );
	
	delete [] aux;
}
#endif

//=============================================================================
//=== read_att_map ============================================================
//=============================================================================
#if 0
void read_att_map_mph( float *attmap )
{
	FILE *fid;
	if ( ( fid = fopen( wmh.att_fn.c_str() , "rb") ) == NULL ) error_wmtools_SPECT_mph ( 124, 24, wmh.att_fn );
	fread( attmap, sizeof(float), wmh.vol.Nvox, fid);
	
	bool exist_nan = false;
	
	for (int i = 0 ; i < wmh.vol.Nvox ; i++ ){
		if ((boost::math::isnan)(attmap [ i ])){
			attmap [ i ] = 0;
			exist_nan = true;
		}
	}
	
	if ( exist_nan ) cout << "WARNING: att map contains NaN values. Converted to zero" << endl;
	
	fclose( fid );
}
#endif

//=====================================================================
//======== wm_SPECT_read_value_1d =====================================
//=====================================================================

string wm_SPECT_read_value_1d ( ifstream * stream1, char DELIMITER )
{
    
    size_t pos1, pos2, pos3;
    string line;
    
    int k = 0 ;
    
    while ( !stream1->eof() ){
        getline ( *stream1, line );
        
        pos1 = line.find( DELIMITER );
        
        if ( pos1 != string::npos ){
            k++;
            break;
        }
    }
    
    if ( k == 0 ) error_wmtools_SPECT_mph (888, 0 ,"" );
    
    pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
    pos3 = line.find_first_of(" \t\f\v\n\r", pos2);
    
    return( line.substr( pos2 , pos3 - pos2 ) );
}

//=============================================================================
//=== itoa ====================================================================
//=============================================================================
#if 0
char *itoa(int n,char *s)
{
	int i,sign;
	char c;
	
	if((sign=n)<0) {
		n=-n;
	}
	
	i=0;
	do{
		s[i++]=n%10+'0';
	}
	while((n/=10)>0);
	
	if(sign<0) {
		s[i++]='-';      
	}
	
	s[i]=EOS;
	
	for(int low=0,hi=i-1;low<hi;low++,hi--){
		c=s[low];
		s[low]=s[hi];
		s[hi]=c;      
	}
	
	return(s);
}
#endif

//=============================================================================
//=== free_wm =================================================================
//=============================================================================
//*** this block moved to delete_PinholeSPECTUB_arrays()
//=============================================================================
#if 0 
void free_wm( )
{
    //... freeing wm.val and wm.col ...................................
    
    for( int i = 0 ; i < wmh.prj.Nbt ; i++ ){
        
        delete [] wm.val[ i ];
        delete [] wm.col[ i ];
    }
    
    //... freeing matrix memory....................................
    
    delete [] wm.val;
    delete [] wm.col;
    delete [] wm.ne;
    
    if ( wm.do_save_STIR ){
        delete [] wm.ns;
        delete [] wm.nb;
        delete [] wm.na;
        delete [] wm.nx;
        delete [] wm.ny;
        delete [] wm.nz;
    }
}
#endif

//=============================================================================
//=== free_pcf ==============================================================
//=============================================================================
#if 0
void free_pcf( )
{
    if ( wmh.do_round_cumsum ){
        for ( int i = 0 ; i < pcf.round.dim ; i ++ ) delete [] pcf.round.val[ i ];
        delete [] pcf.round.val;
    }
    
    if ( wmh.do_square_cumsum ){
        for ( int i = 0 ; i < pcf.square.dim ; i ++ ) delete [] pcf.square.val[ i ];
        delete [] pcf.square.val;
    }
    
    if ( wmh.do_depth ) delete pcf.cr_att.val ;
}
#endif

//=============================================================================
//== error_wmtools_SPECT_mph ======================================================
//=============================================================================

void error_wmtools_SPECT_mph( int nerr, int ip, string txt )
{
#if 0
	switch(nerr){

		case 13: printf("\n\nError wmtools_SPECT: not enough parameters in collimator file: %s \n",txt.c_str());break;
		case 30: printf("\n\nError wmtools_SPECT: can not open %s for reading\n",txt.c_str()); break;
		case 31: printf("\n\nError wmtools_SPECT: can not open %s for writing\n",txt.c_str()); break;
        case 55: printf( "\n\nError %d weight3d: dowmsampling. Incongruency factor-dim: %d \n", nerr, ip ); break;
        case 56: printf( "\n\nError %d weight3d: downsampling. Resulting dim bigger than max %d \n", nerr, ip ); break;
        case 77: printf( "\n\nError %d weight3d: convolution. psf_out is not big enough %d \n", nerr, ip ); break;
			
		//... error: value of argv[]..........................
		
		case 122: printf("\n\nError wm_SPECT: file with variable collimator parameters: %s not found\n",txt.c_str() ); break;
		case 124: printf("\n\nError wm_SPECT: can not open attenuation map-> argv[%d]: %s for reading\n", ip, txt.c_str() ); break; //should be case 119: argv[19]
		case 126: printf("\n\nError wm_SPECT: can not open file mask-> argv[%d]: %s for reading\n",ip, txt.c_str() ); break; //should be case 121: argv[21]
        case 150: printf("\n\nError wm_SPECT: list of hole parameters has different length (%d) than number of holes\n", ip); break;
        case 190: printf("\n\nError wm_SPECT: wrong value in detector parameter: %s \n", txt.c_str()); break;
        case 200: printf("\n\nError wm_SPECT: cannot allocate %d element of the variable: %s\n",ip, txt.c_str() ); break;
        case 222: printf("\n\nError wm_SPECT: wrong number of ring rings: %d\n", ip ); break;
        case 333: printf("\n\nError wm_SPECT: missing parameter in hole %d definition: %s\n", ip, txt.c_str() ); break;
        case 334: printf("\n\nError wm_SPECT: %s unknown collimator model. Options: cyl/pol\n", txt.c_str() ); break;
        case 444: printf("\n\nError wm_SPECT: hole %d: wrong hole shape. Hole shape should be either rect or round\n",ip); break;
        case 888: printf("\n\nError wm_SPECT: missing parameter in collimator file\n"); break;
		default: printf("\n\nError wmtools_SPECT: %d unknown error number on error_wmtools_SPECT()",nerr);
	}
	
	exit(0);
#else
    using stir::error;
    switch(nerr){

        case 55: printf( "\n\nError %d weight3d: Dowmsampling. Incongruency factor-dim: %d \n", nerr, ip ); break;
        case 56: printf( "\n\nError %d weight3d: Downsampling. Resulting dim bigger than max %d \n", nerr, ip ); break;
        case 77: printf( "\n\nError %d weight3d: Convolution. psf_out is not big enough %d. Verify geometry. \n", nerr, ip ); break;
        case 78: printf( "\n\nError %d weight3d: Geometric PSF. psf_out is not big enough %d. Verify geometry. \n", nerr, ip ); break;
			
		//... error: value of argv[]..........................
		
		case 122: printf("\n\nError wm_SPECT: File with variable parameters: %s not found.\n",txt.c_str() ); break;
		case 124: printf("\n\nError wm_SPECT: Cannot open attenuation map: %s for reading..\n", txt.c_str() ); break;
		case 126: printf("\n\nError wm_SPECT: Cannot open file mask: %s for reading\n",txt.c_str() ); break;
        case 150: printf("\n\nError wm_SPECT: List of hole parameters has different length (%d) than number of holes.\n", ip); break;
        case 190: printf("\n\nError wm_SPECT: Wrong value in detector parameter: %s \n", txt.c_str()); break;
        case 200: printf("\n\nError wm_SPECT: Cannot allocate %d element of the variable: %s\n",ip, txt.c_str() ); break;
        case 222: printf("\n\nError wm_SPECT: Wrong number of rings: %d\n", ip ); break;
        case 333: printf("\n\nError wm_SPECT: Missing parameter in hole %d definition: %s\n", ip, txt.c_str() ); break;
        case 334: printf("\n\nError wm_SPECT: %s unknown collimator model. Options: cyl/pol.\n", txt.c_str() ); break;
        case 444: printf("\n\nError wm_SPECT: Hole %d: Wrong hole shape. Hole shape should be either rect or round.\n",ip); break;
        case 888: error("\n\nError wm_SPECT: Missing parameter in collimator file.\n"); break;
		default: printf("\n\nError wmtools_SPECT: %d unknown error number on error_wmtools_SPECT().",nerr);
	}
	
	exit(0);
#endif
}    

//==========================================================================
//=== error_wm ====================================================
//==========================================================================
//=== Originally implemented in wm_SPECT_mph.cpp
//=== Deprecated after main integration into ProjMatricByBinPinholeSPECTUB.cxx  
//=== STIR error handling is with error() and warning()
//==========================================================================
#if 0
void error_wm_SPECT_mph( int nerr, string txt)
{
    string opcions[]={
        "\nargv[1]  Matrix file: Weight matrix filename (without extension index)",
        
        "\nargv[2]  Image box: Number of columns (int)",
        "\nargv[3]  Image box: Number of rows (int)",
        "\nargv[4]  Image box: Number of slices (the same than projection slices) (int)",
        "\nargv[5]  Image box: Voxel side length(cm). Only square voxels are considered (float cm)",
        "\nargv[6]  Image box: Slice thickness (the same than projection slice thickness) (float cm)",
        
        "\nargv[7]  Image: First slice to reconstruct (1 to Nslices)",
        "\nargv[8]  Image: Last slice to reconstruct (1 to Nslices)",
        "\nargv[9]  Image: Object radius (cm)",
        
        "\nargv[10] Projection: file containig ring information",
        "\nargv[11] Projections: File with the collimator parameters",
        
        "\nargv[12] Matrix: Minimum weight to take into account (over 1. Typically 0.01)",
        "\nargv[13] Matrix: Maximum number of sigmas to consider in PSF calculation (float)",
        
        "\nargv[14] Matrix: Spatial high resolution in which to sample PSF distributions (typically 0.001)",
        "\nargv[15] Matrix: Subsampling factor (usually 1-8)",
        
        "\nargv[16] Matrix: Correction for intrinsic PSF (no/yes)",
        "\nargv[17] Matrix: Correction for impact depth (no/yes)",
        "\nargv[18] Matrix: Correction for attenuation (no/simple/full)",
        "\nargv[19] Matrix: attenuation map (filename/no) (in case of explicit mask)",
        
        "\nargv[20] Matrix: volume masking (att/file/no). Inscrit cylinder by default. att: mask with att=0",
        "\nargv[21] Matrix: explicit mask (filename/no) (in case of explicit mask)",
        
        "\nargv[22]  Matrix file: Format. Options: STIR, FC (FruitCake)"
    };
    
    switch(nerr){
        case 100: cout << endl << "Missing variables" << endl;
            for ( int i = 0 ; i < NUMARG-1 ; i++ ){
                printf( "%s\n", opcions[ i ].c_str() );
            }
            break;
            
        //... error: value of argv[] ........................................
            
        case 101: printf("\n\nError %d wm_SPECT_mph: parametre file: %s not found\n", nerr, txt.c_str() );break;
        case 102: printf("\n\nError %d wm_SPECT_mph: More parametres tan expected in file: %s\n", nerr, txt.c_str() );break;
        case 103: printf("\n\nError %d wm_SPECT_mph: Less parametres tan expected in file: %s\n", nerr, txt.c_str() );break;
        case 107: printf("\n\nError %d wm_SPECT_mph: first slice to reconstruct out of range (1->Nslic): %s \n", nerr, txt.c_str() );break;
        case 108: printf("\n\nError %d wm_SPECT_mph: last slice to reconstruct out of range (first slice->Nslic): %s \n", nerr, txt.c_str() );break;
        case 111: printf("\n\nError %d wm_SPECT_mph: number of subsets should be congruent with number of projection angles\n", nerr ); break;
        case 116: printf("\n\nError %d wm_SPECT_mph: invalid option for argv[16]. Options: no/yes. Read value: %s \n", nerr, txt.c_str() );break;
        case 117: printf("\n\nError %d wm_SPECT_mph: invalid option for argv[17]. Options: no/yes. Read value: %s \n", nerr, txt.c_str() );break;
        case 118: printf("\n\nError %d wm_SPECT_mph: invalid option for argv[18]. Options: no/simple/full. Read value: %s \n", nerr, txt.c_str() );break;
        case 120: printf("\n\nError %d wm_SPECT_mph: invalid option for argv[20]. Options: no/att/file. Read value: %s \n", nerr, txt.c_str() );break;
        case 122: printf("\n\nError %d wm_SPECT_mph: invalid option for argv[22]. Options: STIR/FC. Read value: %s \n", nerr, txt.c_str() );break;
            
        //... other errors...........................................................
            
        case 150: printf("\n\nError %d wm_SPECT: second delimiter missing in file of parameters. Param: %s", nerr, txt.c_str() ); break;
        case 200: printf("\n\nError %d wm_SPECT: cannot allocate the variable: %s\n", nerr, txt.c_str() );break;
            
        default: printf("\n\nError %d wm_SPECT: unknown error number", nerr);
    }
    
    exit(0);
}
#endif

} // end of namespace

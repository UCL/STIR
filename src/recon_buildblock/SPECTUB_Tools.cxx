/*
* Copyright (c) 2013,
* Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. All rights reserved.
* This software is distributed WITHOUT ANY WARRANTY; 
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  \author Carles Falcon
*/

//system libraries
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include "stir/error.h"
#include <boost/format.hpp>

using namespace std;
//using std::string;

//... user defined libraries .......................................

#include "stir/recon_buildblock/SPECTUB_Tools.h"
#include "stir/recon_buildblock/SPECTUB_Weight3d.h"


namespace SPECTUB {

#define NUMARG 29  

#define EPSILON 1e-12
#define EOS '\0'

#define maxim(a,b) ((a)>=(b)?(a):(b))
#define minim(a,b) ((a)<=(b)?(a):(b))
#define abs(a) ((a)>=0?(a):(-a))
#define SIGN(a) (a<-EPSILON?-1:(a>EPSILON?1:0))
 
#ifndef M_PI
#define M_PI 3.14159265
#endif

#define DELIMITER1 '#' //delimiter character in input parameter text file
#define DELIMITER2 '%' //delimiter character in input parameter text file

//... global variables ..............................................

extern wm_da_type wm;
extern wmh_type wmh; 
extern float * Rrad;


//=============================================================================
//=== write_wm_FC =============================================================
//=============================================================================

void write_wm_FC()
{
	FILE *fid;
	
	int ia_acum = 0;
	
	if ( (fid = fopen( wm.OSfn.c_str(), "wb" ) ) == NULL ) error_wmtools_SPECT( 31, wm.OSfn );
	
	fwrite ( &(wm.NbOS), sizeof(int), 1, fid);  // to write number of rows of wm (NbOS)
	fwrite ( &(wm.Nvox), sizeof(int), 1, fid);  // to write number of columns of wm (Nvox)
	
	//... number of non-zero elements in the weight matrix .......
	
	int ne = 0;
	for ( int j=0 ; j < wm.NbOS ; j++ ){
		ne += wm.ne[j];
    }
	fwrite ( &ne, sizeof(int), 1, fid);         // to write number of non-zeros element in the weight matrix
	
	//... to write the array of weights (along rows) ..............
	
	for ( int i = 0 ; i < wm.NbOS ; i++ ){
		for (int j = 0 ; j < wm.ne[i] ; j++ ){
			fwrite ( &wm.val[ i ][ j ], sizeof(float), 1, fid);
		}
	}
	
	//... to write the column index of each weight (volume index of the voxel the weight is associated to) ....
	
    for ( int i = 0 ; i < wm.NbOS ; i++ ){
		for ( int j = 0 ; j < wm.ne[ i ] ; j++ ){
			fwrite ( &wm.col[ i ][ j ] ,sizeof(int) ,1 , fid);
		}
	}
	//... to write the indexs of the array of weights where a change of row happens .........
	
	for ( int i = 0 ; i < wm.NbOS ; i++ ){
		fwrite ( &ia_acum, sizeof(int), 1, fid);
		ia_acum += wm.ne[i];
	}

	//... to write the total number of saved weights ..........................
	
	fwrite ( &ia_acum, sizeof(int), 1, fid);
	
	cout << "number of non-zero elemnts: " << ia_acum << endl;

	fclose (fid);
}

//=============================================================================
//=== write_wm_hdr ============================================================
//=============================================================================

void write_wm_hdr()
{
	ofstream stream1( wm.fn_hdr.c_str() );
	if( !stream1 ) error_wmtools_SPECT( 31, wm.fn_hdr );  
	
	//....... image and projections characteristics.........

	stream1 << "Header for the matrix " << wm.fn << endl;
	stream1 << "number of columns: " << wmh.vol.Ncol << endl;
	stream1 << "number of rows: " << wmh.vol.Nrow << endl;
	stream1 << "number of slices: " << wmh.vol.Nsli << endl;
	stream1 << "voxel size (cm): " << wmh.vol.szcm << endl;
	stream1 << "slice thickness (cm): " << wmh.vol.thcm << endl;
	
	stream1 << "number of bins per line: " << wmh.prj.Nbin << endl;
	stream1 << "bin size (cm): " << wmh.prj.szcm << endl;
	stream1 << "number of angles: " << wmh.prj.Nang << endl;
	stream1 << "first angle (deg): " << wmh.prj.ang0 << endl;
	stream1 << "angle increment between consecutive projections (deg): " << wmh.prj.incr << endl;
	
	stream1 << "first slice to reconstruct : " << wmh.vol.first_sl << endl;
	stream1 << "last slice to reconstruct : " << wmh.vol.last_sl << endl;
	stream1 << "number of subsets in which to split the matrix: " << wmh.prj.NOS << endl;
	stream1 << "number of angles per subsets: " << wmh.prj.NangOS << endl;

	stream1 << "minimum weight (geometrical contribution): " << wmh.min_w << endl;
	stream1 << "psf resolution (discretization interval for Gaussian): " << wmh.psfres << endl;
	stream1 << "maximum number of sigmas in psf calculation: " << wmh.maxsigm << endl;
	
	//........ rotation radius................................
	
	if ( wmh.fixed_Rrad ) stream1 << "fixed rotation radius :" << wmh.Rrad[ 0 ] << " cm"	<< endl;
	else stream1 << "variable rotation radius from  :" << wmh.Rrad_fn << endl;
		
	//......... psf and collimator parameters .................

	
	stream1 << "psf correction: " << wmh.do_psf << endl;
	if ( wmh.do_psf ){
		if ( wmh.do_psf_3d ) stream1 << "\t mode: 3d " << endl;
		else stream1 << "\t mode: 2d " << endl;
		if ( wmh.predef_col ) stream1 << "\tpredefined collimator number: " << wmh.COL.num << endl;
		else stream1 << "\tcollimator parameters from: " << wmh.col_fn << endl; 
		
		if ( wmh.COL.do_fb ) stream1 << "collimator geometry: fanbeam " << endl;
		else stream1 << "collimator geometry: parallel" << endl;
	}
	else{
		if ( wmh.COL.num == 0 ) stream1 << "collimator geometry: parallel " << endl;
		else stream1 << "collimator geometry: fanbeam with focal distance : " << wmh.COL.F << endl;
	}
	
	stream1 << "attenuation correction: " << wmh.do_att << endl;
	if ( wmh.do_att ){
		if ( wmh.do_full_att ) stream1 << "\t mode: full " << endl;
		else stream1 << "\t mode: simple " << endl;
	}
			
	//......... masking ....................................
	
	stream1 << "masking: " << wmh.do_msk << endl;
	if ( wmh.do_msk ){
		if ( wmh.do_msk_cyl ) stream1 << "\tmask type: cyl" << endl; 
		if ( wmh.do_msk_att ) stream1 << "\tmask type: att" << endl; 
		if ( wmh.do_msk_file ){
			stream1 << "\tmask type: file" << endl; 
			stream1	<< "\tmask file name: " << wmh.msk_fn << endl;
			}
		if ( wmh.do_msk_slc	){
			stream1 << "first slice: " << wmh.vol.first_sl << endl; 
			stream1	<< "last slice: " << wmh.vol.last_sl << endl;
		}
		
	}
	stream1.close();
}

//=============================================================================
//=== write_wm_STIR ===========================================================
//=============================================================================

void write_wm_STIR()
{
	int seg_num = 0;             // segment number for STIR matrix (always zero)
	FILE *fid;
	
	if ( ( fid = fopen( wm.OSfn.c_str() , "wb" )) == NULL ) error_wmtools_SPECT( 31, wm.OSfn );
	
	//...loop for matrix elements: projection index ..................
	
	for( int j = 0 ; j < wm.NbOS ; j++ ){
		
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

//=============================================================================
//=== index_calc ==============================================================
//=============================================================================

void index_calc ( int *indexs )
{
	if ( wmh.prj.NOS == 1 ){
		for ( int i = 0 ; i < wmh.prj.Nang ; i++ ){
			indexs[ i ] = i;           // when one single matrix, sequential order
		}
	}
	else{
		int j, *ple, *iOS, *a, *sa, *dif ;
		
		iOS = new int [ wmh.prj.NOS ];
		ple = new int [ wmh.prj.NOS ];
		a   = new int [ wmh.prj.NOS ];
		sa  = new int [ wmh.prj.NOS ];
		dif = new int [ wmh.prj.NOS + 1 ];
		
		//... to initialize variables ..................................
		
		for ( int i = 0 ; i < wmh.prj.NOS ; i++ ){
			iOS[ i ] = ple[ i ] = a[ i ] = sa[ i ] = dif[ i ] = 0;
		}
		dif[ wmh.prj.NOS ] = 0;
		ple[ 0 ]  = 1;
		
		//... to fill differences vector .................................
		
		int OS2 = wmh.prj.NOS * wmh.prj.NOS ;
		
		for ( int i = 1 ; i <= wmh.prj.NOS ; i++ ){
			dif[ i ] += 2 * ( i * ( i - wmh.prj.NOS ) ) + OS2 ;
		}
		
		//... first angle for each subset: angle having a maximum distance with all precedent angles ...
		
		int im = 0;     // first index is always set to zero
		
		for ( int k = 1 ; k < wmh.prj.NOS ; k++ ){
			
			for ( int i = 1 ; i < wmh.prj.NOS ; i++ ){
				if( !ple[ i ] ){
					j = i - im;
					a[ i ] = dif[ abs( j ) ];
				}
			}
			
			for( int i = 0 ; i < wmh.prj.NOS ; i++ ){
				a[ i ]  *= ( 1 - ple[ i ] );
				sa[ i ] *= ( 1 - ple[ i ] );
				sa[ i ] += a[ i ];
			}
			
			int m = 0;
			int n = 0;
			
			for ( int i = 0 ; i < wmh.prj.NOS ; i++ ){
				m = maxim( m, sa[ i ] );
			}
			
			for( int i = 1 ; i < wmh.prj.NOS ; i++ ){
				if( !ple[ i ] ){
					m = minim( m, sa[ i ] );
				}
			}
			
			for ( int i = 1 ; i < wmh.prj.NOS ; i++ ){
				if( sa[ i ] == m ){
					n = maxim( n, a[i] );
				}
			}
			for ( int i = wmh.prj.NOS - 1 ; i > 0 ; i-- ){
				if( sa[ i ] == m ){
					if( a[ i ] <= n ){
						n = a[ i ];
						im = i;
					}
				}
			}				
			iOS[ k ]  = im;
			ple[ im ] = 1;			
		}
		
		//... to fill the rest of angles of each subset ................
		
		for( int i = 0 ; i < wmh.prj.NOS ; i++ ){
			
			for( int j = 0 ; j < wmh.prj.NangOS ; j++ ){
				
				indexs[ i * wmh.prj.NangOS + j ] = iOS[ i ] + wmh.prj.NOS * j;
			}
		}
		
		delete [] iOS;
		delete [] a;
		delete [] sa;
		delete [] dif;
		delete [] ple;
	}
	
}

//=============================================================================
//=== read rotation radius ==================================================
//=============================================================================

void read_Rrad()
{
	string line;
	ifstream stream1( wmh.Rrad_fn.c_str() );
	if( !stream1 ) error_wmtools_SPECT( 114, wmh.Rrad_fn );
	
	int i = 0;
	
	while ( !stream1.eof() ){
		getline ( stream1, line );
		Rrad[ i ] = atof ( line.c_str() );
		i++;
	}
	
	if ( i != wmh.prj.Nang ) error_wmtools_SPECT( 11, wmh.Rrad_fn ); 
	stream1.close();
	
	return;
}

//=============================================================================
//=== col params ==============================================================
//=============================================================================

//void col_params( collim_type *COL )
//{
//	cout << "Using collimator: " <<  COL->num << endl; 
//	
//	switch(COL->num){
//			
//		case 1:  //...................fanbeam: ELSCINT
//			COL->F     = (float)35.5;
//			COL->L     = (float)4.;
//			COL->A_h   = (float)0.3369;   
//			COL->A_v   = (float)0.3369; 
//			COL->D     = (float)0.8;
//			COL->w     = (float)0.0866;
//			COL->insgm = (float)0.17;
//			COL->do_fb = true;
//			break;
//			
//		case 2: //....................fanbeam: ELSCINT D=0
//			COL->F     = (float)35.5;
//			COL->L     = (float)4.;
//			COL->A_h   = (float)0.3369; 
//			COL->A_v   = (float)0.3369; 
//			COL->D     = (float)0.;
//			COL->w     = (float)0.0866;
//			COL->insgm = (float)0.17;
//			COL->do_fb = true;
//			break;
//			
//		case 3: //....................parallel 3: low resolution
//			COL->A     = (float)0.0275;
//			COL->B     = (float)0.2;
//			COL->do_fb = false;
//			break;
//			
//		case 4: //....................parallel 4: high resolution
//			COL->A     = (float)0.0172;
//			COL->B     = (float)0.2;
//			COL->do_fb = false;
//			break;
//			
//		case 5: //....................parallel 5: (ECAM)
//			COL->A     = (float)0.0167;
//			COL->B     = (float)0.1405;
//			COL->do_fb = false;
//			break;
//			
//		case 6: //....................fan_beam: prism3000
//			COL->F     = (float)65.0;
//			COL->L     = (float)2.7;
//			COL->A_h   = (float)0.3575; 
//			COL->A_v   = (float)0.3360; 
//			COL->D     = (float)0.0;
//			COL->w     = (float)0.0866;
//			COL->insgm = (float)0.17;
//			COL->do_fb = true;
//			break;
//			
//		case 10: //...................parallel: ECAM with L=40 mm
//			COL->A     = (float)0.0101;
//			COL->B     = (float)0.0998;
//			COL->do_fb = false;
//			break;
//			
//		case 11: //...................fan beam ELSCINT L=2,405 cm 
//			COL->F     = (float)35.5;
//			COL->L     = (float)2.405;
//			COL->A_h   = (float)0.3369; 
//			COL->A_v   = (float)0.3369; 
//			COL->D     = (float)0.8;
//			COL->w     = (float)0.0866;
//			COL->insgm = (float)0.17;
//			COL->do_fb = true;
//			break; 
//			
//		case 13: //...................parallel: Hammamatsu collimator
//			COL->A     = (float)0.0205;
//			COL->B     = (float)0.10245;
//			COL->do_fb = false;
//			break;
//			
//		case 14: //...................parallel. hexagonal holes. apotema=0.57mm, L=24mm, s=0.125mm
//			COL->A     = (float)0.0178;     
//			COL->B     = (float)0.0886;
//			COL->do_fb = false;
//			break;
//			
//		case 15: //...................parallel. hexagonal holes apotema=0.57mm, L=24mm, s=0.125mm
//			COL->A     = (float)0.0247;     
//			COL->B     = (float)0.0752;
//			COL->do_fb = false;
//			break;
//			
//		case 16: //...................parallel: Sentinella S102 colimador: experimental parameters
//			COL->A     = (float)0.0166;     
//			COL->B     = (float)0.0924;
//			COL->do_fb = false;
//			break;
//			
//		case 17: //...................parallel Infinia Hawkeye: experimental parametres
//			COL->A     = (float)0.0163;     
//			COL->B     = (float)0.1466;
//			COL->do_fb = false;
//			break;
//			
//		default:
//			char p[3];	// auxiliar variable for itoa
//			error_wmtools_SPECT( 21, itoa(COL->num,p));
//	}
//}

///=============================================================================
//=== read collimator params ==================================================
//=============================================================================

//void read_col_params( collim_type *COL )
//{
//	string line;
//	ifstream stream1( wmh.col_fn.c_str() );
//	if( !stream1 ) error_wmtools_SPECT( 122, wmh.col_fn );	
//	
//	getline ( stream1, line );
//	
//	if ( line[ 0 ] == 'f' )  COL->do_fb = true;
//	else{
//		if ( line[ 0 ] == 'p' )  COL->do_fb = false;
//		else error_wmtools_SPECT( 12, wmh.col_fn);
//	}
//
//	if ( COL->do_fb ){
//				
//		getline ( stream1, line );
//		COL->F = atof( line.c_str() );     // Focal length (cm)
//				
//		getline ( stream1, line );
//		COL->L = atof( line.c_str() );     // collimator to detector distance (? cm)
//		
//		getline ( stream1, line );
//		COL->A_h = atof( line.c_str() );   // linear factor for dependency of sigma on distance (fanbeam horizontal)
//		
//		getline ( stream1, line );
//		COL->A_v = atof( line.c_str() );   // linear factor for dependency of sigma on distance (fanbeam vertical)
//		
//		getline ( stream1, line );
//		COL->D = atof( line.c_str() );     // (?)
//		
//		getline ( stream1, line );
//		COL->w = atof( line.c_str() );     // collimator thickness (? cm)
//	
//		if( !stream1.eof() ) error_wmtools_SPECT( 13, wmh.col_fn );
//		
//		getline ( stream1, line );
//		COL->insgm = atof( line.c_str() ); // intrinsic sigma (cristal resolution cm?)
//	
//	}
//	else{
//		getline ( stream1, line );
//		COL->A = atof( line.c_str() );          // linear factor for dependency of sigma on distance (parallel): sigma=A*dist+B 
//		
//		if( !stream1.eof() ) error_wmtools_SPECT( 13, wmh.col_fn	);
//		
//		getline ( stream1, line );
//		COL->B = atof( line.c_str() );      // Independent factor for dependency of sigma on distance: sigma=A*dist+B	
//	}	
//	
//	stream1.close();
//	return;
//}


//==========================================================================
//=== calc_sigma_v =========================================================
//==========================================================================

float calc_sigma_v( voxel_type	vox, collim_type COL)
{
	float sigma;
	if ( COL.do_fb ){
		float xc    = (float)2. * COL.A_v * COL.w * ( vox.dv2dp + COL.L + COL.D ) / COL.L;
		sigma = sqrt( COL.insgm * COL.insgm + xc * xc );
		
	}
	else sigma = COL.A * vox.dv2dp + COL.B ;
	
	return( sigma );
}


//=============================================================================
//=== fill_ang ================================================================
//=============================================================================

void fill_ang ( angle_type *ang )
{
	float DX    = (float) 0.5 / wmh.psfres ;
	float dg2rd = (float)M_PI / (float)180. ;
	
	for ( int i = 0; i < wmh.prj.Nang ; i++ ){
		
		//... ratios calculation .......................................................
		
		float deg = wmh.prj.ang0 + (float)i * wmh.prj.incr ;	// angle in degrees
		ang[ i ].cos = cos( deg * dg2rd );						// cosinus of the angle
		ang[ i ].sin = sin( deg * dg2rd );						// sinus of the angle
		
		//... first octane (0->45º) equivalent angle and its trigonometric ratios .......
		
		float angR = fabs( deg );
		int   quad = (int) floor( angR / (float)90. );			 // quadrant 
		
		angR = fabs( angR - (float)90. * (float)quad );			 // reduced angle: equivalent angle in 0->45º interval
		if ( angR > (float)45. ) angR = fabs( (float)90. - angR );   
	
		float sinR = (float)sin( angR * dg2rd );		// sinus of the reduced angle
		float cosR = (float)cos( angR * dg2rd );		// cosinus of the reduced angle
		
		//... parametres of the oblique projection of a square voxel size 1 (half a trapezoid) .......
		
		if ( !wmh.do_psf ){
			
			if ( angR < EPSILON ){
				
				ang[ i ].p  = (float)1. ;
				ang[ i ].N1 = ang[ i ].N2 = (int) floor( DX );
				ang[ i ].m  = ang[ i ].n  = (float)0.;
			}
			else{
				ang[ i ].p  = (float)1. / cosR;                               // plateau highness
				ang[ i ].m  = -wmh.psfres / ( sinR * cosR );                  // slope of the trapezoid in DX units (negative)
				ang[ i ].n  = ( cosR + sinR ) * (float)0.5 / ( cosR * sinR ); // independent term of the slope of the trapezoid (cm)
				ang[ i ].N1 = (int) floor( (float) fabs( cosR - sinR ) * DX ); // index of the first vertice (end of plateau) in res units
				ang[ i ].N2 = (int) floor( ( cosR + sinR ) * DX );             // index of the second vertice (end of the slope) in res units
			}
			
			ang[ i ].vxprj.lngd2 = ang[ i ].N2;
			ang[ i ].vxprj.lng   = 2 * ang[ i ].N2;
			ang[ i ].vxprj.res   = wmh.psfres;
		}
		//... rotation radius ................................................................
		
		ang[ i ].Rrad = Rrad[ i ];              // assignation of (variable) rotation radius
		
		//... coordinates of the first bin of each projection and increments for consecutive bins ....
		
		if(wmh.do_att){
			
			ang[ i ].incx  = wmh.prj.szcm * ang[ i ].cos;
			ang[ i ].incy  = wmh.prj.szcm * ang[ i ].sin;
			
			ang[ i ].xbin0 = -ang[ i ].Rrad * ang[ i ].sin - wmh.prj.lngcmd2 * ang[ i ].cos ; 
			ang[ i ].ybin0 =  ang[ i ].Rrad * ang[ i ].cos - wmh.prj.lngcmd2 * ang[ i ].sin ;
		}
	}
}

//=============================================================================
//=== generate msk ============================================================
//=============================================================================

void generate_msk ( bool *msk_3d, bool *msk_2d, float *attmap, volume_type * vol )
{
	//... initialzation of msk to true .........................
	
	for ( int i = 0 ; i < vol->Nvox ; i++ ){
		msk_3d[ i ] = true;											  
	}
	
	//... initialzation of msk_2d to false .....................
	
	for ( int i = 0 ; i < vol->Npix ; i++ ){
		msk_2d[ i ] = false;											  
	}
	
	//... to create mask from attenuation map ..................
	
	if ( wmh.do_msk_att ){                                   
		for ( int i = 0 ; i < wmh.vol.Nvox ; i++ ){
			msk_3d[ i ] = ( attmap[ i ] > EPSILON );                
		}
	}
	else {
		//... to create a cylindrical mask...................... 
		
		if (wmh.do_msk_cyl){								      
			
			float Rmax2,xi,yi;
			
			if ( vol->Nrow >= vol->Ncol ) Rmax2 = vol->Nrowd2 * vol->Nrowd2;   // Maximum allowed radius (distance from volume centre)  
			else Rmax2 = vol->Ncold2 * vol->Ncold2;
			
			int ip = -1;                                                       // in-plane index of the voxel    
			
			for ( int i = 0 ; i < vol->Ncol ; i++ ){
				
				xi = i - vol->Ncold2 + (float)0.5 ;
				xi *= xi;
				
				for ( int j=0 ; j < vol->Nrow ; j++ ){
					
					ip++;
					yi  = j - vol->Nrowd2 + (float)0.5 ;
					yi *= yi;
					
					if ( ( xi + yi ) > Rmax2 ){
						
						for ( int k = 0 ; k < vol->Nsli ; k ++){   
							
							 msk_3d[ ip + k * vol->Npix ] = false;              // loop for all the slices
						}
					}
				}
			}
		}
		
		else {
			//... to read a mask from a (int) file ....................
			
			if ( wmh.do_msk_file ) read_msk_file( msk_3d );             
		}
	}

	
	//... to apply slice mask (to remove slices from matrix) ..............
	
	if ( wmh.do_msk_slc ){
		for ( int i = 0 ; i < wmh.vol.first_sl ; i++ ){
			for ( int j = 0 ; j < wmh.vol.Npix ; j++ ) {
				msk_3d[ i * wmh.vol.Npix + j ] = false;
			}
		}
		for ( int i = wmh.vol.last_sl ; i < wmh.vol.Nsli ; i++ ){
			for ( int j = 0 ; j < wmh.vol.Npix ; j++ ) {
				msk_3d[ i * wmh.vol.Npix + j ] = false;
			}
		}
	}
	
	//... to collapse mask to 2d_mask .........
	
	if ( wmh.do_msk_cyl ){
		for ( int i = 0 ; i < wmh.vol.Npix ; i++ ){
			msk_2d[ i ] = msk_3d[ i + wmh.vol.first_sl * wmh.vol.Npix  ];
		}
	}
	else{
		for ( int i = 0 ; i < wmh.vol.Npix ; i++ ){
			
			for ( int k = wmh.vol.first_sl ; k < wmh.vol.last_sl ; k++ ){
			    
				if ( msk_3d[ k * wmh.vol.Npix + i ] ){
					msk_2d[ i ] = true;
					break;
				}
			}
		}
	}
}

//=============================================================================
//=== read_mask file ==========================================================
//=============================================================================

void read_msk_file( bool *msk )
{
	FILE *fid;
	int *aux;
	
	aux = new int [ wmh.vol.Nvox ];
	
	if ( (fid = fopen( wmh.msk_fn.c_str() , "rb")) == NULL) error_wmtools_SPECT( 126, wmh.msk_fn);
	fread( aux, sizeof(int), wmh.vol.Nvox, fid);
	fclose(fid);
	
	for (int i = 0 ; i < wmh.vol.Nvox ; i ++ ){
		msk[i] = ( aux[i] != 0 );
	}
	
	delete [] aux;
}

//=============================================================================
//=== read_att_map ============================================================
//=============================================================================

void read_att_map( float *attmap )
{
	FILE *fid;
	if ( ( fid = fopen( wmh.att_fn.c_str() , "rb") ) == NULL ) error_wmtools_SPECT ( 124, wmh.att_fn );
	fread( attmap, sizeof(float), wmh.vol.Nvox, fid);
	
	bool exist_nan = false;
	
	for (int i = 0 ; i < wmh.vol.Nvox ; i++ ){
		if ((boost::math::isnan)(attmap [ i ])){
			attmap [ i ] = 0;
			exist_nan = true;
		}
	}
	
	if ( exist_nan ) cout << "WARNING: att map contains NaN values. Converted to zero" << endl;
	
	fclose(fid);
}

//==========================================================================
//=== max_psf_szb ==========================================================
//==========================================================================

int max_psf_szb( angle_type *ang )
{ 
	int maxszb;
	float Rrad_max = ang[0].Rrad;
	
	for( int i = 1; i < wmh.prj.Nang ; i++ ){
		if ( ang[ i ].Rrad > Rrad_max ) Rrad_max = ang[ i ].Rrad;  // maximum rotation radius 
	}
	
	if ( !wmh.do_psf ){                                                                         // NO-PSF                            
		
		if ( !wmh.COL.do_fb ){																	// parallel			
			maxszb = (int)( (float) sqrt( (float)2. ) * wmh.vol.szcm / wmh.prj.szcm ) + 3; 
		}
		
		else{																					// fanbeam
			float dpmax = wmh.vol.szcm * maxim( wmh.vol.Ncold2, wmh.vol.Nrowd2) + Rrad_max;
			
			float lon = wmh.COL.F - dpmax;
			if ( lon < EPSILON ) error_wmtools_SPECT( 46, "");
			
			//... maximum lenght of psf in bins ........................
			
			float f = (int)( (float) sqrt( (float)2. ) * (wmh.vol.szcm / wmh.prj.szcm) * ( wmh.COL.F / lon ) ) + 3;
			maxszb = minim ( f , wmh.prj.Nbin );
		}
	}
	else{																						// PSF
		voxel_type vox;
		
		if ( wmh.COL.do_fb ){
			vox.costhe = (float)1. / sqrt( wmh.prj.lngcmd2 * wmh.prj.lngcmd2 / ( wmh.COL.F * wmh.COL.F ) + (float)1.);	
		}
		//... maximum length of psf in bins ........................
		
		vox.dv2dp = Rrad_max + wmh.vol.szcm * maxim( wmh.vol.Ncold2, wmh.vol.Nrowd2 ) * (float)1.5;
		float sig_h_max_cm = calc_sigma_h( vox, wmh.COL );
		maxszb = (int)floor( wmh.maxsigm * (float)2. * sig_h_max_cm / wmh.prj.szcm ) + 3;
	
		if ( wmh.do_psf_3d ){
			float sig_v_max_cm = calc_sigma_v( vox, wmh.COL );
			int maxszb_v = (int)floor( wmh.maxsigm * (float)2. * sig_v_max_cm / wmh.prj.thcm ) + 3;
			maxszb = maxim( maxszb , maxszb_v ); 	
		}
	}

	return( maxszb );
}

//==========================================================================
//=== calc_sigma_h =========================================================
//==========================================================================

float calc_sigma_h( voxel_type vox, collim_type COL )
{
	float sigma;
	
	if ( COL.do_fb ){
		float denom = sqrt( COL.L * COL.L * (COL.F - vox.dv2dp) * (COL.F - vox.dv2dp) - COL.w * COL.w * (COL.L + (float)2. * vox.dv2dp) * (COL.L + (float)2. * vox.dv2dp));
		float xc    = COL.A_h * (vox.dv2dp + COL.L + COL.D) * COL.w * ( (float)2. * COL.F + COL.L) / (vox.costhe * denom);
		sigma       = sqrt( COL.insgm * COL.insgm + xc * xc );
	}
	else sigma = COL.A * vox.dv2dp + COL.B ;
	
	return( sigma );
}

//=============================================================================
//=== itoa ====================================================================
//=============================================================================

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

//=============================================================================
//=== free_wm =================================================================
//=============================================================================

void free_wm( wm_type *f )
{
	delete [] f->ar;
	delete [] f->ja;
	delete [] f->ia;
}

//=============================================================================
//=== free_wm_da ==============================================================
//=============================================================================

void free_wm_da( wm_da_type *f )
{
	for(int i=0; i< f->NbOS; i++){
		delete [] f->val[i];
		delete [] f->col[i];
	}
	delete [] f->val;
	delete [] f->col;
	delete [] f->ne;
	
	if ( f->do_save_STIR ){
		delete [] f->nb;
		delete [] f->ns;
		delete [] f->na;
		delete [] f->nx;
		delete [] f->ny;
		delete [] f->nz;
	}
}

//=============================================================================
//== error_wmtools_SPECT ======================================================
//=============================================================================

void error_wmtools_SPECT( int nerr, string txt )
{
#if 0
  switch(nerr){
  case 11: printf("\n\nError wm_SPECT: number of variable rotation radius in file: %s different from number of angles\n",txt.c_str());break;
  case 12: printf("\n\nError wm_SPECT: first parameter in collimator file should be 'p' or 'f' to indicate parallel or fanbeam collimator\n");break;
  case 13: printf("\n\nError wm_SPECT: not enough parameters in collimator file: %s \n",txt.c_str());break;
  case 21: printf("\n\nError wmtools_SPECT: undefined collimator. Collimator %s not found\n",txt.c_str()); break;
  case 30: printf("\n\nError wmtools_SPECT: can not open %s for reading\n",txt.c_str()); break;
  case 31: printf("\n\nError wmtools_SPECT: can not open %s for writing\n",txt.c_str()); break;
  case 46: printf( "\n\nError weight3d: there are voxels near or further than de FOCAL lenght\n"); break;
  case 50: printf("\n\nError wmtools_SPECT: No header stored in %s \n",txt.c_str()); break;
			
    //... error: value of argv[]..........................
		
  case 114: printf("\n\nError wm_SPECT: file with variable rotation radius: %s not found\n",txt.c_str());break;
  case 122: printf("\n\nError wm_SPECT: file with variable collimator parameters: %s not found\n",txt.c_str());break;
  case 124: printf("\n\nError wm_SPECT: can not open attenuation map-> argv[24]: %s for reading\n",txt.c_str()); break;
  case 126: printf("\n\nError wm_SPECT: can not open file mask-> argv[26]: %s for reading\n",txt.c_str()); break;
			
  default: printf("\n\nError wmtools_SPECT: unknown error number on error_wmtools_SPECT()"); 
  }
	
  exit(0);
#else
  using stir::error;
  switch(nerr){
  case 11: printf("\n\nError wm_SPECT: number of variable rotation radius in file: %s different from number of angles\n",txt.c_str());break;
  case 12: error("\n\nError wm_SPECT: first parameter in collimator file should be 'p' or 'f' to indicate parallel or fanbeam collimator\n");break;
  case 13: printf("\n\nError wm_SPECT: not enough parameters in collimator file: %s \n",txt.c_str());break;
  case 21: printf("\n\nError wmtools_SPECT: undefined collimator. Collimator %s not found\n",txt.c_str()); break;
  case 30: printf("\n\nError wmtools_SPECT: can not open %s for reading\n",txt.c_str()); break;
  case 31: printf("\n\nError wmtools_SPECT: can not open %s for writing\n",txt.c_str()); break;
  case 46: error( "\n\nError weight3d: there are voxels near or further than de FOCAL lenght\n"); break;
  case 50: printf("\n\nError wmtools_SPECT: No header stored in %s \n",txt.c_str()); break;
			
    //... error: value of argv[]..........................
		
  case 114: printf("\n\nError wm_SPECT: file with variable rotation radius: %s not found\n",txt.c_str());break;
  case 122: printf("\n\nError wm_SPECT: file with variable collimator parameters: %s not found\n",txt.c_str());break;
  case 124: printf("\n\nError wm_SPECT: can not open attenuation map-> argv[24]: %s for reading\n",txt.c_str()); break;
  case 126: printf("\n\nError wm_SPECT: can not open file mask-> argv[26]: %s for reading\n",txt.c_str()); break;
			
  default: error("\n\nError wmtools_SPECT: unknown error number on error_wmtools_SPECT()"); 
  }
	
#endif
}    

#if 0
void error_wm_SPECT( int nerr, string txt)
{
	string opcions[]={
		"\nargv[1]  Matrix file: Weight matrix filename (without extension index)",
				
		"\nargv[2]  Image box: Number of columns (int)",
		"\nargv[3]  Image box: Number of rows (int)",
		"\nargv[4]  Image box: Number of slices (the same than projection slices) (int)",
		"\nargv[5]  Image box: Voxel side length(cm). Only square voxels are considered (float cm)",
		"\nargv[6]  Image box: Slice thickness (the same than projection slice thickness) (float cm)",
		
		"\nargv[7]  Projections: Number of bins in a single projection (1d) (int)",
		"\nargv[8]  Projections: Bin size (float cm)",
		"\nargv[9]  Projections: Number of angles (int)",
		"\nargv[10] Projections: Initial angle (float degrees)",
		"\nargv[11] Projections: Angle increment between consecutive projections (float degrees)",
		
		"\nargv[12] Matrix: Number of subsets in which to split matrix (1= all the matrix in one single file)",
		"\nargv[13] Matrix: Fixed projection radius (yes/no)",
		"\nargv[14] Matrix: File containing rotation radius/ Rrad value (float: cm) from center to collimator plane",
		"\nargv[15] Matrix: Minimum weight to take into account (over 1. Typically 0.01)",
		"\nargv[16] Matrix: Maximum number of sigmas to consider in PSF calculation (float)",
		"\nargv[17] Matrix: Spatial resolution in which to sample PSF distributions (typically 0.0001)",
		"\nargv[18] Matrix: First slice to reconstruct (1 to Nslices)",
		"\nargv[19] Matrix: Last slice to reconstruct (1 to Nslices)",
		
		"\nargv[20] Matrix: Correction for PSF (2d/3d/no)",
		"\nargv[21] Matrix: Predefined collimator. If no correction for PSF set to yes (yes/no)",
		"\nargv[22] Matrix: File with the collimator parameters/predefined collimator number (see help for options)\n\tIf no correction for PSF, argv[22]= 0 (int) (parallel) or argv[22]= focal distance (float, cm) (fanbeam)", 
		
		"\nargv[23] Matrix: Correction for attenuation (simple/full/no)",
		"\nargv[24] Matrix: attenuation map (filename/no) (in case of explicit mask)",
		"\nargv[25] Matrix: volume masking (cyl/att/file/no). cyl: inscrit cylinder, att: no weight calculated where att=0",
		"\nargv[26] Matrix: explicit mask (filename/no) (in case of explicit mask)",
		
		"\nargv[27]  Matrix file: Format. Options: STIR, FC (FruitCake)",
		"\nargv[28]  Matrix file: to save matrix header as an a part file (yes/no)"
	};
	
	 string predef_collim[]={
		"(1)  fan_beam_elscint\n",
		"(2)  fan_beam_elscint_simset\n",
		"(3)  paralel3_low resolution\n",
		"(4)  paralel4_high resolution\n",
		"(5)  paralel5_LEHR_siemens_ecam\n",
		"(6)  fan_beam_prism3000\n",
		"(10) parallel ECAM_L4cm\n",
		"(11) fanbeam_elscint_L24.05cm\n",
		"(13) parallel square holes collimator\n",
		"(14) parallel hexagonal holes collimator\n",
		"(15) parallel hexagonal holes collimator 2\n",
		"(16) parallel Sentinella_collimator\n",
		"(17) parallel Infinia_Hawkeye\n"
	};
	int NUMCOL = 13;
	
	switch(nerr){
		case 100: cout << endl << "Missing variables" << endl;
			for ( int i = 0 ; i < NUMARG-1 ; i++ ){
				printf( "%s\n", opcions[ i ].c_str() );						
			}
			printf("\n\nPredefined collimator options:\n");
			for ( int i = 0 ; i < NUMCOL ; i++ ){
				printf( "\t%s\n", predef_collim[ i ].c_str() );
			}
			break;
		
		//... error: value of argv[]..........................
		
		case 101: printf("\n\nError wm_SPECT: parametre file: %s not found\n",txt.c_str());break;
		case 102: printf("\n\nError wm_SPECT: More parametres tan expected in file: %s\n",txt.c_str());break;
		case 103: printf("\n\nError wm_SPECT: Less parametres tan expected in file: %s\n",txt.c_str());break;
		case 112: printf("\n\nError wm_SPECT: number of subsets should be congruent with number of projection angles\n"); break;		
		case 113: printf("\n\nError wm_SPECT: invalid option for argv[13]. Options: yes/no. Read value: %s \n",txt.c_str());break;
		case 114: printf("\n\nError wm_SPECT: file with variable rotation radius: %s not found\n",txt.c_str());break;
		case 118: printf("\n\nError wm_SPECT: first slice to reconstruct out of range (1->Nslic): %s \n",txt.c_str());break;
		case 119: printf("\n\nError wm_SPECT: last slice to reconstruct out of range (first slice->Nslic): %s \n",txt.c_str());break;
		case 120: printf("\n\nError wm_SPECT: invalid option for argv[20]. Options: no/2d/3d. Read value: %s \n",txt.c_str());break;
		case 121: printf("\n\nError wm_SPECT: invalid option for argv[21]. Options: yes/no. Read value: %s \n",txt.c_str());break;
		case 122: printf("\n\nError wm_SPECT: file with variable collimator parameters: %s not found\n",txt.c_str());break;
		case 123: printf("\n\nError wm_SPECT: invalid option for argv[23]. Options: no/simple/full. Read value: %s \n",txt.c_str());break;
		case 124: printf("\n\nError wm_SPECT: can not open attenuation map-> argv[24]: %s for reading\n",txt.c_str()); break;
		case 125: printf("\n\nError wm_SPECT: invalid option for argv[25]. Options: no/cyl/att/file. Read value: %s \n",txt.c_str());break;
		case 126: printf("\n\nError wm_SPECT: can not open file mask-> argv[26]: %s for reading\n",txt.c_str()); break;
		case 127: printf("\n\nError wm_SPECT: invalid option for argv[27]. Options: STIR/FC. Read value: %s \n",txt.c_str());break;
		case 128: printf("\n\nError wm_SPECT: invalid option for argv[28]. Options: yes/no. Read value: %s \n",txt.c_str());break;
			
		//... other errors...........................................................
			
		case 150: printf("\n\nError wm_SPECT: second delimiter missing in file of parameters. Param: %s", txt.c_str() ); break;
		case 200: printf("\n\nError wm_SPECT: cannot allocate the variable: %s\n",txt.c_str());break;		

		default: printf("\n\nError wm_SPECT: wrong error number");
	}
	
	exit(0);
}
#endif

} // namespace SPECTUB

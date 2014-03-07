/*
 * Copyright (c) 2014, 
 * Institute of Nuclear Medicine, University College of London Hospital, UCL, London, UK.
 * Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. All rights reserved.
 * This software is distributed WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 
 \author Carles Falcon
 Please, report bugs to cfalcon@ub.edu
 */

//... system libraries ...............................................................

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

using namespace std;
using std::string;

//... user defined libraries .............................................................

#include "wmtools_SPECT_mph.h"
#include "weight3d_SPECT_mph.h"

//... functions from wm_SPECT.2.0............................

void error_wm_SPECT_mph( int nerr, string txt);      //list of error messages
void wm_inputs_mph( char ** argv, int argc );
void read_inputs_mph( vector<string> param );

//.. global variables ....................................................................

wmh_mph_type wmh;       // weight matrix header. Global variable
wm_da_type wm;          // double array weight matrix structure. Global variable
pcf_type pcf;           // pre-calculated functions

//==========================================================================
//=== main =================================================================
//==========================================================================

int main( int argc, char **argv)
{
    bool  *msk_3d;        // voxels to be included in matrix (no weight calculated outside the mask)
    float *attmap;        // attenuation map
	
    psf2d_type psf_bin;   // structure for total psf distribution in bins (dibimensional)
    psf2d_type psf_subs;  // structure for total psf distribution: mid resolution (dibimensional)
    psf2d_type psf_aux;   // structure for total psf distribution: mid resolution auxiliar for convolution (bimensional)
    psf2d_type kern;      // structure for intrinsic psf distribution: mid resolution (dibimensional)
    
    string header_suffix = ".wmhdr";  // suffix to add to matrix filename for weight matrix header filename
    
    double ini = clock();
	
	//... to read parameters and to calculate derivated variables ..........
	
	wm_inputs_mph( argv, argc );
	
	//... to read attenuation map ..................................................
	
	if ( wmh.do_att ){
		
        attmap = new float [ wmh.vol.Nvox ];
		read_att_map_mph( attmap );
	}
	else attmap = NULL;
	
	//... to generate mask..........................................................

    msk_3d = new bool [ wmh.vol.Nvox ];
    
    generate_msk_mph( msk_3d, attmap );
    
    //... initialize psf2d in bins ..................................................
    
  	wmh.max_amp =  ( wmh.prj.rad - wmh.ro ) / ( wmh.collim.rad - wmh.ro );
    
    psf_bin.max_dimx = (int) floorf ( wmh.max_hsxcm * wmh.max_amp / wmh.prj.szcm ) + 2 ;
    psf_bin.max_dimz = (int) floorf ( wmh.max_hszcm * wmh.max_amp / wmh.prj.thcm ) + 2 ;
    
    //... distributions at mid resolution ...........................................
    
    if ( wmh.do_subsamp ){
        
        psf_subs.max_dimx = psf_bin.max_dimx * wmh.subsamp ;
        psf_subs.max_dimz = psf_bin.max_dimz * wmh.subsamp ;
        
        if ( wmh.do_depth ){
            
            psf_subs.max_dimx += ( 1 + (int) ceilf( wmh.prj.crth *  wmh.tmax_aix / wmh.prj.szcm ) ) * wmh.subsamp ;
            psf_subs.max_dimz += ( 1 + (int) ceilf( wmh.prj.crth *  wmh.tmax_aiz / wmh.prj.thcm ) ) * wmh.subsamp ;
        }
        
        if ( wmh.do_psfi ){
            
            int dimx = (int) ceil( (float)0.5 * wmh.prj.sgm_i * wmh.Nsigm / wmh.prj.szcm ) ;
            int dimz = (int) ceil( (float)0.5 * wmh.prj.sgm_i * wmh.Nsigm / wmh.prj.thcm ) ;
            
            kern.dimx = kern.max_dimx = 2 * wmh.subsamp * dimx + 1 ;
            kern.dimz = kern.max_dimz = 2 * wmh.subsamp * dimz + 1 ;
            kern.ib0  = - dimx;
            kern.jb0  = - dimz;
            kern.lngxcmd2 =  kern.lngzcmd2 = wmh.prj.sgm_i * wmh.Nsigm / (float)2.;
            
            kern.val = new float * [ kern.max_dimz ];
            
            for ( int i = 0 ; i < kern.max_dimz ; i++ )
                kern.val[ i ] = new float [ kern.max_dimx ];
            
            fill_psfi ( &kern );
            
            psf_subs.max_dimx += kern.max_dimx - 1 ;
            psf_subs.max_dimz += kern.max_dimz - 1 ;
            
            psf_aux.max_dimx = psf_aux.dimx = psf_subs.max_dimx ;
            psf_aux.max_dimz = psf_aux.dimz = psf_subs.max_dimz ;

            psf_aux.val = new float * [ psf_aux.max_dimz ];
            
            for ( int i = 0 ; i < psf_aux.max_dimz ; i++ ) psf_aux.val[ i ] = new float [ psf_aux.max_dimx ];
        }
        
        psf_subs.val = new float * [ psf_subs.max_dimz ];
    
        for ( int i = 0 ; i < psf_subs.max_dimz ; i++ ) psf_subs.val[ i ] = new float [ psf_subs.max_dimx ];
        
        psf_bin.max_dimx = psf_subs.max_dimx / wmh.subsamp + 2 ;
        psf_bin.max_dimz = psf_subs.max_dimz / wmh.subsamp + 2 ;
    }
    
    psf_bin.val = new float * [ psf_bin.max_dimz ];
    
    for ( int i = 0 ; i < psf_bin.max_dimz ; i++ ) psf_bin.val[ i ] = new float [ psf_bin.max_dimx ];
    
    //... size estimation .........................................................
    
    int * Nitems;                                              // number of non-zero elements for each weight matrix row
    Nitems  = new int [ wmh.prj.Nbt ];
    
    for ( int i = 0 ; i < wmh.prj.Nbt ; i++ ) Nitems[ i ] = 1;     // Nitems initializated to one
    
    wm_calculation_mph ( false , &psf_bin, &psf_subs, &psf_aux, &kern, attmap, msk_3d, Nitems );  // size esmitation
    
    int ne = 0;
    
    for ( int i = 0 ; i < wmh.prj.Nbt ; i++ ) ne += Nitems[ i ];
    
    cout << "\nwm_SPECT. Size estimation done. time (s): " << double( clock() - ini ) / CLOCKS_PER_SEC <<endl;
    cout << "\ntotal number of non-zero weights: " << ne << endl;
    if ( wm.do_save_STIR ) cout << "estimated matrix size: " << (ne + 10* wmh.prj.Nbt)/104857.6  << " Mb\n" << endl;
    else cout << "estimated matrix size: " << ne/131072 << " Mb\n" << endl;
    
    //... wm_alloc ................................
    
    wm_alloc( Nitems );
    
    //... wm calculation ...........................
    
    wm_calculation_mph ( true, &psf_bin, &psf_subs, &psf_aux, &kern, attmap, msk_3d, Nitems );
    
    cout << "\nwm_SPECT. Weight matrix calculation done. time (s): " << double( clock()-ini )/CLOCKS_PER_SEC <<endl;
    
    //... to write the matrix into a file ..........................
    
    cout <<  "\nwriting weight matrix..." << endl;
    
    if ( wm.do_save_STIR ) write_wm_STIR_mph();
    
    else write_wm_FC_mph();
    
    //... to save matrix header .............................

    wm.fn_hdr = wm.fn + header_suffix ;
    
    write_wm_hdr_mph();
    
    //... freeing memory .............................................
    
    free_wm();
    
    free_pcf();
    
    for ( int i = 0 ; i < psf_bin.max_dimz ; i++ ) delete [] psf_bin.val[ i ];
    delete [] psf_bin.val;
    
    if ( wmh.do_subsamp ){
        for ( int i = 0 ; i < psf_subs.max_dimz ; i++ ) delete [] psf_subs.val[ i ];
        delete [] psf_subs.val;
    }
    
    if ( wmh.do_psfi ){
        for ( int i = 0 ; i < kern.max_dimz ; i++ ) delete [] kern.val[ i ];
        delete [] kern.val;
    }
    
    delete [] Nitems;
    
    if ( wmh.do_att ) delete [] attmap;
    
    delete [] msk_3d;
    
    cout<<"\nwm_SPECT done. Execution time (s): " << double( clock()-ini )/CLOCKS_PER_SEC << endl;
    
    return( 0 );
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%% associated functions: reading, setting up variables and error messages %%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//==========================================================================
//=== wm_inputs ============================================================
//==========================================================================

void wm_inputs_mph( char **argv, int argc )
{
    vector<string> param;
    string line;
    size_t pos1, pos2, pos3, pos4;
    
    param.push_back ("0");
    
    if ( argc == 2 ){		  // argv[1] is a file containing the parametres
        
        //... to open text file to get parametres ......................
        
        ifstream stream1( argv[ 1 ] );
        if( !stream1 ) error_wm_SPECT_mph( 101, argv[ 1 ]);
        
        //... to get values ............................................
        
        int i=0;
        
        while ( !stream1.eof() ){
            getline ( stream1, line );
            
            pos1 = line.find( DELIMITER1 );
            if ( pos1 == string::npos ) continue;
            i++;
            if (i >= NUMARG ) error_wm_SPECT_mph( 102, argv[ 1 ] );
            
            pos2 = line.find_first_not_of(" \t\f\v\n\r", pos1 + 1 );
            pos3 = line.find( DELIMITER2 );
            if ( pos3 == string::npos ) {
                char aux[ 3 ];
                error_wm_SPECT_mph( 150, itoa ( i, aux ) );
            }
            pos4 = line.find_first_of( " \t\f\v\n\r", pos2 );
            pos3 = minim( pos3, pos4 );
            
            param.push_back( line.substr( pos2 , pos3 - pos2 ) );
        }
        stream1.close();
        if ( i != NUMARG -1 ) error_wm_SPECT_mph( 103, argv[ 1 ] );
    }
    
    else{
        cout << "number of arg: " << argc << endl;
        for ( int i = 0 ; i < argc ; i++ ) cout << i << ": " << argv[ i ] << endl;
        
        if ( argc != NUMARG ) error_wm_SPECT_mph( 100, "" );
        for ( int i = 1 ; i < argc ; i++ ) param.push_back( argv[ i ] );
    }
    
    read_inputs_mph( param );
}

//==========================================================================
//=== read_inputs ==========================================================
//==========================================================================

void read_inputs_mph(vector<string> param)
{
    
    //.... matrix file .................
    
    wm.fn =param[ 1 ];
    
    //.....image parameters......................
    
    wmh.vol.Dimx = atoi( param[ 2 ].c_str() );  // Image: number of columns
    wmh.vol.Dimy = atoi( param[ 3 ].c_str() );  // Image: number of rows
    wmh.vol.Dimz = atoi( param[ 4 ].c_str() );  // Image: and projections: number of slices
    wmh.vol.szcm = atof( param[ 5 ].c_str() );  // Image: voxel size (cm)
    wmh.vol.thcm = atof( param[ 6 ].c_str() );  // Image: slice thickness (cm)
    
    wmh.vol.first_sl = atoi( param[ 7 ].c_str() ) - 1;   // Image: first slice to take into account (no weight bellow)
    wmh.vol.last_sl  = atoi( param[ 8 ].c_str() );       // Image: last slice to take into account (no weights above)
    
    if ( wmh.vol.first_sl < 0 || wmh.vol.first_sl > wmh.vol.Dimz ) error_wm_SPECT_mph( 107, param[ 7 ] );
    if ( wmh.vol.last_sl <= wmh.vol.first_sl || wmh.vol.last_sl > wmh.vol.Dimz ) error_wm_SPECT_mph( 108, param[ 8 ] );
    
    wmh.ro = atof( param[ 9 ].c_str() );         // Image: object radius (cm)
    
    //..... geometrical and other derived parameters of the volume structure...............
    
    wmh.vol.Npix    = wmh.vol.Dimx * wmh.vol.Dimy;
    wmh.vol.Nvox    = wmh.vol.Npix * wmh.vol.Dimz;
    
    wmh.vol.FOVxcmd2  = (float) wmh.vol.Dimx * wmh.vol.szcm / (float) 2.;   // half of the size of the image volume, dimension x (cm);
    wmh.vol.FOVcmyd2  = (float) wmh.vol.Dimy * wmh.vol.szcm / (float) 2.;   // half of the size of the image volume, dimension y (cm);
    wmh.vol.FOVzcmd2  = (float) wmh.vol.Dimz * wmh.vol.thcm / (float) 2.;
    
    wmh.vol.x0      = - wmh.vol.FOVxcmd2 + (float)0.5 * wmh.vol.szcm ;  // x coordinate of first voxel
    wmh.vol.y0      = - wmh.vol.FOVcmyd2 + (float)0.5 * wmh.vol.szcm ;  // y coordinate of first voxel
    wmh.vol.z0      = - wmh.vol.FOVzcmd2 + (float)0.5 * wmh.vol.thcm ;  // z coordinate of first voxel
    
    //...ring parameters ................................................
    
    wmh.detector_fn = param[ 10 ].c_str();
    
    //....collimator parameters ........................................
    
    wmh.collim_fn   = param[ 11 ].c_str();
    
    //... resolution parameters ..............................................
    
    wmh.mn_w    = atof( param[ 12 ].c_str() );
    wmh.Nsigm   = atof( param[ 13 ].c_str() );
    wmh.highres = atof( param[ 14 ].c_str() );
    wmh.subsamp = atoi( param[ 15 ].c_str() );
    
    wmh.do_subsamp = false;
    
    //...correction for intrinsic PSF....................................
    
    if ( param[ 16 ] == "no" ) wmh.do_psfi = false;
    else{
        if ( param[ 16 ] == "yes" ) wmh.do_psfi = true;
        else error_wm_SPECT_mph( 116, param[ 16 ] );
        wmh.do_subsamp = true;
    }
    
    //... impact depth .........................
    
    if ( param[ 17 ] == "no" ) {
        wmh.do_depth = false;
    }
    else{
        if ( param[ 17 ] == "yes" ) wmh.do_depth = true;
        else error_wm_SPECT_mph( 117, param[ 17 ] );
        wmh.do_subsamp = true;
    }
    
    //... attenuation parameters .........................
    
    if ( param[ 18 ] == "no" ) {
        wmh.do_att = wmh.do_full_att = false;
    }
    else{
        wmh.do_att = true;
        
        if ( param[ 18 ] == "simple" ) wmh.do_full_att = false;
        else {
            if ( param[ 18 ] == "full" ) wmh.do_full_att = true;
            else error_wm_SPECT_mph( 118, param[ 18 ] );
        }
        
        wmh.att_fn = param[ 19 ];
        
        cout << "Attenuation filename = " << wmh.att_fn << endl;
    }
    
    //... masking parameters.............................
    
    if( param[ 20 ] == "no" ) wmh.do_msk_att = wmh.do_msk_file = false;
    
    else{
        
        if( param[ 20 ] == "att" ) wmh.do_msk_att = true;
        
        else{
            if( param[ 20 ] == "file" ){
                wmh.do_msk_file = true;
                
                wmh.msk_fn = param[ 21 ];
                
                cout << "MASK filename = " << wmh.msk_fn << endl;
            }
            else error_wm_SPECT_mph( 120, param[ 20 ]);
        }
    }
        
    // ..... matrix format ..............................
    
    if ( param[ 22 ] == "STIR" ) wm.do_save_STIR = true;
    else {
        if ( param[ 22 ] == "FC" ) wm.do_save_STIR = false;
        else error_wm_SPECT_mph( 122, param[ 22 ] );
    }
    
    //... initialization of do_variables to false..............
    
    wmh.do_round_cumsum       = wmh.do_square_cumsum       = false ;
    
    //... files with complentary information .................
    
    read_prj_params_mph();
    read_coll_params_mph();
    
    //... precalculated functions ................
    
    fill_pcf();
    
    //... other variables .........................
    
    wm.Nbt     = wmh.prj.Nbt;                                                // number of rows of the weight matrix
    wm.Nvox    = wmh.vol.Nvox;                                               // number of columns of the weight matrix
    wmh.mndvh2 = ( wmh.collim.rad - wmh.ro ) * ( wmh.collim.rad - wmh.ro );  // reference distance ^2 for efficiency
    
    //... control of read parameters ..............
    
    cout << "\n\nMatrix name:" << wm.fn << endl;
    cout << "\nImage. Nrow: " << wmh.vol.Dimy << "\t\tNcol: " << wmh.vol.Dimx << "\t\tvoxel_size: " << wmh.vol.szcm<< endl;
    cout << "\nNumber of slices: " << wmh.vol.Dimz << "\t\tslice_thickness: " << wmh.vol.thcm << endl;
    cout << "\nFOVxcmd2: " << wmh.vol.FOVxcmd2 << "\t\tFOVcmyd2: " << wmh.vol.FOVcmyd2 << "\t\tradius object: " << wmh.ro <<endl;
    
    if ( wmh.do_att ){
        cout << "\nCorrection for atenuation: " << wmh.att_fn << "\t\tdo_mask: " << wmh.do_msk_att << endl;
        cout << "\nAttenuation map: " << wmh.att_fn << endl;
    }
    
    cout << "\nMinimum weight: " << wmh.mn_w << endl;
}

//==========================================================================
//=== error_wm ====================================================
//==========================================================================

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

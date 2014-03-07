/*
 * Copyright (c) 2014, 
 * Institute of Nuclear Medicine, University College of London Hospital, UCL, London, UK.
 * Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. All rights reserved.
 * This software is distributed WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 
 \author Carles Falcon
 */

//system libraries
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>

//user defined libraries

#include "wmtools_SPECT_mph.h"
#include "weight3d_SPECT_mph.h"

namespace SPECTUB_mph
{

using namespace std;

#define in_limits( a ,l1 , l2 ) ( (a) < (l1) ? (l1) : ( (a) > (l2) ? (l2) : (a) ) )

//==========================================================================
//=== wm_calculation =======================================================
//==========================================================================

void wm_calculation_mph ( bool do_calc,
                          psf2d_type *psf_bin,
                          psf2d_type *psf_subs,
                          psf2d_type *psf_aux,
                          psf2d_type *kern,
                          float *attmap,
                          bool *msk_3d,
                          int *Nitems )
{
    voxel_type vox;        // structure with voxel information
	bin_type bin;          // structure with bin information
    lor_type l;            // structure with lor information
    discrf2d_type *f;      // structure with cumsum function
    
    float weight;
	float coeff_att = (float) 1.;
	int   jp;
    float w_max= -1.;
    int Dimxd2, Dimyd2;
    
    //... collimator parameters ........................................
    
    mphcoll_type  * c  = & wmh.collim;
    
    if ( !do_calc ){
        
        //... STIR indices .......................................................................
        
        if ( wm.do_save_STIR ){
            
            Dimxd2 = wmh.vol.Dimx / 2 ;
            Dimyd2 = wmh.vol.Dimy / 2 ;
            
            wm.ns = new int [ wmh.prj.Nbt ];
            wm.nb = new int [ wmh.prj.Nbt ];
            wm.na = new int [ wmh.prj.Nbt ];
            
            wm.nx = new short int [ wmh.vol.Nvox ];
            wm.ny = new short int [ wmh.vol.Nvox ];
            wm.nz = new short int [ wmh.vol.Nvox ];
            
            jp = -1;											// projection index (row index of the weight matrix )
            
            for ( int j = 0 ; j < wmh.prj.Ndt ; j++ ){
                
                for ( int k = 0 ; k < wmh.prj.Nsli ; k++ ){

                    int nbd2 = wmh.prj.Nbin / 2;
                    
                    for ( int i = 0 ; i < wmh.prj.Nbin ; i++){
                        
                        jp++;
                        wm.na[ jp ] = j;
                        wm.nb[ jp ] = i - nbd2;
                        wm.ns[ jp ] = k;
                    }
                }
            }
        }
    }
	
    //=== LOOP1: IMAGE SLICES ================================================================
    
    for ( vox.iz = wmh.vol.first_sl ; vox.iz < wmh.vol.last_sl ; vox.iz++ ){
        
        vox.z  = wmh.vol.z0 + vox.iz * wmh.vol.thcm;
        
        cout << "weights: " << 100.*(vox.iz+1)/wmh.vol.Dimz << "%" << endl;
        
        //=== LOOP2: IMAGE ROWS =======================================================================
        
        for ( vox.iy = 0 , vox.ip = 0 ; vox.iy < wmh.vol.Dimy ; vox.iy++ ){
   
            vox.y = wmh.vol.y0 + vox.iy * wmh.vol.szcm ;              // y coordinate of the voxel (index 0->Dimy-1: ix)
            
            //=== LOOP3: IMAGE COLUMNS =================================================================
            
            for ( vox.ix = 0 ; vox.ix < wmh.vol.Dimx ; vox.ix++, vox.ip++ ){
 
                vox.iv = vox.iz * wmh.vol.Npix + vox.ip;
                
                if ( !msk_3d[ vox.iv ] ) continue;
                
                vox.x = wmh.vol.x0 + vox.ix * wmh.vol.szcm ;           // x coordinate of the voxel (index 0->Dimx-1: ix)
                
                //=== LOOP4: DETELS: DETECTOR ELEMENTS ===========================================
                
                for( int k = 0 ; k < wmh.prj.Ndt ; k++ ){
                    
                    detel_type * d  = & wmh.detel[ k ];
                    
                    //... cordinates of the voxel in the rotated reference system. .................
                    
                    vox.x1  =   vox.x * d->costh + vox.y * d->sinth ;
                    vox.y1  =  -vox.x * d->sinth + vox.y * d->costh ;
                    
                    //=== LOOP5: HOLES PER DETEL ====================================
                    
                    for( int ih = 0 ; ih < d->nh ; ih++ ){
                        
                        hole_type * h = &c->holes[ d->who[ ih ] ];
                        
                        if ( !check_xang_par( &vox, h ) ) continue;
                        if ( !check_zang_par( &vox, h ) ) continue;
                        
                        //...vector voxel-hole, angles and distances...............................
                        
                        voxel_projection_mph ( &l , &vox , h );
                        
                        //... hole shape .......................................
                        
                        if ( h->do_round ) f = &pcf.round;
                        else f = &pcf.square;
                        
                        //... geometrical part of the PSF ....................................
                        
                        if ( wmh.do_subsamp ){
                            
                            if ( wmh.do_depth ) fill_psf_depth ( psf_subs, &l, f, wmh.subsamp, do_calc );
                            
                            else fill_psf_geo ( psf_subs, &l, f, wmh.subsamp, do_calc );
                            
                            if ( wmh.do_psfi ) psf_convol ( psf_subs, psf_aux, kern, do_calc );
                           
                            downsample_psf( psf_subs, psf_bin, wmh.subsamp, do_calc );
                        }
                        
                        else  fill_psf_geo ( psf_bin, &l, f, 1, do_calc );
                        
                        //... calculus of simple attenuation .............................
                        
                        if ( do_calc ){
                            
                            if ( wmh.do_att && !wmh.do_full_att ){    // simple correction for attenuation
                                
                                bin.x = d->x0 + l.x1d_l * d->costh;   // x coord of the projection of the center of the voxel in
                                bin.y = d->y0 + l.x1d_l * d->sinth;
                                bin.z = d->z0 + l.z1d_l ;
                                
                                coeff_att = calc_att_mph( bin, vox, attmap );
                            }
                        }
                        
                        //=== LOOP6: z-dim of PSF ====================================
                        
                        for ( int j = 0, jb = psf_bin->jb0 ; j < psf_bin->dimz ; j++, jb++ ){
                            
                            if ( jb < 0 ) continue;
                            if ( jb >= wmh.prj.Nsli ) continue;
                            
                            //=== LOOP7: x-dim of PSF ====================================
                            
                            for ( int i = 0 , ib = psf_bin->ib0 ; i < psf_bin->dimx ; i++, ib++ ){
                                
                                if ( ib < 0 ) continue;
                                if ( ib >= wmh.prj.Nbin ) continue;
                                
                                jp = k * wmh.prj.Nbd + jb * wmh.prj.Nbin + ib ;
                                
                                if ( do_calc ) {
                                    
                                    weight = psf_bin->val[ j ][ i ] * l.eff / psf_bin->sum  ;
                                    
                                    if ( weight < wmh.mn_w ) continue ;
                                    
                                    //... to fill image STIR indices ...........................
                                    
                                    if ( wm.do_save_STIR ){
                       
                                        wm.nx[ vox.iv ] = (short int)( vox.ix - Dimxd2 ) ; // centered index for STIR format
                                        wm.ny[ vox.iv ] = (short int)( vox.iy - Dimyd2 ) ; // centered index for STIR format
                                        wm.nz[ vox.iv ] = (short int)  vox.iz ;            // non-centered index for STIR format
                                    }
                                    
                                    //... calculus of full attenuation ...............
                                    
                                    if ( wmh.do_full_att ){
                                        
                                        bin.x = d->xbin0 + (float) ib * d->incx ;
                                        bin.y = d->ybin0 + (float) ib * d->incy ;
                                        bin.z = d->zbin0 + (float) jb * wmh.prj.thcm ;
                                        
                                        coeff_att = calc_att_mph( bin, vox, attmap );
                                    }
                                    
                                    //... calculus and storage of the weight............
                                    
                                    weight = weight * coeff_att;
                                    
                                    if ( weight > w_max ) w_max = weight;
                                    
                                    wm.col[ jp ][ wm.ne[ jp ] ] = vox.iv;
                                    wm.val[ jp ][ wm.ne[ jp ] ] = weight;
                                    wm.ne[ jp ]++;
                                    
                                    if ( wm.ne[ jp ] >= Nitems[ jp ] ) error_weight3d( 45, "" );
                                }
                                
                                else Nitems[ jp ]++;    // for size estimation
                                
                            }   //....... end 0f LOOP7: x-dim of PSF
                        }   //........... end 0f LOOP6: z-dim of PSF
					}   //............... end of LOOP5: hole in detection element
				}   //................... end of LOOP4: detection element
			}   //....................... end of LOOP3: image rows
		}   //........................... end of LOOP2: image cols
	}   //............................... end of LOOP1: image slices
    
   if ( do_calc ) cout << "Maximum weight: " << w_max << endl;
}

//==========================================================================
//=== fill_psfi ============================================================
//==========================================================================

void fill_psfi( psf2d_type * kern )
{
    float K0 = (float)0.39894228040143 / wmh.prj.sgm_i ; //Normalization factor: 1/sqrt(2*M_PI)/sigma
    float f1 = - (float) 0.5 / ( wmh.prj.sgm_i * wmh.prj.sgm_i );
	
    float * g1d;
    float * g2d;
    
    g1d = new float [ kern->dimx ];
    for ( int i = 0 ; i < kern->dimx ; i++ ) g1d[ i ] = (float) 0.;
    
    g2d = new float [ kern->dimz ];
    for ( int i = 0; i < kern->dimz ; i++ ) g2d[ i ] = (float) 0.;
    
    int dimxd2 = kern->dimx / 2;
    int dimzd2 = kern->dimz / 2;
    
    float res1 = wmh.prj.szcm / wmh.subsamp ;
    float res2 = wmh.prj.thcm / wmh.subsamp ;
    
    //... 1d density function ..................
    
    float x  = 0;
	g1d[ dimxd2 ] = K0 ;
	
	for( int i = 1 ; i <= dimxd2  ; i++ ){
		
		x += res1 ;
		g1d[ dimxd2 - i ] = K0 * exp( f1 * x * x );
		g1d[ dimxd2 + i ] = g1d[ dimxd2  - i ];
	}
    
    //... 1d density function ....................
    
    float y  = 0;
	g2d[ dimzd2 ] = K0 ;
    
	for( int i = 1 ; i <= dimzd2  ; i++ ){
		
		y += res2 ;
		g2d[ dimzd2 - i ] = K0 * exp( f1 * y * y );
		g2d[ dimzd2 + i ] = g2d[ dimzd2  - i ];
	}
    
    //... to fill kern ..................
    
    float sum = (float) 0. ;
	
    for ( int j = 0 ; j < kern->dimz ; j++){

        for ( int i = 0 ; i < kern->dimx ; i++ ){
            
            kern->val[ j ][ i ] = g2d[ j ] * g1d[ i] ;
            
            sum += kern->val[ j ][ i ];
        }
    }

    //... normalization to area 1 ........................
    
    for ( int j = 0 ; j < kern->dimz ; j++){
        
        for ( int i = 0 ; i < kern->dimx ; i++ ){

            kern->val[ j ][ i ] /= sum ;
        }
    }
    
    delete [] g1d;
    delete [] g2d;
}

//==========================================================================
//=== check_xang_par =======================================================
//==========================================================================

bool check_xang_par( voxel_type * v, hole_type * h ){
    
    bool ans = true;
    
    //...vector voxel-hole, angles and distances...............................
    
    float ux1 = h->x1 - v->x1 ;
    float uy1 = h->y1 - v->y1 ;
    
    if ( uy1 <= EPSILON ) error_weight3d ( 88, "" );
    
    float a = atan2f( ux1, uy1 ) ;
    
    if (  a > h->ax_M  || a < h->ax_m ) ans = false ;
    
    return( ans );
}

//==========================================================================
//=== check_zang_par =======================================================
//==========================================================================

bool check_zang_par( voxel_type * v, hole_type * h ){
    
    bool ans = true;
    
    float uz1 = h->z1 - v->z ;
    float uy1 = h->y1 - v->y1 ;
    
    float a = atan2f( uz1 , uy1 ) ;
    
    if (  a > h->az_M  || a < h->az_m ) ans = false ;
    return ( ans );
}

//==========================================================================
//=== voxel_projection =====================================================
//==========================================================================

void voxel_projection_mph ( lor_type * l, voxel_type * v, hole_type * h )
{
   
    //...vector voxel-hole, angles and distances...............................
    
    float ux1 = h->x1 - v->x1 ;
    float uy1 = h->y1 - v->y1 ;
    float uz1 = h->z1 - v->z  ;
    
    if ( uy1 <= EPSILON ) error_weight3d(88, "" );
    
    //...vector voxel-hole and distances...............................
    
    float dxyz_2 = ux1 * ux1 + uy1 * uy1 + uz1 * uz1 ;
    float dvh_l  = sqrtf( dxyz_2 );
    
    ux1 /= dvh_l;
    uy1 /= dvh_l;
    uz1 /= dvh_l;
    
    //...distance over voxel-hole line from voxel and hole to detection plane..........
    
    float dvd_l = ( wmh.prj.rad - v->y1 ) / uy1 ;
    
    l->x1d_l = v->x1 + dvd_l * ux1 ;
    l->z1d_l = v->z  + dvd_l * uz1 ;
    
    //...shadow of the hole ..........
    
    l->hsxcm_d = h->dxcm * dvd_l / dvh_l ;
    l->hszcm_d = h->dzcm * dvd_l / dvh_l ;
    l->hsxcm_d_d2 = l->hsxcm_d / (float)2. ;
    l->hszcm_d_d2 = l->hszcm_d / (float)2. ;
    
    //... values at detection + crystal distance ................................
    
    if ( wmh.do_depth ){
        
        float dvdc_l   = ( wmh.prj.radc - v->y1 )/ uy1 ;

        l->hsxcm_dc = h->dxcm * dvdc_l / dvh_l ;
        l->hszcm_dc = h->dzcm * dvdc_l / dvh_l ;
        
        l->hsxcm_dc_d2 = l->hsxcm_d / (float)2. ;
        l->hszcm_dc_d2 = l->hszcm_d / (float)2. ;

        l->x1dc_l   = v->x1 + dvdc_l * ux1 ;
        l->z1dc_l   = v->z  + dvdc_l * uz1 ;
    }
    
    //... effectiveness ......................................................
    
    l->eff = wmh.mndvh2 / dxyz_2 * fabsf ( uy1 ) ;    
}

//==========================================================================
//=== fill_psf_geo =========================================================
//==========================================================================

void fill_psf_geo ( psf2d_type * psf, lor_type *l, discrf2d_type *f, int factor, bool do_calc )
{
    
    psf->xc = l->x1d_l + wmh.prj.FOVxcmd2;   // x distance of center of PSF to the begin of the FOVcm
    psf->zc = l->z1d_l + wmh.prj.FOVzcmd2;   // z distance of center of PSF to the begin of the FOVcm
    
    float xm = psf->xc - l->hsxcm_d_d2 ;
    float xM = psf->xc + l->hsxcm_d_d2 ;
    float zm = psf->zc - l->hszcm_d_d2 ;
    float zM = psf->zc + l->hszcm_d_d2 ;
    
    float resx = f->res * l->hsxcm_d ;
    float resz = f->res * l->hszcm_d ;
    
	//... first and last bin indices (they can be out of bound) ........
	
    psf->ib0 = (int) floorf( xm / wmh.prj.szcm ) ;
    psf->jb0 = (int) floorf( zm / wmh.prj.thcm ) ;
    
    int ib1 = (int) floorf ( xM / wmh.prj.szcm ) + 1 ;
    int jb1 = (int) floorf ( zM / wmh.prj.thcm ) + 1 ;
    
    //... number of elements of the PSF ..............................................................
    
    psf->dimx = ( ib1 - psf->ib0 ) * factor ;
    psf->dimz = ( jb1 - psf->jb0 ) * factor ;
    
    //... increment of incides in PSF space to cover a bin ...........................................
    
    if ( do_calc ){
        
        int if1, if2, jf1, jf2 ;
        
        int incxf = (int) roundf( wmh.prj.szcm / ( resx * (float)factor ) );
        int inczf = (int) roundf( wmh.prj.thcm / ( resz * (float)factor ) );
        
        //... to fill psf ...........................
        
        float x0 = (float)psf->ib0 * wmh.prj.szcm ;   // x distance from the fisrt bin in psf from the begin of the FOVcm
        float z0 = (float)psf->jb0 * wmh.prj.thcm ;   // z distance from the fisrt bin in psf from the begin of the FOVcm
        
        int if0 = (int) roundf( ( x0 - xm ) / resx ); // index of limit of the first bin of psf, in f space. It should be negative
        int jf0 = (int) roundf( ( z0 - zm ) / resz ); // index of limit of the first bin of psf, in f space. It should be negative
        
        psf->sum = (float)0. ;
                        
        for ( int j = 0 ; j < psf->dimz ; j++ ){
            
            jf1 = minim ( maxim ( jf0 + j * inczf, 0 ), f->j_max );
            jf2 = maxim ( minim ( jf0 + ( j + 1 ) * inczf, f->j_max ) , 0 ) ;
            
            for ( int i = 0 ; i < psf->dimx ; i++ ){
                
                if1 = minim ( maxim ( if0 + i * incxf, 0 ), f->i_max );
                if2 = maxim ( minim ( if0 + ( i + 1 ) * incxf, f->i_max ) , 0 ) ;
                
                psf->val [ j ][ i ] = f->val [ jf2 ][ if2 ] + f->val [ jf1 ][ if1 ] - f->val [ jf1 ][ if2 ] - f->val [ jf2 ][ if1 ];
   
                psf->sum += psf->val [ j ][ i ];
            }
        }
    }
}

//=============================================================================
//=== fill_psf_depth ==========================================================
//=============================================================================

void fill_psf_depth( psf2d_type *psf, lor_type *l, discrf2d_type *f, int factor, bool do_calc )
{

    float xc_d = l->x1d_l + wmh.prj.FOVxcmd2;   // x distance of center of PSF from the begin of the FOVcm
    float zc_d = l->z1d_l + wmh.prj.FOVzcmd2;   // z distance of center of PSF from the begin of the FOVcm
    
    float xc_dc = l->x1dc_l + wmh.prj.FOVxcmd2; // x distance of center of PSF from the begin of the FOVcm
    float zc_dc = l->z1dc_l + wmh.prj.FOVzcmd2; // z distance of center of PSF from the begin of the FOVcm
    
    float resx_d = f->res * l->hsxcm_d ;
    float resz_d = f->res * l->hszcm_d ;
    
    float resx_dc = f->res * l->hsxcm_dc ;
    float resz_dc = f->res * l->hszcm_dc ;
    
    psf->xc = ( xc_d + xc_dc ) / (float)2.;
    psf->zc = ( zc_d + zc_dc ) / (float)2.;
    
    //... distance for correction for attenuation inside the crystal ............................
    
    float dcr   = sqrtf ( ( l->x1d_l - l->x1dc_l ) * ( l->x1d_l - l->x1dc_l ) +
                          ( l->z1d_l - l->z1dc_l ) * ( l->z1d_l - l->z1dc_l ) + wmh.prj.crth_2 ) ;
    
	//... first and last bin indices (they can be out of bound) ........
	
    int ib0_d = (int) floorf( ( xc_d - l->hsxcm_d_d2 ) / wmh.prj.szcm ) ;
    int jb0_d = (int) floorf( ( zc_d - l->hszcm_d_d2 ) / wmh.prj.thcm ) ;
    
    int ib1_d = (int) floorf ( ( xc_d + l->hsxcm_d_d2 ) / wmh.prj.szcm ) + 1 ;
    int jb1_d = (int) floorf ( ( zc_d + l->hszcm_d_d2 ) / wmh.prj.thcm ) + 1 ;
    
    int ib0_dc = (int) floorf( ( xc_dc - l->hsxcm_dc_d2 ) / wmh.prj.szcm ) ;
    int jb0_dc = (int) floorf( ( zc_dc - l->hszcm_dc_d2 ) / wmh.prj.thcm ) ;
    
    int ib1_dc = (int) floorf ( ( xc_dc + l->hsxcm_dc_d2 ) / wmh.prj.szcm ) + 1 ;
    int jb1_dc = (int) floorf ( ( zc_dc + l->hszcm_dc_d2 ) / wmh.prj.thcm ) + 1 ;
    
    //... number of elements of the PSF ..............................................................
    
    psf->ib0 = min( ib0_d, ib0_dc);
    psf->jb0 = min( jb0_d, jb0_dc);
 
    int ib1 = max( ib1_d, ib1_dc);
    int jb1 = max( jb1_d, jb1_dc);
    
    psf->dimx = ( ib1 - psf->ib0 ) * factor  ;
    psf->dimz = ( jb1 - psf->jb0 ) * factor  ;
    
    if ( do_calc ){
        
        int if_d, jf_d, if_dc, jf_dc;
        float v;
        
        float x0_d = (float) psf->ib0 * wmh.prj.szcm;   // x distance from the fisrt bin in psf to the begin of the FOVcm
        float z0_d = (float) psf->jb0 * wmh.prj.thcm;   // z distance from the fisrt bin in psf to the begin of the FOVcm
        
        //... increments in f space ...................................................
        
        int incxf_d  = (int) roundf( wmh.prj.szcm / ( resx_d  * (float) factor ) );
        int inczf_d  = (int) roundf( wmh.prj.thcm / ( resz_d  * (float) factor ) );
        
        int incxf_dc = (int) roundf( wmh.prj.szcm / ( resx_dc * (float) factor ) );
        int inczf_dc = (int) roundf( wmh.prj.thcm / ( resz_dc * (float) factor ) );
        
        //... to fill psf ...........................
   
        int if0_d = ( x0_d - xc_d + l->hsxcm_d_d2 ) / resx_d ; // index of edge of the first bin of psf, in f space. It should be negative
        int jf0_d = ( z0_d - zc_d + l->hszcm_d_d2 ) / resz_d ; // index of erdge of the first bin of psf, in f space. It should be negative
        
        int if0_dc = ( x0_d - xc_dc + l->hsxcm_dc_d2 ) / resx_dc ; // index of edge of the first bin of psf, in f space (negative)
        int jf0_dc = ( z0_d - zc_dc + l->hszcm_dc_d2 ) / resz_dc ; // index of edge of the first bin of psf, in f space (negative)
        

        //... to initilize to zero ..............................
        
        for ( int j = 0 ; j < psf->dimz ; j++ ){
            
            for ( int i = 0 ; i < psf->dimx ; i++ ) psf->val [ j ][ i ] = (float)0. ;
        }
        psf->sum = (float)0. ;
        
        //... central part of PSF ..........................................
        
        for ( int j = 1 ; j < psf->dimz ; j++ ){
            
            jf_d  = jf0_d  + j * inczf_d  ;
            jf_dc = jf0_dc + j * inczf_dc ;
            
            for ( int i = 1 ; i < psf->dimx ; i++ ){

                if_d  = if0_d  + i * incxf_d  ;
                if_dc = if0_dc + i * incxf_dc ;
                
                v = bresenh_f( if_d, jf_d, if_dc, jf_dc, f->val, f->i_max, f->j_max, dcr );
                
                psf->val [ j ][ i ] += v ;
                psf->val [ j - 1 ][ i - 1 ] += v ;
                psf->val [ j - 1 ][ i ] -= v ;
                psf->val [ j ][ i - 1 ] -= v ;
            }
        }
        
        //... vertical edges PSF ..........................................
        
        for ( int j = 1 ; j < psf->dimz ; j++ ){
            
            jf_d  = jf0_d  + j * inczf_d  ;
            jf_dc = jf0_dc + j * inczf_dc ;
            
            if_d  = if0_d  ;
            if_dc = if0_dc ;

            v = bresenh_f( if_d, jf_d, if_dc, jf_dc, f->val, f->i_max, f->j_max, dcr );
            
            psf->val [ j ][ 0 ] += v ;
            psf->val [ j - 1 ][ 0 ] -= v ;
            
            if_d  = if0_d   + psf->dimx * incxf_d  ;
            if_dc = if0_dc  + psf->dimx * incxf_d  ;
            
            v = bresenh_f( if_d, jf_d, if_dc, jf_dc, f->val, f->i_max, f->j_max, dcr );
            
            psf->val [ j ][ psf->dimx - 1 ] -= v ;
            psf->val [ j - 1 ][ psf->dimx - 1 ] += v ;
        }
        
        //... horizontal edges PSF ..........................................
        
        for ( int i = 1 ; i < psf->dimx ; i++ ){
            
            jf_d  = jf0_d  ;
            jf_dc = jf0_dc ;

            if_d  = if0_d  + i * incxf_d  ;
            if_dc = if0_dc + i * incxf_dc ;

            v = bresenh_f( if_d, jf_d, if_dc, jf_dc, f->val, f->i_max, f->j_max, dcr );
            
            psf->val [ 0 ][ i ] += v ;
            psf->val [ 0 ][ i - 1 ] -= v ;
            
            jf_d  = jf0_d   + psf->dimz * inczf_d  ;
            jf_dc = jf0_dc  + psf->dimz * inczf_d  ;
       
            v = bresenh_f( if_d, jf_d, if_dc, jf_dc, f->val, f->i_max, f->j_max, dcr );
            
            psf->val [ psf->dimz - 1 ][ i ] -= v ;
            psf->val [ psf->dimz - 1 ][ i - 1 ] += v ;
        }
        
        //... four corners ............................................
        
        if_d  = if0_d  ;
        if_dc = if0_dc ;
        
        jf_d  = jf0_d  ;
        jf_dc = jf0_dc ;
        
        v = bresenh_f( if_d, jf_d, if_dc, jf_dc, f->val, f->i_max, f->j_max, dcr );
        
        psf->val [ 0 ][ 0 ] += v ;
        psf->sum += v ;
        
        //...
        
        if_d  = if0_d   + psf->dimx * incxf_d  ;
        if_dc = if0_dc  + psf->dimx * incxf_d  ;
        
        v = bresenh_f( if_d, jf_d, if_dc, jf_dc, f->val, f->i_max, f->j_max, dcr );
        
        psf->val [ 0 ][ psf->dimx - 1 ] -= v ;
        psf->sum -= v ;
        
        //...
        
        jf_d  = jf0_d   + psf->dimz * inczf_d  ;
        jf_dc = jf0_dc  + psf->dimz * inczf_d  ;
        
        v = bresenh_f( if_d, jf_d, if_dc, jf_dc, f->val, f->i_max, f->j_max, dcr );

        psf->val [ psf->dimz - 1 ][ psf->dimx - 1  ] += v ;
        psf->sum += v ;
        
        //...
        
        if_d  = if0_d  ;
        if_dc = if0_dc ;
        
        v = bresenh_f( if_d, jf_d, if_dc, jf_dc, f->val, f->i_max, f->j_max, dcr );
        
        psf->val [ psf->dimz - 1 ][ 0 ] -= v ;
        psf->sum -= v ;
    }
}

//==========================================================================
//=== downsample_psf =======================================================
//==========================================================================

void downsample_psf ( psf2d_type * psf_in, psf2d_type * psf_out, int factor, bool do_calc  )
{
    
    //... temporal check to remove .........................
    
    if ( ( psf_in->dimx % factor) != 0 ) error_wmtools_SPECT_mph( 55, psf_in->dimx , "dimx" );
    if ( ( psf_in->dimz % factor) != 0 ) error_wmtools_SPECT_mph( 55, psf_in->dimz , "dimz" );
    
    //... dims .................................
    
    psf_out->dimx = psf_in->dimx / factor ;
    psf_out->dimz = psf_in->dimz / factor ;
    
    psf_out->ib0 = psf_in->ib0;
    psf_out->jb0 = psf_in->jb0;
    
    if ( do_calc ){
        
        if ( psf_out->dimx > psf_out->max_dimx ) error_wmtools_SPECT_mph( 56, psf_out->dimx , "dimx" );
        if ( psf_out->dimz > psf_out->max_dimz ) error_wmtools_SPECT_mph( 56, psf_out->dimz , "dimz" );
        
        //... to fill values ........................
        
        psf_out->sum = (float) 0. ;
        
        if ( factor == 1 ){
            
            for ( int j = 0 ; j < psf_out->dimz ; j++ ){
                for ( int i = 0 ; i < psf_out->dimx ; i++ ){
                    
                    psf_out->val[ j ][ i ] = psf_in->val[ j ][ i ] ;
                    psf_out->sum += psf_out->val[ j ][ i ];
                }
            }
        }
        else{
            for ( int j = 0 ; j < psf_out->dimz ; j++ ){
                
                int jfa = j * factor;
                
                for ( int i = 0 ; i < psf_out->dimx ; i++ ){
                    
                    int ifa = i * factor;
                    
                    psf_out->val[ j ][ i ] = (float) 0.;
                    
                    for ( int m = 0 ; m < factor ; m++ ){
                        for ( int n = 0 ; n < factor ; n++ ){
                            
                            psf_out->val[ j ][ i ] += psf_in->val[ jfa + m ][ ifa + n ] ;
                        }
                    }
                    psf_out->sum += psf_out->val[ j ][ i ];
                }
            }
        }
    }
}

//==========================================================================
//=== psf_conv =============================================================
//==========================================================================

void psf_convol( psf2d_type * psf, psf2d_type * psf_aux, psf2d_type * kern, bool do_calc )
{
    int dimx = psf->dimx + kern->dimx - 1 ;
    
    if ( dimx > psf_aux->max_dimx || dimx > psf->max_dimx ) error_wmtools_SPECT_mph(77, dimx , "conv_dimx");
    
    int dimz = psf->dimz + kern->dimz - 1 ;
    
    if ( dimz > psf_aux->max_dimz || dimz > psf->max_dimz ) error_wmtools_SPECT_mph(77, dimz , "conv_dimz");
    
    //... convolution .......................
    
    if ( do_calc ){
        
        for ( int j = 0 ; j < dimz ; j++ ) {
            
            int N1 = kern->dimz - j - 1 ;
            int N2 = psf->dimz + N1 ;
            
            for ( int i = 0 ; i < dimx ; i++ ) {
                
                int M1 = kern->dimx - i - 1 ;
                int M2 = psf->dimx + M1 ;
                
                psf_aux->val[ j ][ i ] = (float)0. ;
                
                for ( int n = max( N1 , 0 ) ; n < min( N2 , kern->dimz ) ; n++ ) {
                    
                    for ( int m = max( M1 , 0 ) ; m < min( M2 , kern->dimx ) ; m++ ) {
                        
                        psf_aux->val[ j ][ i ] += kern->val[ n ][ m ] * psf->val[ n - N1 ][ m - M1 ];
                    }
                }
            }
        }
        
        //... to refill psf with new values ..................
        
        for ( int j = 0 ; j < dimz ; j++ ) {
            
            for ( int i = 0 ; i < dimx ; i++ ) {
                
                psf->val[ j ][ i ] = psf_aux->val[ j ][ i ];
            }
        }
    }
    //... sizes and position ....................
    
    psf->ib0 += kern->ib0 ;
    psf->jb0 += kern->jb0 ;
    
    psf->dimx = dimx;
    psf->dimz = dimz;
}

//==========================================================================
//=== bresenh_f ============================================================
//==========================================================================

float bresenh_f( int i1, int j1, int i2, int j2, float ** f , int imax, int jmax, float dcr )
{
    
    int er;	//the error term
    int di, dj;
    
    float acum = f[ in_limits ( j1, 0, imax ) ][ in_limits ( i1, 0, jmax ) ];
    
    //... difference between starting and ending points..........
    
    int Di = i2 - i1;
    int Dj = j2 - j1;
    
    int dist   = max ( abs( Di ), abs( Dj ) );
    int ie     = 0;
    float inc_ie = dcr / ( (float)dist * wmh.highres );
    
    //... calculate direction of the vector and store in ix and iy.....
    
    if ( Di >= 0 ) di = 1;
    else {
        di = -1;
        Di = -Di;
    }
    
    if ( Dj >= 0 ) dj = 1;
    else {
        dj = -1;
        Dj = -Dj;
    }
    
    //... scale deltas and store in dx2 and dy2.....
    
    int Di2 = Di * 2;
    int Dj2 = Dj * 2;
    
    if ( Di > Dj ){	      // dx is the major axis.......
    
        //... initialize the error term.........
        
        er = Dj2 - Di;
        
        for ( int k = 0 ; k < Di ; k++ ){
        
            if (er >= 0){
                
                er -= Di2;
                j1 += dj;
            }
            
            er += Dj2;
            i1 += di;
            ie = (int) floorf( inc_ie * (float)k );
            
            if ( ie > pcf.cr_att.i_max ) cout << " out of bounds a bresenh_f " << endl;
            
            acum += f[ in_limits ( j1, 0, imax ) ][ in_limits ( i1, 0, jmax ) ] * pcf.cr_att.val[ ie ];
        }
        if ( Di > 0 ) acum /= Di ;
    }
    
    else {		      // dy is the major axis..............
    
        //... initialize the error term ................
        
        er = Di2 - Dj;
        
        for ( int k = 0 ; k < Dj ; k++ ){
            
            if (er >= 0){
                
                er -= Dj2;
                i1 += di;
            }
            
            er += Di2;
            j1 += dj;
            
            ie = (int) floorf( inc_ie * (float)k );
            
            if ( ie > pcf.cr_att.i_max ) cout << " out of bounds a bresenh_f " << endl;
            //cout << pcf.cr_att.val[ ie ] << endl;
            
            acum += f[ in_limits ( j1, 0, imax ) ][ in_limits ( i1, 0, jmax ) ] * pcf.cr_att.val[ ie ];
            
        }
        if ( Dj > 0 ) acum /= Dj ;
    }
    return( acum );
}
    
//=============================================================================
//=== cal_att_mph =============================================================
//=============================================================================

float calc_att_mph( bin_type bin, voxel_type vox, float * attmap )
{
	float dx, dy, dz;
	float dlast_x, dlast_y, dlast_z, dlast;
	float next_x, next_y, next_z;
	int   cas;
    int iv = vox.iv;
	
	//... vector from voxel to bin and the sign of its components ....
	
	float ux = bin.x - vox.x;       // first component of voxel_to_bin vector
	float uy = bin.y - vox.y;       // second component of voxel_to_bin vector
	float uz = bin.z - vox.z;       // third component of voxel_to_bin vector
	
	int signx = SIGN( ux );           // sign of ux
	int signy = SIGN( uy );           // sign of uy
	int signz = SIGN( uz );           // sign of uz
    
	//... corresponding unary vector ...................................
	
	float dpb = sqrt( ux * ux + uy * uy + uz * uz ); // distance from voxel_to_bin (modulus of [ux,uy,uz])
    
	ux = ux / dpb + EPSILON;        // unit vector ux
	uy = uy / dpb + EPSILON;		// unit vector uy
	uz = uz / dpb + EPSILON;        // unit vector uz
    
    //... increment of variables along att pathway ............................
    
    int inc_vi_y = signy * wmh.vol.Dimx;
    int inc_vi_z = signz * wmh.vol.Npix;
    float inc_dx = wmh.vol.szcm * signx / ux ;
	float inc_dy = wmh.vol.szcm * signy / uy ;
    float inc_dz = wmh.vol.thcm * signz / uz ;
    
	//... next and last distance to the attenuation map grip ..................
	
	if ( signx < 0 ){
		next_x  = wmh.vol.x0 + ( (float) vox.ix - (float) 0.5 ) * wmh.vol.szcm ;
		dlast_x = ( -wmh.vol.FOVxcmd2 - vox.x ) / ux  ;
	}
	else{
		next_x  = wmh.vol.x0 + ( (float) vox.ix + (float) 0.5 ) * wmh.vol.szcm ;
		dlast_x = ( wmh.vol.FOVxcmd2 - vox.x ) / ux ;
	}
	
	if ( signy < 0 ){
		next_y  = wmh.vol.y0 + ( (float) vox.iy - (float) 0.5 ) * wmh.vol.szcm ;
		dlast_y = ( -wmh.vol.FOVcmyd2 - vox.y ) / uy ;
	}
	else{
		next_y  = wmh.vol.y0 + ( (float) vox.iy + (float) 0.5 ) * wmh.vol.szcm ;
		dlast_y = ( wmh.vol.FOVcmyd2 - vox.y ) / uy ;
	}
	
	if ( signz < 0 ){
		next_z  = wmh.vol.z0 +( (float) vox.iz - (float) 0.5 ) * wmh.vol.thcm ;
		dlast_z = ( -wmh.vol.FOVzcmd2 - vox.z ) / uz ;
	}
	else{
		next_z  = wmh.vol.z0 + ( (float) vox.iz + (float) 0.5 ) * wmh.vol.thcm ;
		dlast_z = ( wmh.vol.FOVzcmd2 - vox.z ) / uz ;
	}
	
	dlast = minim ( minim ( dlast_x, dlast_y ) , minim ( dlast_z , dpb ) );
	
	// ... distance to next planes  ...
	
	dx = ( next_x - vox.x ) / ux ;
	dy = ( next_y - vox.y ) / uy ;
	dz = ( next_z - vox.z ) / uz ;
	
	//... variables initialization .....................................
	
	float dant  = (float)0. ;   // previous distance (distance from voxel to the last change of voxel in the attenuation map)
    float att_coef = (float)0.;
	
	//... loop while attenuation ray is inside the attenuation map
	
	for(;;){
		
		cas = comp_dist( dx, dy, dz, dlast );
        
        switch(cas){
            case 0:
                att_coef = exp( -att_coef );
                return ( att_coef );
            case 1:
                att_coef += ( dx - dant ) * attmap[ iv ];
                dant      = dx ;
                iv       += signx ;
                dx       += inc_dx ;
                break;
            case 2:
                att_coef += ( dy - dant ) * attmap[ iv ];
                dant      = dy ;
                iv       += inc_vi_y ;
                dy       += inc_dy ;
                break;
            case 3:
                att_coef += ( dz - dant ) * attmap[ iv ];
                dant      = dz;
                iv       += inc_vi_z ;
                dz       += inc_dz ;
                break;
            default:
                error_weight3d (40, "");
        }
    }
}

//=============================================================================
//=== comp_dist ===============================================================
//=============================================================================

int comp_dist( float dx,
              float dy,
              float dz,
              float dlast)
{
	int cas;
	
	if ( dx < dy){
		if ( dx< dz) {
			if ( dx > dlast ) cas = 0;   // case 0: end of the iteration
			else cas = 1;                // case 1: minimum value = dx. Next index change in attenuation map is in x direction
		}
		else {
			if ( dz > dlast ) cas = 0;   // case 0: end of the iteration
			else cas = 3;                // case 3: minimum value = dz. Next index change in attenuation map is in z direction
		}
	}
	else{
		if ( dy < dz) {
			if ( dy > dlast ) cas = 0;   // case 0: end of the iteration
			else cas = 2;                // case 2: minimum value = dy. Next index change in attenuation map is in y direction
		}
		else {
			if ( dz > dlast ) cas = 0;   // case 0: end of the iteration
			else cas = 3;                // case 3: minimum value = dz. Next index change in attenuation map is in z direction		}
		}
	}
	return( cas );
}

//==========================================================================
//=== error_weight3d =======================================================
//==========================================================================

void error_weight3d ( int nerr, string text )
{
	switch(nerr){
		case 13: printf( "\n\nError %d weight3d: wm.NbOS and/or wm.Nvox are negative", nerr ); break;
		case 21: printf( "\n\nError %d weight3d: undefined collimator. Collimator %s not found\n", nerr ,text.c_str() ); break;
		case 30: printf( "\n\nError %d weight3d: can not open \n%s for reading\n", nerr, text.c_str() ); break;
		case 31: printf( "\n\nError %d weight3d: can not open \n%s for writing\n", nerr, text.c_str() ); break;
		case 40: printf( "\n\nError %d weight3d: wrong codification in comp_dist function", nerr );break;
		case 45: printf( "\n\nError %d weight3d: Realloc needed for WM\n", nerr ); break;
		case 47: printf( "\n\nError %d weight3d: psf length greater than maxszb in calc_psf_bin\n", nerr ); break;
		case 49: printf( "\n\nError %d weight3d: attpth larger than allocated\n", nerr ); break;
		case 50: printf( "\n\nError %d weight3d: No header stored in %s \n", nerr, text.c_str() ); break;
        case 88: printf( "\n\nError %d weight3d: voxel located behin or within the hole.\nRevise volume settings or use cyl mask\n", nerr ); break;
        default: printf( "\n\nError %d weight3d: %d unknown error number on error_weight3d()", nerr, nerr );
	}
	exit(0);
}

} // end of namespace

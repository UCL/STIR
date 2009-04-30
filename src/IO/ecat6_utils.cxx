//
// $Id$
//


/*! 
  \file
  \ingroup ECAT

  \brief Implementation of ECAT 6 CTI functions to access data
  \author Kris Thielemans (conversions from/to VAX floats, longs)
  \author PARAPET project
  $Revision$
  $Date$

  \warning This file relies on ByteOrderDefine.h to find out if it 
  has to byteswap. This ideally would be changed to use the class stir::ByteOrder. 
  Make sure you run test/test_ByteOrder.

*/
/*
  Copyright (C) 2000 PARAPET partners
  Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
  See STIR/LICENSE.txt for details
  */

/* History

  Original version contained code based on files by Larry Byars
  converted to C++ etc by PARAPET project

  KT 10/11/2000
  - added support for attenuation files
  - added support for data types different from 16 bit ints.
  - added a bit more diagonistics for file IO errors

  KT 11/01/2001 
  - added cti_read_norm_subheader,get_normheaders and removed get_attndata 
    as it was identical to get_scandata

  KT 10/09/2004
  - removed aliasing bugs in get_vax_float etc

  KT 13/01/2008 
  replace original CTI code with calls to LLN matrix library:
  - introduced mhead_ptr in various functions
  - have #define STIR_ORIGINAL_ECAT6 to be able to switch between old and new version

  KT 29/04/2009
  removed CTI-derived code
*/
#include "stir/IO/stir_ecat_common.h"
#include "stir/IO/ecat6_utils.h"     

#ifndef STIR_ORIGINAL_ECAT6
// we will need file_data_to_host which is declared in machine_indep.h
// However, that file has a problem with the definition of swab on some systems
// so we declare it here
//#include "machine_indep.h"
extern "C" int file_data_to_host(char *dptr, int nblks, int dtype);
extern "C" FILE *mat_create(char *fname, Main_header *mhead);
#endif

#include "stir/ByteOrder.h"
#include "stir/ByteOrderDefine.h"
#include "boost/static_assert.hpp"
#include "boost/cstdint.hpp"
#include <algorithm> // for std::swap
#include <limits.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// replace bcopy with memcpy
#define bcopy(src, dest, length) memcpy(dest, src, length)
#define toblocks(x) ((x + (MatBLKSIZE - 1))/MatBLKSIZE)

BOOST_STATIC_ASSERT(sizeof(unsigned short)==2); 



START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT6


int get_scanheaders (FILE *fptr, long matnum, ECAT6_Main_header *mhead, 
                     Scan_subheader *shead, ScanInfoRec *scanParams)
{
    int status;
    MatDir entry;

        // check the header
    status = cti_read_ECAT6_Main_header (fptr, mhead);
    if (status != EXIT_SUCCESS) return EXIT_FAILURE;
    
    if (mhead->file_type != matScanFile) {
	printf ("\n- file is not a scan file, type = %d\n", mhead->file_type);
#ifdef STIR_ORIGINAL_ECAT6
	dump_ECAT6_Main_header (0, mhead);
#endif
	return EXIT_FAILURE;
    }

        // look up matnum in scan file
    if (!cti_lookup (fptr, mhead, matnum, &entry)) {
	printf ("\n- specified matrix not in scan file\n");
#ifdef STIR_ORIGINAL_ECAT6
	dump_ECAT6_Main_header (0, mhead);
#endif
	return EXIT_FAILURE;
    }

        // read scan subheader
    status = cti_read_scan_subheader (fptr, mhead, entry.strtblk, shead);
    if (status != EXIT_SUCCESS) {
	printf ("\n- error reading scan subheader\n");
	return EXIT_FAILURE;
    }

    scanParams->strtblk = entry.strtblk + 1;
    scanParams->nblks = entry.endblk - entry.strtblk;
#ifndef STIR_ORIGINAL_ECAT6
    scanParams->nprojs = shead->num_r_elements;
    scanParams->nviews = shead->num_angles;
#else
    scanParams->nprojs = shead->dimension_1;
    scanParams->nviews = shead->dimension_2;
#endif
    scanParams->data_type = shead->data_type;
#ifdef STIR_ORIGINAL_ECAT6
    if (shead->data_type != mhead->data_type)
        printf("\nget_scanheader warning: \n"
"data types differ between main header (%d) and subheader (%d)\n"
"Using value from subheader\n", mhead->data_type, shead->data_type);
#endif

    return EXIT_SUCCESS;
}

int get_scandata (FILE *fptr, char *scan, ScanInfoRec *scanParams)
{
    int status;

        // read data from scan file
    if (!scan) return EXIT_FAILURE;

    status= cti_rblk(fptr, scanParams->strtblk, (char *) scan, scanParams->nblks);
    if (status != EXIT_SUCCESS) 
      return EXIT_FAILURE;
    return 
      file_data_to_host(scan, scanParams->nblks,scanParams->data_type);
}


int get_attnheaders (FILE *fptr, long matnum, ECAT6_Main_header *mhead, 
                     Attn_subheader *shead, ScanInfoRec *attnParams)
{
    int status;
    MatDir entry;

        // check the header
    status = cti_read_ECAT6_Main_header (fptr, mhead);
    if (status != EXIT_SUCCESS) return EXIT_FAILURE;
    
    if (mhead->file_type != matAttenFile) {
	printf ("\n- file is not a attn file, type = %d\n", mhead->file_type);
#ifdef STIR_ORIGINAL_ECAT6
	dump_ECAT6_Main_header (0, mhead);
#endif
	return EXIT_FAILURE;
    }

        // look up matnum in attn file
    if (!cti_lookup (fptr, mhead, matnum, &entry)) {
	printf ("\n- specified matrix not in attn file\n");
#ifdef STIR_ORIGINAL_ECAT6
	dump_ECAT6_Main_header (0, mhead);
#endif
	return EXIT_FAILURE;
    }

        // read attn subheader
    status = cti_read_attn_subheader (fptr, mhead, entry.strtblk, shead);
    if (status != EXIT_SUCCESS) {
	printf ("\n- error reading attn subheader\n");
	return EXIT_FAILURE;
    }

    attnParams->strtblk = entry.strtblk + 1;
    attnParams->nblks = entry.endblk - entry.strtblk;
#ifndef STIR_ORIGINAL_ECAT6
    attnParams->nprojs = shead->num_r_elements;
    attnParams->nviews = shead->num_angles;
#else
    attnParams->nprojs = shead->dimension_1;
    attnParams->nviews = shead->dimension_2;
#endif
    attnParams->data_type = shead->data_type;
#ifdef STIR_ORIGINAL_ECAT6
    if (shead->data_type != mhead->data_type)
        printf("\nget_attnheader warning: \n"
"data types differ between main header (%d) and subheader (%d)\n"
"Using value from subheader\n", mhead->data_type, shead->data_type);
#endif

    return EXIT_SUCCESS;
}



int get_normheaders (FILE *fptr, long matnum, ECAT6_Main_header *mhead, 
                     Norm_subheader *shead, ScanInfoRec *normParams)
{
    int status;
    MatDir entry;

        // check the header
    status = cti_read_ECAT6_Main_header (fptr, mhead);
    if (status != EXIT_SUCCESS) return EXIT_FAILURE;
    
    if (mhead->file_type != matNormFile) {
	printf ("\n- file is not a norm file, type = %d\n", mhead->file_type);
#ifdef STIR_ORIGINAL_ECAT6
	dump_ECAT6_Main_header (0, mhead);
#endif
	return EXIT_FAILURE;
    }

        // look up matnum in norm file
    if (!cti_lookup (fptr, mhead, matnum, &entry)) {
        printf ("\n- specified matrix not in norm file\n");
#ifdef STIR_ORIGINAL_ECAT6
	dump_ECAT6_Main_header (0, mhead);
#endif
	return EXIT_FAILURE;
    }

        // read norm subheader
    status = cti_read_norm_subheader (fptr, mhead, entry.strtblk, shead);
    if (status != EXIT_SUCCESS) {
	printf ("\n- error reading norm subheader\n");
	return EXIT_FAILURE;
    }

    normParams->strtblk = entry.strtblk + 1;
    normParams->nblks = entry.endblk - entry.strtblk;
#ifndef STIR_ORIGINAL_ECAT6
    normParams->nprojs = shead->num_r_elements;
    normParams->nviews = shead->num_angles;
#else
    normParams->nprojs = shead->dimension_1;
    normParams->nviews = shead->dimension_2;
#endif
    normParams->data_type = shead->data_type;
#ifdef STIR_ORIGINAL_ECAT6
    if (shead->data_type != mhead->data_type)
        printf("\nget_normheader warning: \n"
"data types differ between main header (%d) and subheader (%d)\n"
"Using value from subheader\n", mhead->data_type, shead->data_type);
#endif

    return EXIT_SUCCESS;
}

#ifndef STIR_ORIGINAL_ECAT6
FILE*
cti_create(const char * const fname, const Main_header *mhead)
{
  return mat_create(const_cast<char *>(fname), const_cast<Main_header *>(mhead));
}

int     cti_read_ECAT6_Main_header (FILE *fptr, ECAT6_Main_header *h)
{
  const int cti_status = mat_read_main_header(fptr, h);
  return cti_status==0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

long    cti_numcod (int frame, int plane, int gate, int data, int bed)
{
  return mat_numcod(frame, plane, gate, data, bed);
}

void cti_numdoc (long matnum, Matval *matval)
{
  mat_numdoc(matnum, matval);
}

int     cti_rblk (FILE *fptr, int blkno, void *bufr, int nblks)
{
  const int cti_status = mat_rblk(fptr, blkno, reinterpret_cast<char *>(bufr), nblks);
  return cti_status==0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int     cti_wblk (FILE *fptr, int blkno, void *bufr, int nblks)
{
  const int cti_status = cti_wblk (fptr, blkno, bufr, nblks);
  return cti_status==0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int     cti_enter (FILE *fptr, const ECAT6_Main_header* mhead_ptr, long matnum, int nblks)
{
  return mat_enter(fptr, const_cast<ECAT6_Main_header*>(mhead_ptr), matnum, nblks);
}

int     cti_lookup (FILE *fptr, const ECAT6_Main_header* mhead_ptr, long matnum, MatDir *entry)
{
  return mat_lookup(fptr, const_cast<ECAT6_Main_header*>(mhead_ptr), matnum, entry);
}

int	cti_read_image_subheader (FILE *fptr, const ECAT6_Main_header *h, int   blknum, Image_subheader *header_ptr)
{
  const int cti_status = mat_read_image_subheader (fptr, const_cast<ECAT6_Main_header *>(h), blknum, header_ptr);
  return cti_status==0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int     cti_read_scan_subheader (FILE *fptr, const ECAT6_Main_header *h, int blknum, Scan_subheader *header_ptr)
{
  const int cti_status = mat_read_scan_subheader (fptr, const_cast<ECAT6_Main_header *>(h), blknum, header_ptr);
  return cti_status==0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int     cti_read_attn_subheader (FILE *fptr, const ECAT6_Main_header *h, int blknum, Attn_subheader *header_ptr)
{
  const int cti_status = mat_read_attn_subheader (fptr, const_cast<ECAT6_Main_header *>(h), blknum, header_ptr);
  return cti_status==0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int     cti_read_norm_subheader (FILE *fptr, const ECAT6_Main_header *h, int blknum, Norm_subheader *header_ptr)
{
  const int cti_status = mat_read_norm_subheader (fptr, const_cast<ECAT6_Main_header *>(h), blknum, header_ptr);
  return cti_status==0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int     cti_write_image_subheader (FILE *fptr, const ECAT6_Main_header *h, int blknum, const Image_subheader *header_ptr)
{
  const int cti_status = mat_write_image_subheader (fptr, const_cast<ECAT6_Main_header *>(h), blknum, const_cast<Image_subheader *>(header_ptr));
  return cti_status==0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int     cti_write_scan_subheader (FILE *fptr, const ECAT6_Main_header *h, int blknum, const Scan_subheader *header_ptr)
{
  const int cti_status = mat_write_scan_subheader (fptr, const_cast<ECAT6_Main_header *>(h), blknum, const_cast<Scan_subheader *>(header_ptr));
  return cti_status==0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int file_data_to_host(char *dptr, int nblks, int dtype)
{
  const int cti_status = ::file_data_to_host(dptr, nblks, dtype);
  return cti_status==0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else// STIR_ORIGINAL_ECAT6

#error Original ECAT6 code removed

#endif // STIR_ORIGINAL_ECAT6

int cti_rings2plane (short nrings, short ring0, short ring1) 

{
    int d = (int) (ring0 / (nrings/2)); 

    return (ring1 * nrings/2 + ring0 % (nrings/2) +
            nrings/2 * nrings * d + 1);			 
}

#ifdef STIR_ORIGINAL_ECAT6
#endif // STIR_ORIGINAL_ECAT6

int cti_write_idata (FILE *fptr, int blk, const short *data, int ibytes)
{
    unsigned int nblks;
    char *dataptr;
    int status;

    if (ibytes%MatBLKSIZE != 0)
      {
	warning("Error writing ECAT6 data: data_size should be a multiple of %d.\nNo Data written to file.",
		MatBLKSIZE);
	return (EXIT_FAILURE);
      }      
#if STIRIsNativeByteOrderBigEndian
    char bufr[MatBLKSIZE];

    dataptr = (char *) data;    // point into data buffer

        // we'll use cti_wblk to write the data via another buffer.
        // this way, if we need to transform the data as we went, we can do it.
    nblks = toblocks (ibytes);
    for (unsigned int i=0; i<nblks; i++) {
	bcopy (dataptr, bufr, MatBLKSIZE);
	swab (bufr, bufr, MatBLKSIZE);
	if ((status = cti_wblk (fptr, blk + i, bufr, 1)) != EXIT_SUCCESS) {
	    return (EXIT_FAILURE);
	}
	dataptr += MatBLKSIZE;
    }
    fflush (fptr);

#else
        // write the data in blocks
    nblks = toblocks (ibytes);
    dataptr = (char *) data;
    
    if ((status = cti_wblk (fptr, blk, dataptr, nblks)) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
    fflush (fptr);

#endif
    return (EXIT_SUCCESS);
}

#ifdef STIR_ORIGINAL_ECAT6
#endif // STIR_ORIGINAL_ECAT6

int cti_write_image (FILE *fptr, long matnum, const ECAT6_Main_header *mhead_ptr, const Image_subheader *header,
                     const short *data, int data_size)
{
    int nxtblk, nblocks;
    int status;

    nblocks = (data_size + (MatBLKSIZE - 1)) / MatBLKSIZE;
    nxtblk = cti_enter (fptr, mhead_ptr, matnum, nblocks);
    if (nxtblk <= 0) return (EXIT_FAILURE);

    status = cti_write_image_subheader (fptr, mhead_ptr, nxtblk, header);
    if (status != EXIT_SUCCESS) return (EXIT_FAILURE);
   
    status = cti_write_idata (fptr, nxtblk + 1, data, data_size);
    return status;
}

int cti_write_scan (FILE *fptr, long matnum, const ECAT6_Main_header *mhead_ptr, const Scan_subheader *header,
		    const short *data, int data_size)
{
    int nxtblk, nblocks;
    int status;

    nblocks = (data_size + (MatBLKSIZE - 1)) / MatBLKSIZE;
    nxtblk = cti_enter (fptr, mhead_ptr, matnum, nblocks);
    if (nxtblk <= 0) return (EXIT_FAILURE);

    status = cti_write_scan_subheader (fptr, mhead_ptr, nxtblk, header);
    if (status != EXIT_SUCCESS) return (EXIT_FAILURE);

    status = cti_write_idata (fptr, nxtblk + 1, data, data_size);
    return status;
}

#ifdef STIR_ORIGINAL_ECAT6
#endif

static void fill_string (char *str, int len)
{
    for (int i=0; i<len-2; i++) str[i]='.';
    str[len-2]='\0';
}
#ifdef STIR_ORIGINAL_ECAT6

#endif // STIR_ORIGINAL_ECAT6

Scan_subheader scan_zero_fill() 
{ 
    Scan_subheader v_shead;

#ifndef STIR_ORIGINAL_ECAT6
    v_shead.x_resolution= -1.0;
#else
    v_shead.sample_distance= -1.0;
    v_shead.isotope_halflife= -1.0;
#endif // STIR_ORIGINAL_ECAT6
    v_shead.scale_factor= -1.0;
    v_shead.loss_correction_fctr= -1.0;
    v_shead.tot_avg_cor= -1.0;
    v_shead.tot_avg_uncor= -1.0;
    for(int i=0;i<16;i++) v_shead.cor_singles[i]= -1.0;
    for(int i=0;i<16;i++) v_shead.uncor_singles[i]= -1.0;

    v_shead.gate_duration= 0;
    v_shead.r_wave_offset= -1;
    v_shead.prompts= -1;
    v_shead.delayed= -1;
    v_shead.multiples= -1;
    v_shead.net_trues= -1;
    v_shead.total_coin_rate= -1;
    v_shead.frame_start_time= 0;
    v_shead.frame_duration= 0;
    v_shead.data_type= -1;
#ifndef STIR_ORIGINAL_ECAT6
    v_shead.num_r_elements = -1;
    v_shead.num_angles = -1;
    v_shead.corrections_applied= 0;
#else
    v_shead.dimension_1= -1;
    v_shead.dimension_2= -1;
    v_shead.smoothing= -1;
    v_shead.processing_code= 0;
    v_shead.frame_duration_sec= -1;
#endif
    v_shead.scan_min= -1;
    v_shead.scan_max= -1;

    return(v_shead);
}

Image_subheader img_zero_fill() 
{
    Image_subheader v_ihead;

#ifndef STIR_ORIGINAL_ECAT6
    v_ihead.x_offset= -1.0;
    v_ihead.y_offset= -1.0;
    v_ihead.z_offset= -1.0;
    v_ihead.recon_zoom= -1.0;                    // Image ZOOM from reconstruction
    v_ihead.scale_factor= -1.0;                    // Scale Factor 
    v_ihead.x_pixel_size= -1.0;
    v_ihead.y_pixel_size= -1.0;
    v_ihead.z_pixel_size= -1.0;
    v_ihead.z_rotation_angle = -1.0;
    v_ihead.filter_cutoff_frequency = -1;
    v_ihead.filter_ramp_slope = -1;
    v_ihead.filter_order = -1;
    v_ihead.filter_scatter_fraction = -1;
    v_ihead.filter_scatter_slope = -1;
#else
    v_ihead.x_origin= -1.0;
    v_ihead.y_origin= -1.0;
    v_ihead.recon_scale= -1.0;                    // Image ZOOM from reconstruction
    v_ihead.quant_scale= -1.0;                    // Scale Factor 
    v_ihead.pixel_size= -1.0;
    v_ihead.slice_width= -1.0;
    v_ihead.image_rotation= -1.0;
    v_ihead.plane_eff_corr_fctr= -1.0;
    v_ihead.decay_corr_fctr= -1.0;
    v_ihead.loss_corr_fctr= -1.0;
    v_ihead.ecat_calibration_fctr= -1.0;
    v_ihead.well_counter_cal_fctr= -1.0;
    for(int i=0;i<6;i++) v_ihead.filter_params[i]= -1.0;
#endif
    v_ihead.frame_duration= 0;
    v_ihead.frame_start_time= 0;
#ifdef STIR_ORIGINAL_ECAT6
    v_ihead.recon_duration= -1;
    v_ihead.scan_matrix_num= -1;
    v_ihead.norm_matrix_num= -1;
    v_ihead.atten_cor_matrix_num= -1;
#endif
    v_ihead.data_type= -1;
    v_ihead.num_dimensions= -1;
#ifndef STIR_ORIGINAL_ECAT6
    v_ihead.x_offset = -1;
    v_ihead.y_offset = -1;
    v_ihead.z_offset = -1;
#else
    v_ihead.dimension_1= -1;
    v_ihead.dimension_2= -1;
#endif
    v_ihead.image_min= -1;
    v_ihead.image_max= -1;
#ifndef STIR_ORIGINAL_ECAT6
#else
    v_ihead.slice_location= -1;
    v_ihead.recon_start_hour= -1;
    v_ihead.recon_start_minute= -1;
    v_ihead.recon_start_sec= -1;
#endif
    v_ihead.filter_code= -1;
    v_ihead.processing_code= -1;
#ifdef STIR_ORIGINAL_ECAT6
    v_ihead.quant_units= -1;
    v_ihead.recon_start_day= -1;
    v_ihead.recon_start_month= -1;
    v_ihead.recon_start_year= -1;
#endif

    fill_string(v_ihead.annotation, 40);

    return(v_ihead);
}

#ifdef STIR_ORIGINAL_ECAT6

#endif // STIR_ORIGINAL_ECAT6

END_NAMESPACE_ECAT6
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

//
// $Id$
//


/*! 
  \file
  \ingroup ECAT

  \brief Implementation of ECAT 6 CTI functions to access data
  \author Larry Byars
  \author Kris Thielemans (conversions from/to VAX floats, longs)
  \author PARAPET project
  $Revision$
  $Date$

  \warning This file relies on ByteOrderDefine.h to find out if it 
  has to byteswap. This ideally would be changed to use the class ByteOrder. 
  Make sure you run test/test_ByteOrder.

*/
/*
  Copyright (C) CTI PET Inc.
  Copyright (C) 2000 PARAPET partners
  Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
  See STIR/LICENSE.txt for details
  */

/* History

  based on files by Larry Byars
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
*/
#include "stir/IO/stir_ecat_common.h"
#include "stir/IO/ecat6_utils.h"     
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
	dump_ECAT6_Main_header (0, mhead);
	return EXIT_FAILURE;
    }

        // look up matnum in scan file
    if (!cti_lookup (fptr, matnum, &entry)) {
	printf ("\n- specified matrix not in scan file\n");
	dump_ECAT6_Main_header (0, mhead);
	return EXIT_FAILURE;
    }

        // read scan subheader
    status = cti_read_scan_subheader (fptr, entry.strtblk, shead);
    if (status != EXIT_SUCCESS) {
	printf ("\n- error reading scan subheader\n");
	return EXIT_FAILURE;
    }

    scanParams->strtblk = entry.strtblk + 1;
    scanParams->nblks = entry.endblk - entry.strtblk;
    scanParams->nprojs = shead->dimension_1;
    scanParams->nviews = shead->dimension_2;
    scanParams->data_type = shead->data_type;
    if (shead->data_type != mhead->data_type)
        printf("\nget_scanheader warning: \n"
"data types differ between main header (%d) and subheader (%d)\n"
"Using value from subheader\n", mhead->data_type, shead->data_type);

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
	dump_ECAT6_Main_header (0, mhead);
	return EXIT_FAILURE;
    }

        // look up matnum in attn file
    if (!cti_lookup (fptr, matnum, &entry)) {
	printf ("\n- specified matrix not in attn file\n");
	dump_ECAT6_Main_header (0, mhead);
	return EXIT_FAILURE;
    }

        // read attn subheader
    status = cti_read_attn_subheader (fptr, entry.strtblk, shead);
    if (status != EXIT_SUCCESS) {
	printf ("\n- error reading attn subheader\n");
	return EXIT_FAILURE;
    }

    attnParams->strtblk = entry.strtblk + 1;
    attnParams->nblks = entry.endblk - entry.strtblk;
    attnParams->nprojs = shead->dimension_1;
    attnParams->nviews = shead->dimension_2;
    attnParams->data_type = shead->data_type;
    if (shead->data_type != mhead->data_type)
        printf("\nget_attnheader warning: \n"
"data types differ between main header (%d) and subheader (%d)\n"
"Using value from subheader\n", mhead->data_type, shead->data_type);

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
	dump_ECAT6_Main_header (0, mhead);
	return EXIT_FAILURE;
    }

        // look up matnum in norm file
    if (!cti_lookup (fptr, matnum, &entry)) {
	printf ("\n- specified matrix not in norm file\n");
	dump_ECAT6_Main_header (0, mhead);
	return EXIT_FAILURE;
    }

        // read norm subheader
    status = cti_read_norm_subheader (fptr, entry.strtblk, shead);
    if (status != EXIT_SUCCESS) {
	printf ("\n- error reading norm subheader\n");
	return EXIT_FAILURE;
    }

    normParams->strtblk = entry.strtblk + 1;
    normParams->nblks = entry.endblk - entry.strtblk;
    normParams->nprojs = shead->dimension_1;
    normParams->nviews = shead->dimension_2;
    normParams->data_type = shead->data_type;
    if (shead->data_type != mhead->data_type)
        printf("\nget_normheader warning: \n"
"data types differ between main header (%d) and subheader (%d)\n"
"Using value from subheader\n", mhead->data_type, shead->data_type);

    return EXIT_SUCCESS;
}


long cti_numcod (int frame, int plane, int gate, int data, int bed)
{
#if 0
    switch (scanner) {
        case camRPT:    
            return ((frame & 0x1FF) | ((bed & 0xF) << 12) 
                    | ((plane & 0xFF) << 16) 
                    | (((plane >> 8) & 0x7) << 9) 
                    | ((gate & 0x3F) << 24) 
                    | ((data & 0x3) << 30) 
                    | (1 << 24));
            break;  
		
        default:
#endif
            return ((frame)|((bed&0xF)<<12)|((plane&0xFF)<<16)|(((plane&0x300)>>8)<<9)|
                    ((gate&0x3F)<<24)|((data&0x3)<<30)|((data&0x4)<<9));
#if 0
            break;
    }
#endif
}

void cti_numdoc (long matnum, Matval *matval)
{
#if 0
    switch (scanner) {
        case camRPT: // Same for both ECAT 953 RTS1 and ECAT 953 RTS2 
            matval->frame = matnum & 0x1FF;
            matval->plane = ((matnum >> 16) & 0xFF) + (((matnum >> 9) & 0x7) << 8);
            matval->gate  = (matnum >> 24) & 0x3F;
            matval->data  = (matnum >> 30) & 0x3;
            matval->bed   = (matnum >> 12) & 0xF;
            break;
		
        default:
#endif
            matval->frame = matnum&0x1FF;
            matval->plane = ((matnum>>16)&0xFF) + (((matnum>>9)&0x3)<<8);
            matval->gate  = (matnum>>24)&0x3F;
            matval->data  = ((matnum>>9)&0x4)|(matnum>>30)&0x3;
            matval->bed   = (matnum>>12)&0xF;
#if 0
            break;
    }
#endif
}

int cti_rings2plane (short nrings, short ring0, short ring1) 

{
    int d = (int) (ring0 / (nrings/2)); 

    return (ring1 * nrings/2 + ring0 % (nrings/2) +
            nrings/2 * nrings * d + 1);			 
}

int cti_rblk (FILE *fptr, int blkno, void *bufr, int nblks)
{
    int n, err;

    if (!fptr || !bufr) return EXIT_FAILURE;
   
    err = fseek (fptr, (long) (blkno - 1) * MatBLKSIZE, 0);
    if (err) 
    {
      // KT 11/03/2004 disabled diagnostics as sometimes we don't want to see it.
      // perror("cti_rblk: error in fseek");
      return (EXIT_FAILURE);
    }
   
    n = fread (bufr, sizeof (char), nblks * MatBLKSIZE, fptr);
    if (n != nblks * MatBLKSIZE) 
    {
      // KT 11/03/2004 disabled diagnostics as sometimes we don't want to see it.
      // perror("cti_rblk: error in fread");
      return (EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}

int cti_wblk (FILE *fptr, int blkno, void *bufr, int nblks)
{
    int err;

    if (!fptr || !bufr) return EXIT_FAILURE;

        // seek to position in file
    err = fseek (fptr, (long) (blkno - 1) * MatBLKSIZE, 0);
    if (err) return (EXIT_FAILURE);

    err = fwrite (bufr, sizeof (char), nblks * MatBLKSIZE, fptr);
    if (err != nblks * MatBLKSIZE) return EXIT_FAILURE;
		
    return EXIT_SUCCESS;
}

int cti_read_ECAT6_Main_header (FILE *fptr, ECAT6_Main_header *h)
{
    int status;

#if STIRIsNativeByteOrderBigEndian
  if(ByteOrder::get_native_order() != ByteOrder::big_endian)
#else
  if(ByteOrder::get_native_order() != ByteOrder::little_endian)
#endif
    error("Error in File %s: STIRIsNativeByteOrderBigEndian preprocessor define is determined incorrectly. Correct please. \n", __FILE__);

    short b[MatBLKSIZE];
    char * const bb = (char *) b;

        // read main header at block 1 into buf
    status = cti_rblk (fptr, 1,  bb, 1);

    if (status != EXIT_SUCCESS) 
      {
	return EXIT_FAILURE;
      }

        // copy all the strings
    strncpy (h->original_file_name, bb + 28, 20);
    strncpy (h->node_id, bb + 56, 10);
    strncpy (h->isotope_code, bb + 78, 8);
    strncpy (h->radiopharmaceutical, bb + 90, 32);
    strncpy (h->study_name, bb + 162, 12);
    strncpy (h->patient_id, bb + 174, 16);
    strncpy (h->patient_name, bb + 190, 32);
    h->patient_sex = bb [222];
    strncpy (h->patient_age, bb + 223, 10);
    strncpy (h->patient_height, bb + 233, 10);
    strncpy (h->patient_weight, bb + 243, 10);
    h->patient_dexterity = bb [253];
    strncpy (h->physician_name, bb + 254, 32);
    strncpy (h->operator_name, bb + 286, 32);
    strncpy (h->study_description, bb + 318, 32);
    strncpy (h->facility_name, bb + 356, 20);
    strncpy (h->user_process_code, bb + 462, 10);

#if STIRIsNativeByteOrderBigEndian // we have to swap bytes in order to read ints and floats */
    swab ((char *) b, (char *) b, MatBLKSIZE);
#endif
    h->sw_version = b [24];
    h->data_type = b [25];
    h->system_type = b [26];
    h->file_type = b [27];
    h->scan_start_day = b [33];
    h->scan_start_month = b [34];
    h->scan_start_year = b [35];
    h->scan_start_hour = b [36];
    h->scan_start_minute = b [37];
    h->scan_start_second = b [38];
    h->isotope_halflife = get_vax_float ((unsigned short *) b, 43);
    h->gantry_tilt = get_vax_float ((unsigned short *) b, 61);
    h->gantry_rotation = get_vax_float ((unsigned short *) b, 63);
    h->bed_elevation = get_vax_float ((unsigned short *) b, 65);
    h->rot_source_speed = b [67];
    h->wobble_speed = b [68];
    h->transm_source_type = b [69];
    h->axial_fov = get_vax_float ((unsigned short *) b, 70);
    h->transaxial_fov = get_vax_float ((unsigned short *) b, 72);
    h->transaxial_samp_mode = b [74];
    h->coin_samp_mode = b [75];
    h->axial_samp_mode = b [76];
    h->calibration_factor = get_vax_float ((unsigned short *) b, 77);
    h->calibration_units = b [79];
    h->compression_code = b [80];
    h->acquisition_type = b [175];
    h->bed_type = b [176];
    h->septa_type = b [177];
    h->num_planes = b [188];
    h->num_frames = b [189];
    h->num_gates = b [190];
    h->num_bed_pos = b [191];
    h->init_bed_position = get_vax_float ((unsigned short *) b, 192);
    for (int i=0; i<15; i++)
	h->bed_offset [i] = get_vax_float ((unsigned short *) b, 194 + 2 * i);

    h->plane_separation = get_vax_float ((unsigned short *) b, 224);
    h->lwr_sctr_thres = b [226];
    h->lwr_true_thres = b [227];
    h->upr_true_thres = b [228];
    h->collimator = get_vax_float ((unsigned short *) b, 229);

    return EXIT_SUCCESS;
}

int cti_read_scan_subheader (FILE *fptr, int blknum, Scan_subheader *h)
{
    int status ;

    short b[MatBLKSIZE];

    status = cti_rblk (fptr, blknum, (char *) b, 1);   // read the block
    if (status != EXIT_SUCCESS) {
	return (EXIT_FAILURE);
    }

#if STIRIsNativeByteOrderBigEndian // we have to swap bytes in order to read the ints
    swab ((char *) b, (char *) b, MatBLKSIZE);
#endif

    h->data_type = b [63];
    h->dimension_1 = b [66];
    h->dimension_2 = b [67];
    h->smoothing = b [68];
    h->processing_code = b [69];
    h->sample_distance = get_vax_float ((unsigned short *) b, 73);
    h->isotope_halflife = get_vax_float ((unsigned short *) b, 83);
    h->frame_duration_sec = b [85];
    h->gate_duration = get_vax_long ((unsigned short *) b, 86);
    h->r_wave_offset = get_vax_long ((unsigned short *) b, 88);
    h->scale_factor = get_vax_float ((unsigned short *) b, 91);
    h->scan_min = b [96];
    h->scan_max = b [97];
    h->prompts = get_vax_long ((unsigned short *) b, 98);
    h->delayed = get_vax_long ((unsigned short *) b, 100);
    h->multiples = get_vax_long ((unsigned short *) b, 102);
    h->net_trues = get_vax_long ((unsigned short *) b, 104);
    for (int i=0; i<16; i++) {
	h->cor_singles [i] = get_vax_float ((unsigned short *) b, 158 + 2 * i);
	h->uncor_singles [i] = get_vax_float ((unsigned short *) b, 190 + 2 * i);
    }
    h->tot_avg_cor = get_vax_float ((unsigned short *) b, 222);
    h->tot_avg_uncor = get_vax_float ((unsigned short *) b, 224);
    h->total_coin_rate = get_vax_long ((unsigned short *) b, 226);
    h->frame_start_time = get_vax_long ((unsigned short *) b, 228);
    h->frame_duration = get_vax_long ((unsigned short *) b, 230);
    h->loss_correction_fctr = get_vax_float ((unsigned short *) b, 232);
    return EXIT_SUCCESS;
}


int cti_read_attn_subheader(FILE* fptr, int blknum, Attn_subheader *header)
{
  short int bufr[256];
  int  err;
  
  err = cti_rblk( fptr, blknum, bufr, 1);
  if (err) return(err);
  
#if STIRIsNativeByteOrderBigEndian // we have to swap bytes in order to read the ints
  swab ((char *) bufr, (char *) bufr, MatBLKSIZE);
#endif
  
  header->data_type = bufr[63];
  header->attenuation_type = bufr[64];
  header->dimension_1 = bufr[66];
  header->dimension_2 = bufr[67];
  header->scale_factor = get_vax_float((unsigned short *)bufr, 91);
  header->x_origin = get_vax_float((unsigned short *)bufr, 93);
  header->y_origin = get_vax_float((unsigned short *)bufr, 95);
  header->x_radius = get_vax_float((unsigned short *)bufr, 97);
  header->y_radius = get_vax_float((unsigned short *)bufr, 99);
  header->tilt_angle = get_vax_float((unsigned short *)bufr, 101);
  header->attenuation_coeff = get_vax_float((unsigned short *)bufr, 103);
  header->sample_distance = get_vax_float((unsigned short *)bufr, 105);
  return EXIT_SUCCESS;
}


int cti_read_norm_subheader(FILE* fptr, int blknum, Norm_subheader *header)
{
  short int bufr[256];
  int  err;
  
  err = cti_rblk( fptr, blknum, bufr, 1);
  if (err) return(err);
  
#if STIRIsNativeByteOrderBigEndian // we have to swap bytes in order to read the ints
  swab ((char *) bufr, (char *) bufr, MatBLKSIZE);
#endif
  
  header->data_type = bufr[63];
  header->dimension_1 = bufr[66];
  header->dimension_2 = bufr[67];
  header->scale_factor = get_vax_float((unsigned short *)bufr, 91);
  header->norm_hour = bufr[93];
  header->norm_minute = bufr[94];
  header->norm_second = bufr[95];
  header->norm_day = bufr[96];
  header->norm_month = bufr[97];
  header->norm_year = bufr[98];
  header->fov_source_width = get_vax_float((unsigned short *)bufr, 99);

  return EXIT_SUCCESS;
}

int cti_read_image_subheader (FILE *fptr, int blknum, Image_subheader *ihead)
{
    int status;
    char  *bb;

        // alloc buffer
    short b[MatBLKSIZE];

        // read block into buffer
    status = cti_rblk (fptr, blknum, (char *) b, 1);   // read the block
    if (status != EXIT_SUCCESS) {
	return (EXIT_FAILURE);
    }

    bb = (char *) b;
    strncpy (ihead->annotation, bb + 420, 40);

#if STIRIsNativeByteOrderBigEndian // we have to swap bytes in order to read the ints
    swab ((char *) b, (char *) b, MatBLKSIZE);
#endif
	
        // fill in the image subheader
    ihead->x_origin = get_vax_float ((unsigned short *)b, 80);
    ihead->y_origin = get_vax_float ((unsigned short *)b, 82);
    ihead->recon_scale = get_vax_float ((unsigned short *)b, 84);
    ihead->quant_scale = get_vax_float ((unsigned short *)b, 86);
    ihead->pixel_size = get_vax_float ((unsigned short *)b, 92);
    ihead->slice_width = get_vax_float ((unsigned short *)b, 94);
    
    ihead->data_type = b[63];
    ihead->num_dimensions = b[64];
    ihead->dimension_1 = b[66];
    ihead->dimension_2 = b[67];
    ihead->image_min = b[88];
    ihead->image_max = b[89];
    
    ihead->image_rotation = get_vax_float ((unsigned short *)b, 148); 
    ihead->plane_eff_corr_fctr = get_vax_float ((unsigned short *)b, 150);
    ihead->decay_corr_fctr = get_vax_float ((unsigned short *)b, 152);
    ihead->loss_corr_fctr = get_vax_float ((unsigned short *)b, 154);
    ihead->ecat_calibration_fctr = get_vax_float ((unsigned short *)b, 194);
    ihead->well_counter_cal_fctr = get_vax_float ((unsigned short *)b, 196);
    for (int i=0; i<6; i++)
        ihead->filter_params[i] = get_vax_float ((unsigned short *)b, 198 + 2 * i);
    
    ihead->frame_duration = get_vax_long ((unsigned short *)b, 104);
    ihead->frame_start_time = get_vax_long ((unsigned short *)b, 98);
    ihead->recon_duration = get_vax_long ((unsigned short *)b, 104);
    ihead->scan_matrix_num = get_vax_long ((unsigned short *)b, 119);
    ihead->norm_matrix_num = get_vax_long ((unsigned short *)b, 121);
    ihead->atten_cor_matrix_num = get_vax_long ((unsigned short *)b, 123);
    
    ihead->slice_location = b [100];
    ihead->recon_start_hour = b [101];
    ihead->recon_start_minute = b [102];
    ihead->recon_start_sec = b [103];
    
    ihead->filter_code = b [118];
    ihead->processing_code = b [188];
    ihead->quant_units = b [190];
    
    ihead->recon_start_day = b [191];
    ihead->recon_start_month = b [192];
    ihead->recon_start_year = b [193];

    return (EXIT_SUCCESS);
}

BOOST_STATIC_ASSERT(sizeof(boost::int32_t)==4);

FILE *cti_create (const char *fname, const ECAT6_Main_header *mhead)
{
    FILE *fptr;
    int status;
    boost::int32_t *bufr;

        // open the file and write the header into it.
    fptr = fopen (fname, "wb+");
    if (!fptr) return fptr;

    status = cti_write_ECAT6_Main_header (fptr, mhead);
    if (status != EXIT_SUCCESS) {
	fclose (fptr);
	return NULL;
    }
	
        // create a First Directory Block in the file
    bufr = (boost::int32_t *) calloc (MatBLKSIZE / sizeof (boost::int32_t), sizeof (boost::int32_t));
    if (!bufr) {
	fclose (fptr);
	return NULL;
    }

    bufr [0] = 31;          // mystery number
    bufr [1] = 2;           // next block
	
#if STIRIsNativeByteOrderBigEndian // we must do some swapping about */
    swaw ((short *) bufr, (short *) bufr, MatBLKSIZE/2);
    swab ((char *) bufr, (char *) bufr, MatBLKSIZE);
#endif

    status = cti_wblk (fptr, MatFirstDirBlk, (char *) bufr, 1);
    if (status != EXIT_SUCCESS) {
        fclose (fptr);
	free (bufr);
	return NULL;
    }
    free (bufr);
    return (fptr);
}

int cti_enter (FILE *fptr, long matnum, int nblks)
{
    int i, dirblk, nxtblk, busy, oldsize;
    boost::int32_t *dirbufr;           // buffer for directory block
    int status;

        // set up buffer for directory block
    dirbufr = (boost::int32_t *) calloc (MatBLKSIZE / sizeof (boost::int32_t), sizeof (boost::int32_t));
    if (!dirbufr) return 0;

        // read first directory block from file
    dirblk = MatFirstDirBlk;
    
    status = cti_rblk (fptr, dirblk, (char *) dirbufr, 1);
    if (status != EXIT_SUCCESS) {
        free (dirbufr);
	return 0;
    }

#if STIRIsNativeByteOrderBigEndian
    swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
    swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE / 2);
#endif

    status = EXIT_SUCCESS;
    busy = 1;

        // search for our matnum in directory blocks
    while (busy && status == EXIT_SUCCESS) {
        nxtblk = dirblk + 1;
    
            // see if matnum entry is in this block
        // KT added unsigned to avoid compiler warnings
        for (i=4; (unsigned)i<MatBLKSIZE / sizeof (boost::int32_t); i+=sizeof (boost::int32_t)) {
	    if (dirbufr [i] == 0) { // skip to next block
		busy = 0;
		break;
	    }
	    else if (dirbufr [i] == matnum) {     // found it
                    // see if this entry has reserved enough space for us
                    // see if there's enough room
		oldsize = dirbufr [i + 2] - dirbufr [i + 1] + 1;
		if (oldsize < nblks) { // delete old entry and create new one
                    dirbufr [i] = 0xFFFFFFFF;
		    
#if STIRIsNativeByteOrderBigEndian
		    swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE / sizeof (short));
		    swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
#endif
		    
                    status = cti_wblk (fptr, dirblk, (char *) dirbufr, 1);
		    
#if STIRIsNativeByteOrderBigEndian
		    swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
		    swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE / sizeof (short));
#endif
		    nxtblk = dirbufr [i + 2] + 1;
		}
		else {    // enough room here
                        // get pointer to next block from entry
		    nxtblk = dirbufr [i + 1];
		    dirbufr [0] ++;
		    dirbufr [3] --;
		    busy = 0;
		    break;
		}
	    }
	    else nxtblk = dirbufr [i + 2] + 1;
	} // end of i loop
    
	if (!busy) break;  // hit end of block, or found it
    
	if (dirbufr [1] != MatFirstDirBlk) { // hop to next block
	    dirblk = dirbufr [1];
	    status = cti_rblk (fptr, dirblk, (char *) dirbufr, 1);
	    if (status != EXIT_SUCCESS) {
		status = EXIT_FAILURE;     // get out
		break;
	    }
#if STIRIsNativeByteOrderBigEndian
	    swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
	    swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE / 2);
#endif
	} 
        else { // add a new block
	    
                // modify this block to point to next block
	    dirbufr [1] = nxtblk;
	    
                // do some swapping for good measure
#if STIRIsNativeByteOrderBigEndian
	    swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE/2);
	    swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
#endif
	    
            status = cti_wblk (fptr, dirblk, (char *) dirbufr, 1);
	    
                // prepare new directory block
	    dirbufr [0] = 31;
	    dirbufr [1] = MatFirstDirBlk;
	    dirbufr [2] = dirblk;
	    dirbufr [3] = 0;
	    dirblk = nxtblk;
	    for (i=4; i<MatBLKSIZE / 4; i++) dirbufr [i] = 0;
	}
    } // end of busy loop
    
    if (status == EXIT_SUCCESS) { // add new entry
        dirbufr [i] = matnum;
        dirbufr [i+1] = nxtblk;
        dirbufr [i+2] = nxtblk + nblks;
        dirbufr [i+3] = 1;
        dirbufr [0] --;
        dirbufr [3] ++;
	 
#if STIRIsNativeByteOrderBigEndian
        swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE/2);
        swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
#endif
	 
            // write to directory block
        cti_wblk (fptr, dirblk, (char *) dirbufr, 1);
    }

    fflush (fptr);
    free (dirbufr);

    if (status != EXIT_SUCCESS) return 0;
    return (nxtblk);
}

int cti_lookup (FILE *fptr, long matnum, MatDir *entry)
{
    int blk, status;
    int nfree, nxtblk, prvblk, nused, matnbr, strtblk, endblk, matstat;
    boost::int32_t *dirbufr;

        // set up buffer for directory block
    
    dirbufr = (boost::int32_t *) malloc (MatBLKSIZE);
    if (!dirbufr) return 0;

    blk = MatFirstDirBlk;
    status = EXIT_SUCCESS;
    while (status == EXIT_SUCCESS) { // look through the blocks in the file
            // read a block and examine the matrix numbers in it
	status = cti_rblk (fptr, blk, dirbufr, 1);
	if (status != EXIT_SUCCESS) break;
#if STIRIsNativeByteOrderBigEndian   // read into byte buffer and swap
        swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
        swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE / sizeof(short));
#endif
	
            // get directory block info
	nfree  = dirbufr [0];
	nxtblk = dirbufr [1];
	prvblk = dirbufr [2];
	nused  = dirbufr [3];

            // look through the entries in this block
        // KT added unsigned to avoid compiler warnings
	for (int i=4; (unsigned)i<MatBLKSIZE / sizeof (boost::int32_t); i+=sizeof (boost::int32_t)) {
	    matnbr  = dirbufr [i];
	    strtblk = dirbufr [i + 1];
	    endblk  = dirbufr [i + 2];
	    matstat = dirbufr [i + 3];
	    if (matnum == matnbr) { // got our entry
		entry->matnum  = matnbr;
		entry->strtblk = strtblk;
		entry->endblk  = endblk;
		entry->matstat = matstat;
		free (dirbufr);
		return 1;     // we were successful
	    }
        }

        blk = nxtblk;       // point to next block
	if (blk <= MatFirstDirBlk) break;
    }
    free (dirbufr);
    return 0;       // we were unsuccessful
}

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

int cti_write_image_subheader (FILE *fptr, int blknum, const Image_subheader *header)
{
    int status;
    char *bbufr;
    short bufr[MatBLKSIZE / sizeof (short)];

    bbufr = (char *) bufr;
    // transfer subheader information
    bufr [63] = header->data_type;
    bufr [64] = header->num_dimensions;
    bufr [66] = header->dimension_1;
    bufr [67] = header->dimension_2;
    hostftovaxf (header->x_origin, (unsigned short *) &bufr [80]);
    hostftovaxf (header->y_origin, (unsigned short *) &bufr [82]);
    hostftovaxf (header->recon_scale, (unsigned short *) &bufr [84]);
    hostftovaxf (header->quant_scale, (unsigned short *) &bufr [86]);
    bufr [88] = header->image_min;
    bufr [89] = header->image_max;
    hostftovaxf (header->pixel_size, (unsigned short *) &bufr [92]);
    hostftovaxf (header->slice_width, (unsigned short *) &bufr [94]);
    hostltovaxl (header->frame_duration, (unsigned short *) &bufr [96]);
    hostltovaxl (header->frame_start_time, (unsigned short *) &bufr [98]);
    bufr [100] = header->slice_location;
    bufr [101] = header->recon_start_hour;
    bufr [102] = header->recon_start_minute;
    bufr [103] = header->recon_start_sec;
    hostltovaxl (header->recon_duration, (unsigned short *) &bufr [104]);
    bufr [118] = header->filter_code;
    hostltovaxl (header->scan_matrix_num, (unsigned short *) &bufr [119]);
    hostltovaxl (header->norm_matrix_num, (unsigned short *) &bufr [121]);
    hostltovaxl (header->atten_cor_matrix_num, (unsigned short *) &bufr [123]);
    hostftovaxf (header->image_rotation, (unsigned short *) &bufr [148]);
    hostftovaxf (header->plane_eff_corr_fctr, (unsigned short *) &bufr [150]);
    hostftovaxf (header->decay_corr_fctr, (unsigned short *) &bufr [152]);
    hostftovaxf (header->loss_corr_fctr, (unsigned short *) &bufr [154]);
    bufr [188] = header->processing_code;
    bufr [190] = header->quant_units;
    bufr [191] = header->recon_start_day;
    bufr [192] = header->recon_start_month;
    bufr [193] = header->recon_start_year;
    hostftovaxf (header->ecat_calibration_fctr, (unsigned short *) &bufr [194]);
    hostftovaxf (header->well_counter_cal_fctr, (unsigned short *) &bufr [196]);

    for (int i=0; i<6; i++)
        hostftovaxf (header->filter_params [i], (unsigned short *) &bufr [198+2*i]);

#if STIRIsNativeByteOrderBigEndian
    swab ((char *) bufr, (char *) bufr, MatBLKSIZE);
#endif

    bcopy (header->annotation, bbufr + 420, 40);
    // write to matrix file

    status = cti_wblk (fptr, blknum, bbufr, 1);
    fflush(fptr);
    if (status != EXIT_SUCCESS) return (EXIT_FAILURE);
    return (EXIT_SUCCESS);
}

int cti_write_ECAT6_Main_header (FILE *fptr, const ECAT6_Main_header *header)
{
    char *bbufr;
    int status;

    short bufr[MatBLKSIZE / sizeof (short)];
    bbufr = (char *) bufr;

    bufr [24] = header->sw_version;
    bufr [25] = header->data_type;
    bufr [26] = header->system_type;
    bufr [27] = header->file_type;
    bufr [33] = header->scan_start_day;
    bufr [34] = header->scan_start_month;
    bufr [35] = header->scan_start_year;
    bufr [36] = header->scan_start_hour;
    bufr [37] = header->scan_start_minute;
    bufr [38] = header->scan_start_second;
    hostftovaxf (header->isotope_halflife, (unsigned short *) &bufr [43]);
    hostftovaxf (header->gantry_tilt, (unsigned short *) &bufr [61]);
    hostftovaxf (header->gantry_rotation, (unsigned short *) &bufr [63]);
    hostftovaxf (header->bed_elevation, (unsigned short *) &bufr [65]);
    bufr [67] = header->rot_source_speed;
    bufr [68] = header->wobble_speed;
    bufr [69] = header->transm_source_type;
    hostftovaxf (header->axial_fov, (unsigned short *) &bufr [70]);
    hostftovaxf (header->transaxial_fov, (unsigned short *) &bufr [72]);
    bufr [74] = header->transaxial_samp_mode;
    bufr [75] = header->coin_samp_mode;
    bufr [76] = header->axial_samp_mode;
    hostftovaxf (header->calibration_factor, (unsigned short *) &bufr [77]);
    bufr [79] = header->calibration_units;
    bufr [80] = header->compression_code;
    bufr [175] = header->acquisition_type;
    bufr [176] = header->bed_type;
    bufr [177] = header->septa_type;
    bufr [188] = header->num_planes;
    bufr [189] = header->num_frames;
    bufr [190] = header->num_gates;
    bufr [191] = header->num_bed_pos;
    hostftovaxf (header->init_bed_position, (unsigned short *) &bufr [192]);
    for (int i=0; i<15; i ++)
	hostftovaxf (header->bed_offset [i], (unsigned short *) &bufr [194 + 2 * i]);
    hostftovaxf (header->plane_separation, (unsigned short *) &bufr [224]);
    bufr [226] = header->lwr_sctr_thres;
    bufr [227] = header->lwr_true_thres;
    bufr [228] = header->upr_true_thres;
    hostftovaxf (header->collimator, (unsigned short *) &bufr [229]);

#if STIRIsNativeByteOrderBigEndian
    swab ((char *) bufr, (char *) bufr, MatBLKSIZE);
#endif

    bcopy (header->original_file_name, bbufr + 28, 20);
    bcopy (header->node_id, bbufr + 56, 10);
    bcopy (header->isotope_code, bbufr + 78, 8);
    bcopy (header->radiopharmaceutical, bbufr + 90, 32);
    bcopy (header->study_name, bbufr + 162, 12);
    bcopy (header->patient_id, bbufr + 174, 16);
    bcopy (header->patient_name, bbufr + 190, 32);
    bbufr [222] = header->patient_sex;
    bcopy (header->patient_age, bbufr + 223, 10);
    bcopy (header->patient_height, bbufr + 233, 10);
    bcopy (header->patient_weight, bbufr + 243, 10);
    bbufr [253] = header->patient_dexterity;
    bcopy (header->physician_name, bbufr + 254, 32);
    bcopy (header->operator_name, bbufr + 286, 32);
    bcopy (header->study_description, bbufr + 318, 32);
    bcopy (header->facility_name, bbufr + 356, 20);
    bcopy (header->user_process_code, bbufr + 462, 10);
    
    // write main header at block 1    
    //warning("KTXXX writing mainheader from %x",bbufr);
    status = cti_wblk (fptr, 1, (char *) bbufr, 1);
    //warning("KTXXXflush");
    /* warning: valgrind 2.2.0 on debian 'woody' gives an error in the
       next system call:

       Syscall param write(buf) contains uninitialised or unaddressable byte(s)
       at 0x1BBADBF4: write (in /lib/libc-2.2.5.so)
       by 0x1BB54AB7: (within /lib/libc-2.2.5.so)
       by 0x1BB54A15: _IO_do_write (in /lib/libc-2.2.5.so)
       by 0x1BB54E3E: _IO_file_sync (in /lib/libc-2.2.5.so)
       Address 0x1B908000 is not stack'd, malloc'd or (recently) free'd
      KT does not know where this comes from.
    */
    fflush (fptr);
    //warning("KTXXXdone");
    if (status != EXIT_SUCCESS) return (status);
    else return (EXIT_SUCCESS);
}

int cti_write_scan_subheader (FILE *fptr, int blknum, const Scan_subheader *header)
{
    int status;
    short bufr[MatBLKSIZE / sizeof (short)];

        // fill in bufr
    bufr[0] = 256;
    bufr[1] = 1;
    bufr[2] = 22;
    bufr[3] = -1;
    bufr[4] = 25;
    bufr[5] = 62;
    bufr[6] = 79;
    bufr[7] = 106;
    bufr[24] = 37;
    bufr[25] = -1;
    bufr[61] = 17;
    bufr[62] = -1;
    bufr[78] = 27;
    bufr[79] = -1;
    bufr[105] = 52;
    bufr[106] = -1;
    bufr[63] = header->data_type;
    bufr[66] = header->dimension_1;         // x dimension
    bufr[67] = header->dimension_2;         // y_dimension
    bufr[68] = header->smoothing;
    bufr[69] = header->processing_code;
    hostftovaxf (header->sample_distance, (unsigned short *) &bufr[73]);
    hostftovaxf (header->isotope_halflife, (unsigned short *) &bufr[83]);
    bufr[85] = header->frame_duration_sec;
    hostltovaxl (header->gate_duration, (unsigned short *) &bufr[86]);
    hostltovaxl (header->r_wave_offset, (unsigned short *) &bufr[88]);
    hostftovaxf (header->scale_factor, (unsigned short *) &bufr[91]);
    bufr[96] = header->scan_min;
    bufr[97] = header->scan_max;
    hostltovaxl (header->prompts, (unsigned short *) &bufr[98]);
    hostltovaxl (header->delayed, (unsigned short *) &bufr[100]);
    hostltovaxl (header->multiples, (unsigned short *) &bufr[102]);
    hostltovaxl (header->net_trues, (unsigned short *) &bufr[104]);
    for (int i=0; i<16; i++) {
        hostftovaxf (header->cor_singles [i], (unsigned short *) &bufr [158 + 2 * i]);
        hostftovaxf (header->uncor_singles [i], (unsigned short *) &bufr [190 + 2 * i]);
    }
    hostftovaxf (header->tot_avg_cor, (unsigned short *) &bufr[222]);
    hostftovaxf (header->tot_avg_uncor, (unsigned short *) &bufr[224]);
    hostltovaxl (header->total_coin_rate, (unsigned short *) &bufr[226]);
    hostltovaxl (header->frame_start_time, (unsigned short *) &bufr[228]);
    hostltovaxl (header->frame_duration,(unsigned short *)  &bufr[230]);
    hostftovaxf (header->loss_correction_fctr, (unsigned short *) &bufr[232]);

#if STIRIsNativeByteOrderBigEndian
    swab ((char *) bufr, (char *) bufr, MatBLKSIZE);
#endif

    status = cti_wblk (fptr, blknum, (char *) bufr, 1);

    return status;
}

int cti_write_image (FILE *fptr, long matnum, const Image_subheader *header,
                     const short *data, int data_size)
{
    int nxtblk, nblocks;
    int status;

    nblocks = (data_size + (MatBLKSIZE - 1)) / MatBLKSIZE;
    nxtblk = cti_enter (fptr, matnum, nblocks);
    if (nxtblk <= 0) return (EXIT_FAILURE);

    status = cti_write_image_subheader (fptr, nxtblk, header);
    if (status != EXIT_SUCCESS) return (EXIT_FAILURE);
   
    status = cti_write_idata (fptr, nxtblk + 1, data, data_size);
    return status;
}

int cti_write_scan (FILE *fptr, long matnum, const Scan_subheader *header,
		    const short *data, int data_size)
{
    int nxtblk, nblocks;
    int status;

    nblocks = (data_size + (MatBLKSIZE - 1)) / MatBLKSIZE;
    nxtblk = cti_enter (fptr, matnum, nblocks);
    if (nxtblk <= 0) return (EXIT_FAILURE);

    status = cti_write_scan_subheader (fptr, nxtblk, header);
    if (status != EXIT_SUCCESS) return (EXIT_FAILURE);

    status = cti_write_idata (fptr, nxtblk + 1, data, data_size);
    return status;
}

void sfind_minmax (short *buf, short *min, short *max, int bufsize)
{
    register short  *b, foundmax, foundmin;
	
    foundmax = SHRT_MIN;
    foundmin = SHRT_MAX;
    b = buf;
    for (int i=0; i<bufsize; i++) {
        if (*b > foundmax) foundmax = *b;
        if (*b < foundmin) foundmin = *b;
        ++b;
    }
    *min = foundmin;
    *max = foundmax;
}

void ffind_minmax (float *buf, float *min, float *max, int count)
{
    float foundmax, foundmin, *b;
	
    foundmax = FLT_MIN;
    foundmin = FLT_MAX;
    b = buf;

    for (int i=0; i<count; i++, b++) {
        if (*b > foundmax) foundmax = *b;
        if (*b < foundmin) foundmin = *b;
    }
    *min = foundmin;
    *max = foundmax;
}

void swab (char *from, char *to, int length)
{
    register char temp;

    for(int i=0; i<length; i+= 2) 
    {
	temp = from [i + 1];
	to [i + 1] = from [i];
	to [i] = temp;
    }
}

void swaw (short *from, short *to, int length)
{
    register short temp;

    for (int i=0; i<length; i+=2) {
	temp = from [i + 1];
	to [i + 1] = from [i];
	to [i] = temp;
    }
}

/******************* conversions from/to VAX floats to host floats ***************/
/* rewritten by Kris Thielemans */

#ifdef VAX
// conversions are trivial
typedef float VAXfloat;
float VAXfl_to_fl(VAXfloat Va) { return Va; }
VAXfloat fl_to_VAXfl(float a) { return a; }

#else

#if STIRIsNativeByteOrderBigEndian
/* definition for bigendian machines.
   Do swab, swaw first before using this bit field.
*/
typedef struct
        {
          unsigned frc2 : 16;
          unsigned sign : 1;
          unsigned exp  : 8;
          unsigned frc1 : 7;
        } VAXfloat;
#else
/* definition for littleendian machines. */
typedef struct
        { unsigned frc1 : 7;
          unsigned exp  : 8;
          unsigned sign : 1;
          unsigned frc2 : 16;
        } VAXfloat;
#endif

BOOST_STATIC_ASSERT(sizeof(VAXfloat)==4);

/* routines for converting VAX floating point format into own format.
   Code is in a generic form that should work on all machines
   (it also works on VAX).
   This might be slower than bit-manipulations, but ldexp() et al are probably
   written in similar bit-manipulations anyway.

   The code has been tested on Decstation, PC, SUN and VAX.
*/
float VAXfl_to_fl(VAXfloat Va)
{ int sign;

  if (Va.sign)
    sign = -1;
  else
  { if (Va.exp==0)
      return(0.0);
    sign = 1;
  }
  return (float)(sign*ldexp((double)((0x800000 | Va.frc2) + (Va.frc1 * 0x10000)),
                      Va.exp-128-24));
}

VAXfloat fl_to_VAXfl(float a)
{ unsigned long imant;
  double dmant;
  int exp;
  VAXfloat Va;

  if (a==0)
  { Va.sign = Va.exp = 0;
    /* set frc to 0, this is not necessary , 
       but it's easier to check consistency */
    Va.frc1 = 0;
    Va.frc2 = 0;
    return (Va);
  }
  if (a>0.0)
    Va.sign = 0;
  else
  { Va.sign = 1;
    a = -a;
  }
  dmant = frexp(a,&exp);
  if (exp<-127)
  { Va.sign = Va.exp = 0;
    return (Va);
  }
  if (exp>127)
    warning("Floating point number %g too big for VAX format: I'll return garbage\n",a);

  Va.exp = exp + 128;
  imant = (unsigned long)0x7fffff & (unsigned long)ldexp(dmant,24);
  /* Compiler can give "data conversion" warnings on the following,
     but it's OK */
  Va.frc1 = (unsigned)(imant >>16);
  Va.frc2 = (unsigned)(imant & 0xffff);

  return(Va);
}

#endif /* no VAX */

float get_vax_float (const unsigned short *bufr, int off)
{
#ifdef VAX
  float Va;
  memcpy(&Va, bufr+off, sizeof(float));
  return Va;
#else

#if STIRIsNativeByteOrderBigEndian
  unsigned short int tmpbufr[2];
  tmpbufr[0]=bufr[off+1];
  tmpbufr[1]=bufr[off];
  
  VAXfloat Va;
  memcpy(&Va, tmpbufr, sizeof(VAXfloat));
  return VAXfl_to_fl(Va);  
# else
  VAXfloat Va;
  memcpy(&Va, bufr+off, sizeof(VAXfloat));
  return VAXfl_to_fl(Va);
# endif	  
#endif /* not VAX */

}

void hostftovaxf (const float in, unsigned short out [2])
{
  const VAXfloat tmp = fl_to_VAXfl(in);
  memcpy(out, &tmp, 4);
#if STIRIsNativeByteOrderBigEndian         
  std::swap(out[0], out[1]);
  // swab is necessary by caller
#endif
}



/*******************************************************************************
	get_vax_long - get the indexed value from a buffer, a 32-bit vax long, and
		convert it by swapping the words.
		(vax int = vax long int = 32 bits; vax short = 16 bits)
	
	bufr - input data buffer.
	off - index into buffer of first 16-bit word of the 32-bit value to convert.
*******************************************************************************/
BOOST_STATIC_ASSERT(sizeof(long)>=4);

long get_vax_long (const unsigned short *bufr, int off)
{
#if STIRIsNativeByteOrderBigEndian
	return ((static_cast<long>(bufr [off + 1]) << 16) + bufr [off]);
#else
	return ((static_cast<long>(bufr [off]) << 16) + bufr [off+1]);
#endif
}
/*******************************************************************************
	hostltovaxl - convert a sun long int to a vax long int -- i.e. swap the
		16-bit words of the 32-bit long.
		(sun long = sun int = 32 bits)
	
	in - value to convert.
	out - result.
*******************************************************************************/
void hostltovaxl (const long in, unsigned short out [2])
{  
#if STIRIsNativeByteOrderBigEndian
	out [0] = static_cast<unsigned short>(in & 0x0000FFFF);
	out [1] = static_cast<unsigned short>((in & 0xFFFF0000) >> 16);
#else
	out[0] = static_cast<unsigned short>((in & 0xFFFF0000) >> 16);
        out [1] = static_cast<unsigned short>((in & 0x0000FFFF));
#endif
}


void dump_ECAT6_Main_header (FILE *fptr, const ECAT6_Main_header *mhead)
{
    FILE *dptr;
	
    if (!fptr) {
        
        dptr = fopen ("mainheader.dmp", "w");
        if (!dptr)
	  {
	    printf("Dump to mainheader.dmp failed: couldn't open file\n");
	    return;
	  }
        printf ("dumping to mainheader.dmp\n");
    } 
    else dptr = fptr;
	
	// print out all the fields 
    fprintf (dptr, "    MAIN HEADER\n    -----------\n");
    fprintf (dptr, "isotope_halflife = %g\n", mhead->isotope_halflife);
    fprintf (dptr, "gantry_tilt = %g\n", mhead->gantry_tilt);
    fprintf (dptr, "gantry_rotation = %g\n", mhead->gantry_rotation);
    fprintf (dptr, "bed_elevation = %g\n", mhead->bed_elevation);
    fprintf (dptr, "axial_fov = %g\t", mhead->axial_fov);
    fprintf (dptr, "transaxial_fov = %g\n", mhead->transaxial_fov);
    fprintf (dptr, "calibration_factor = %g\n", mhead->calibration_factor);
    fprintf (dptr, "init_bed_position = %g\n", mhead->init_bed_position);
    fprintf (dptr, "plane_separation = %g\n", mhead->plane_separation);
    fprintf (dptr, "collimator = %g\n", mhead->collimator);
    for (int i=0; i<15; i++)
        fprintf (dptr, "bed_offset[%d] = %g\n", i, mhead->bed_offset[i]);

    fprintf (dptr, "num_planes = %d\n", mhead->num_planes);
    fprintf (dptr, "num_frames = %d\n", mhead->num_frames);
    fprintf (dptr, "num_gates = %d \n", mhead->num_gates);
    fprintf (dptr, "num_bed_pos = %d\n", mhead->num_bed_pos);
	
    fprintf (dptr, "sw_version = %d\n", mhead->sw_version);
    fprintf (dptr, "data_type = %d\n", mhead->data_type);
    fprintf (dptr, "system_type = %d\n", mhead->system_type);
    fprintf (dptr, "file_type = %d\n", mhead->file_type);
	
    fprintf (dptr, "scan_start_day = %d\n", mhead->scan_start_day);
    fprintf (dptr, "scan_start_month = %d\n", mhead->scan_start_month);
    fprintf (dptr, "scan_start_year = %d\n", mhead->scan_start_year);
    fprintf (dptr, "scan_start_hour = %d\n", mhead->scan_start_hour);
    fprintf (dptr, "scan_start_minute = %d\n", mhead->scan_start_minute);
    fprintf (dptr, "scan_start_second = %d\n", mhead->scan_start_second);

    fprintf (dptr, "rot_source_speed = %d\n", mhead->rot_source_speed);
    fprintf (dptr, "wobble_speed = %d\n", mhead->wobble_speed);
    fprintf (dptr, "transm_source_type = %d\n", mhead->transm_source_type);
    fprintf (dptr, "transaxial_samp_mode = %d\n", mhead->transaxial_samp_mode);
    fprintf (dptr, "coin_samp_mode = %d\n", mhead->coin_samp_mode);
    fprintf (dptr, "axial_samp_mode = %d\n", mhead->axial_samp_mode);
    fprintf (dptr, "calibration_units = %d\n", mhead->calibration_units);
    fprintf (dptr, "compression_code = %d\n", mhead->compression_code);
    fprintf (dptr, "acquisition_type = %d\n", mhead->acquisition_type);
    fprintf (dptr, "bed_type = %d\n", mhead->bed_type);
    fprintf (dptr, "septa_type = %d\n", mhead->septa_type);
    fprintf (dptr, "lwr_sctr_thres = %d\n", mhead->lwr_sctr_thres);
    fprintf (dptr, "lwr_true_thres = %d\n", mhead->lwr_true_thres);
    fprintf (dptr, "upr_true_thres = %d\n", mhead->upr_true_thres);
	
    fprintf (dptr, "original_file_name: %s\n", mhead->original_file_name);
    fprintf (dptr, "node_id:  %s\n", mhead->node_id);
    fprintf (dptr, "isotope_code:  %s\n", mhead->isotope_code);
    fprintf (dptr, "radiopharmaceutical:  %s\n", mhead->radiopharmaceutical);
    fprintf (dptr, "study_name:  %s\n", mhead->study_name);
    fprintf (dptr, "patient_id:  %s\n", mhead->patient_id);
    fprintf (dptr, "patient_name:  %s\n", mhead->patient_name);
    fprintf (dptr, "patient_age:  %s\n", mhead->patient_age);
    fprintf (dptr, "patient_height:  %s\n", mhead->patient_height);
    fprintf (dptr, "patient_weight:  %s\n", mhead->patient_weight);
    fprintf (dptr, "physician_name:  %s\n", mhead->physician_name);
    fprintf (dptr, "operator_name:  %s\n", mhead->operator_name);
    fprintf (dptr, "study_description:  %s\n", mhead->study_description);
    fprintf (dptr, "facility_name:  %s\n", mhead->facility_name);
    fprintf (dptr, "user_process_code:  %s\n", mhead->user_process_code);

    fprintf (dptr, "patient_sex = %d\n", mhead->patient_sex);
    fprintf (dptr, "patient_dexterity = %d\n", mhead->patient_dexterity);
    fprintf (dptr, "    -- end --\n");

    if (!fptr) fclose (dptr);
}

void fill_string (char *str, int len)
{
    for (int i=0; i<len-2; i++) str[i]='.';
    str[len-2]='\0';
}

ECAT6_Main_header main_zero_fill() 
{
    ECAT6_Main_header v_mhead;

    v_mhead.isotope_halflife= -1.0;
    v_mhead.gantry_tilt= -1.0;
    v_mhead.gantry_rotation= -1.0;
    v_mhead.bed_elevation= -1.0;
    v_mhead.axial_fov= -1.0;
    v_mhead.transaxial_fov= -1.0;
    v_mhead.calibration_factor= -1.0;
    v_mhead.init_bed_position= -1.0;
    v_mhead.plane_separation= -1.0;
    v_mhead.collimator= -1.0;
    for (int i=0; i<16; i++) v_mhead.bed_offset[i]= -1.0;

    v_mhead.num_planes= 0;
    v_mhead.num_frames= 1; // used for matnum, so set coherent default values
    v_mhead.num_gates= 0;
    v_mhead.num_bed_pos= 0;
    v_mhead.sw_version= 64;
    v_mhead.data_type= -1;
    v_mhead.system_type= -1;
    v_mhead.file_type= -1;
    v_mhead.scan_start_day= -1;
    v_mhead.scan_start_month= -1;
    v_mhead.scan_start_year= -1;
    v_mhead.scan_start_hour= -1;
    v_mhead.scan_start_minute= -1;
    v_mhead.scan_start_second= -1;
    v_mhead.rot_source_speed= -1;
    v_mhead.wobble_speed= -1;
    v_mhead.transm_source_type= -1;
    v_mhead.transaxial_samp_mode= -1;
    v_mhead.coin_samp_mode= -1;
    v_mhead.axial_samp_mode= -1;
    v_mhead.calibration_units= -1;
    v_mhead.compression_code= -1;
    v_mhead.acquisition_type= -1;
    v_mhead.bed_type= -1;
    v_mhead.septa_type= -1;
    v_mhead.lwr_sctr_thres= -1;
    v_mhead.lwr_true_thres= -1;
    v_mhead.upr_true_thres= -1;

    fill_string(v_mhead.original_file_name, 20);
    fill_string(v_mhead.node_id, 10);
    fill_string(v_mhead.isotope_code, 8);
    fill_string(v_mhead.radiopharmaceutical, 32);
    fill_string(v_mhead.study_name, 12);
    fill_string(v_mhead.patient_id, 16);
    fill_string(v_mhead.patient_name, 32);
    fill_string(v_mhead.patient_age, 10);
    fill_string(v_mhead.patient_height, 10);
    fill_string(v_mhead.patient_weight, 10);
    fill_string(v_mhead.physician_name, 32);
    fill_string(v_mhead.operator_name, 32);
    fill_string(v_mhead.study_description, 32);
    fill_string(v_mhead.facility_name, 20);
    fill_string(v_mhead.user_process_code, 10);
    v_mhead.patient_sex= 0;
    v_mhead.patient_dexterity= 0;

    return(v_mhead);
}

Scan_subheader scan_zero_fill() 
{ 
    Scan_subheader v_shead;

    v_shead.sample_distance= -1.0;
    v_shead.isotope_halflife= -1.0;
    v_shead.scale_factor= -1.0;
    v_shead.loss_correction_fctr= -1.0;
    v_shead.tot_avg_cor= -1.0;
    v_shead.tot_avg_uncor= -1.0;
    for(int i=0;i<16;i++) v_shead.cor_singles[i]= -1.0;
    for(int i=0;i<16;i++) v_shead.uncor_singles[i]= -1.0;

    v_shead.gate_duration= -1;
    v_shead.r_wave_offset= -1;
    v_shead.prompts= -1;
    v_shead.delayed= -1;
    v_shead.multiples= -1;
    v_shead.net_trues= -1;
    v_shead.total_coin_rate= -1;
    v_shead.frame_start_time= -1;
    v_shead.frame_duration= -1;
    v_shead.data_type= -1;
    v_shead.dimension_1= -1;
    v_shead.dimension_2= -1;
    v_shead.smoothing= -1;
    v_shead.processing_code= -1;
    v_shead.frame_duration_sec= -1;
    v_shead.scan_min= -1;
    v_shead.scan_max= -1;

    return(v_shead);
}

Image_subheader img_zero_fill() 
{
    Image_subheader v_ihead;

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

    v_ihead.frame_duration= -1;
    v_ihead.frame_start_time= -1;
    v_ihead.recon_duration= -1;
    v_ihead.scan_matrix_num= -1;
    v_ihead.norm_matrix_num= -1;
    v_ihead.atten_cor_matrix_num= -1;
    v_ihead.data_type= -1;
    v_ihead.num_dimensions= -1;
    v_ihead.dimension_1= -1;
    v_ihead.dimension_2= -1;
    v_ihead.image_min= -1;
    v_ihead.image_max= -1;
    v_ihead.slice_location= -1;
    v_ihead.recon_start_hour= -1;
    v_ihead.recon_start_minute= -1;
    v_ihead.recon_start_sec= -1;
    v_ihead.filter_code= -1;
    v_ihead.processing_code= -1;
    v_ihead.quant_units= -1;
    v_ihead.recon_start_day= -1;
    v_ihead.recon_start_month= -1;
    v_ihead.recon_start_year= -1;

    fill_string(v_ihead.annotation, 40);

    return(v_ihead);
}


/* adapted by KT from Louvain la Neuve matrix package */
int file_data_to_host(char *dptr, int nblks, int dtype)
{
  int i, j;
  char tmp[512];
  
  switch(dtype)
  {
  case ECAT_Byte_data_type:
    break;
  case ECAT_I2_little_endian_data_type:
    if (ByteOrder::get_native_order() == ByteOrder::big_endian) 
      for (i=0, j=0; i<nblks; i++, j+=512) {
        swab( dptr+j, tmp, 512);
        memcpy(dptr+j, tmp, 512);
      }
      break;
  case ECAT_I4_little_endian_data_type:
    if (ByteOrder::get_native_order() == ByteOrder::big_endian) 
      for (i=0, j=0; i<nblks; i++, j+=512) {
        swab(dptr+j, tmp, 512);
        swaw((short*)tmp, (short*)(dptr+j), 256);
      }
      break;
  case ECAT_R4_VAX_data_type:
    if (ByteOrder::get_native_order() == ByteOrder::big_endian) 
      for (i=0, j=0; i<nblks; i++, j+=512) {
        swab( dptr+j, tmp, 512);
      }
      for (i=0; i<nblks*128; i++)
        ((float *)dptr)[i] = get_vax_float( (unsigned short *)dptr, i*2) ;
      break;
  case ECAT_I2_big_endian_data_type:
    if (ByteOrder::get_native_order() != ByteOrder::big_endian) 
      for (i=0, j=0; i<nblks; i++, j+=512) {
        swab(dptr+j, tmp, 512);
        memcpy(dptr+j, tmp, 512);
      }
      break;
  case ECAT_I4_big_endian_data_type:
  case ECAT_R4_IEEE_big_endian_data_type:
    if (ByteOrder::get_native_order() != ByteOrder::big_endian) 
      for (i=0, j=0; i<nblks; i++, j+=512) {
        swab(dptr+j, tmp, 512);
        swaw((short*)tmp, (short*)(dptr+j), 256);
      }
      break;
  default:	/* something else...treat as Vax I*2 */
    if (ByteOrder::get_native_order() == ByteOrder::big_endian) 
      
      for (i=0, j=0; i<nblks; i++, j+=512) {
        swab(dptr+j, tmp, 512);
        memcpy(dptr+j, tmp, 512);
      }
      break;
  }
  return EXIT_SUCCESS;
}

END_NAMESPACE_ECAT6
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

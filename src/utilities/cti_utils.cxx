/*
 SCCS Revision : $Id$
*/
/*******************************************************************************
    cti_utils.c - PET matrix file access routines.
    
    last revised:
	15 may 94 - akh (add cti_read_image_subheader)

//CL 090398 Adapt the header for implementing CTI librairies on PARSYTEC machine
//KT 251198 moved  _PLATFORM_TRANSPUTER_ dependency after rcn_config.h
            moved EXIT_SUCCES definition to rcn_config.h
	    added SWAP_EM for Macs
	    removed 2 includes
	    store data-type when calling get_scanheaders
            differentiate between data-type when calling get_scandata
	    changed cti_numcod, cti_numdoc and cti_rings2plane with versions 
	       from Larry Byars (except for RPT)
	    word swapping fo get_vax_long and sunltovaxl (note: this name is bad)
*******************************************************************************/

/*{{{  include files */
#include <limits.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* KT 30/11/98 replace bcopy with memcpy on PCs  */
#ifdef _MSC_VER
#define bcopy(src, dest, length) memcpy(dest, src, length)
#endif

#include "CTI/cti_utils.h"          /* interface */

/* KT 25/11/98 removed, as already imported in cti_utils.h
#include "CTI/rcn_config.h"
#include "CTI/rcn_types.h"
*/


/* KT 25/11/98 moved this after rcn_config, to allow this ifdef to work */
#ifdef _PLATFORM_TRANSPUTER_
//#include <mathf.h>
#endif
#include <math.h>

/* KT 25/11/98 moved EXIT_SUCCES to rcn_config.h */

/*#ifdef _PLATFORM_SUN_
typedef struct div_t {int quot, rem;} div_t;
#endif*/
#ifdef _PLATFORM_HP_
 #define _SWAPEM_
#endif
#ifdef _PLATFORM_SUN_
 #define _SWAPEM_
#endif
/* KT 25/11/98 new */
#ifdef _PLATFORM_MAC_
 #define _SWAPEM_
#endif

/*}}}  */

#define toblocks(x) ((x + (MatBLKSIZE - 1))/MatBLKSIZE)

#undef _DEBUG_
/*{{{  get_scanheaders */
/*******************************************************************************
    get_scanheaders - read main header and subheader from scan file
	returns EXIT_SUCCESS if no error.

    fptr - pointer to scan file
    matnum - matnum for scan
    mhead - where to put the main header
    shead - where to put the subheader
    scanParams - where to put the scan parameters
*******************************************************************************/
int get_scanheaders (FILE *fptr, long matnum, Main_header *mhead, 
	Scan_subheader *shead, ScanInfoRec *scanParams)
{
    int     status;
    MatDir      entry;

    /* check the header */
    status = cti_read_main_header (fptr, mhead);
    if (status != EXIT_SUCCESS) {
	#ifdef _DEBUG_
	printf ("\n- get_scanheaders: error reading scan file\n");
	#endif
	return EXIT_FAILURE;
    }
    if (mhead->file_type != matScanFile) {
	printf ("\n- file is not a scan file, type = %d\n",
		mhead->file_type);
	dump_main_header (0, mhead);
	return EXIT_FAILURE;
    }

    /* look up matnum in scan file */
    if (!cti_lookup (fptr, matnum, &entry)) {
	printf ("\n- specified matrix not in scan file\n");
	dump_main_header (0, mhead);
	return EXIT_FAILURE;
    }

    /* read scan subheader */
    status = cti_read_scan_subheader (fptr, entry.strtblk, shead);
    if (status != EXIT_SUCCESS) {
	printf ("\n- error reading scan subheader\n");
	return EXIT_FAILURE;
    }

    scanParams->strtblk = entry.strtblk + 1;
    scanParams->nblks = entry.endblk - entry.strtblk;
    scanParams->nprojs = shead->dimension_1;
    scanParams->nviews = shead->dimension_2;
    /* KT 25/11/98 added data_type */
    scanParams->data_type = shead->data_type;
    if (shead->data_type != mhead->data_type)
    {
      printf("\nget_scanheader warning: \n\
data types differ between main header (%d) and subheader (%d)\n\
Using value from subheader\n", mhead->data_type, shead->data_type);
    }

    return EXIT_SUCCESS;
}
/*}}}  */

/*{{{  get_scandata*/
/*******************************************************************************
    get_scandata - read scan data from file; returns EXIT_FAILURE if the
	data could not be read.

    fptr - scan file
    scan - buffer for the data;  caller must provide
    scanParams - data parameters
*******************************************************************************/
int get_scandata (FILE *fptr, short *scan, ScanInfoRec *scanParams)
{
    int status;

    /* read data from scan file */
    if (!scan) {
	#ifdef _DEBUG_
	printf ("get_scandata: null scan buffer pointer.\n");
	#endif
	return EXIT_FAILURE;
    }

    #ifdef _DEBUG_
    printf ("get_scandata: reading block %d (%d blocks)\n",
	    scanParams->strtblk, scanParams->nblks);
    #endif
    status = cti_rblk (fptr, scanParams->strtblk, (char *) scan,
		       scanParams->nblks);

    if (status != EXIT_SUCCESS) {
	#ifdef _DEBUG_
	printf ("get_scandata: read block %d failed (%d blocks)\n",
		scanParams->strtblk, scanParams->nblks);
	#endif
    }
    else {
        /* KT 25/11/98 differentiate between data-type */

	#ifdef _SWAPEM_
	/* we have to swap bytes in order to read the ints */
        if (scanParams->data_type == matI2Data)
  	  swab ((char *) scan, (char *) scan, scanParams->nblks * MatBLKSIZE);
        else
          if (scanParams->data_type != matSunShort)
	    {
	      printf("\nget_scandata: unsupported data_type %d\n", 
		     scanParams->data_type);
	      return(EXIT_FAILURE);
	    }
        #else
	/* we have to swap bytes in order to read the ints */
        if (scanParams->data_type == matSunShort)
  	  swab ((char *) scan, (char *) scan, scanParams->nblks * MatBLKSIZE);
        else
          if (scanParams->data_type != matI2Data)
	    {
	      printf("\nget_scandata: unsupported data_type %d\n", 
		     scanParams->data_type);
	      return(EXIT_FAILURE);
	    }

	#endif
    }
    return status;
}
/*}}}  */

/*{{{  make_imageheaders*/
/*******************************************************************************
    make_imageheaders - fill in image main header and subheader

    mhead - on entry, should contain the scan file main header; will be
	modified into an image file main header.
    ihead - where to put the image subheader
    image_max - max. value of the image data
    imatnum - matnum for the image
    isize - size of image
    zoom - zoom factor
*******************************************************************************/
void make_imageheaders (Main_header *mhead, Image_subheader *ihead,
	float image_max, long imatnum, int isize, float zoom)
{
    imatnum = imatnum;   /* not used  here */

    /* main header is just a copy of the scan file main header,
     * except for the file type  */
    mhead->file_type = matImageFile;

    /* build the subheader */
    fill_image_subheader (ihead, zoom, isize, image_max);
}
/*}}}  */

/*{{{  cti_numcod */
/*******************************************************************************
	cti_numcod - encode scan information into a single, incomprehensible number.
	
	frame - 
	plane - 
	gate -
	data -
	bed -
*******************************************************************************/
long cti_numcod (CameraType scanner, 
				int frame, int plane, int gate, int data, int bed)
{

  /* KT 01/12/98 use the code from Larry Byars (and the LLN distribution) except for RPT.
     Only difference seems to be that RPT sets gate|=1*/
	switch (scanner) {
#if 0	
		case cam951:      /* ?? */
			/*  ECAT 953 RTS1       */
			return ((frame & 0x1FF) | ((bed & 0xF) << 12) 
					| ((plane & 0xFF) << 16)
					| ((plane >> 8) << 9)
					| ((gate & 0x3F) << 24)
					| ((data & 0x3) << 30));
			break;
#endif		
		case camRPT:    
		/* case cam953: */	
			/*  ECAT 953 RTS2   */
			return ((frame & 0x1FF) | ((bed & 0xF) << 12) 
					| ((plane & 0xFF) << 16) 
					| (((plane >> 8) & 0x7) << 9) 
					| ((gate & 0x3F) << 24) 
					| ((data & 0x3) << 30) 
					| (1 << 24));
			break;  
		
		default:
	  	return ((frame)|((bed&0xF)<<12)|((plane&0xFF)<<16)|(((plane&0x300)>>8)<<9)|
	       ((gate&0x3F)<<24)|((data&0x3)<<30)|((data&0x4)<<9));

			break;
	}
}
/*}}}  */

/*{{{  cti_numdoc */
/*******************************************************************************
	cti_numdoc - unpack encoded data into a nice struct.  
		reverse of cti_numcod ().
	
	matnum - the thingy to decode
	matval - struct containing the decoded values from matnum
******************************************************************************/
void cti_numdoc (CameraType scanner, long matnum, Matval *matval)
{
	switch (scanner) {
	/* KT 01/12/98 use Larry Byras' (and LLN) code, except for RPT */
		case camRPT:    
		/* case cam953: */
		/*  Same for both ECAT 953 RTS1 and ECAT 953 RTS2   */
		  matval->frame = matnum & 0x1FF;
		  matval->plane = ((matnum >> 16) & 0xFF) + (((matnum >> 9) & 0x7) << 8);
		  matval->gate  = (matnum >> 24) & 0x3F;
		  matval->data  = (matnum >> 30) & 0x3;
		  matval->bed   = (matnum >> 12) & 0xF;
		break;
		
		default:			  
		  matval->frame = matnum&0x1FF;
		  matval->plane = ((matnum>>16)&0xFF) + (((matnum>>9)&0x3)<<8);
		  matval->gate  = (matnum>>24)&0x3F;
		  matval->data  = ((matnum>>9)&0x4)|(matnum>>30)&0x3;
		  matval->bed   = (matnum>>12)&0xF;

		break;
	}
}
/*}}}  */

/*{{{  cti_rings2plane */
/*******************************************************************************
	cti_rings2plane - get sinogram plane from ring pair.
	
	ring0 - first ring in ring pair
	ring1 - second ring in ring pair
******************************************************************************/

//CL Change the 1rst argument to the number of rings
//int cti_rings2plane (CameraType scanner, short ring0, short ring1) {
int cti_rings2plane (short nrings, short ring0, short ring1) {
	/* KT TODO this might have to be changed for other scanners as well */
// CL 12/02/99 Now done, this function is generalized for any PET scanner
   //for a PET scanner having "rings" rings
  
    
        
#if 0
	switch (scanner) {
	        /* KT 01/12/98 added 921 */
            case cam921:
                fprintf(stderr, "cti_rings2plane: 921 not supported\n");
                return 0;
            case camRPT:
                nrings=16;
                break;
                
            case cam953:
                nrings=16;
                break;
                
                    /* KT 01/12/98 added 951 */
            case cam951:
                nrings= 16;
                break;
                
            case camGE://CL 150299 Add a new camera type in camera.h
                    nrings= 18;
                    break;
		
		default:
                        /* Note, this case in particular includes 921.
s                           Others scanners might not work, but we didn't check */
/* KT 01/12/98 added  */
		  /* New ACS numbering (2048 planes 0-2047) */
                        //   return ((ring0&0x10)<<5)+((ring0&0x08)<<4)+(ring0&7)+
                        //  ((ring1&0x10)<<4)+((ring1&15)<<3)+1;
                    fprintf(stderr,"cti_rings2plane: Scanner not supported\n");
                    return 0;
                    
                    
	}
     #endif   
//CL 15/02/99 A more generalized formula for any PET scanners
        int d = (int) (ring0 / (nrings/2)); 
        return (ring1 * nrings/2 + ring0 % (nrings/2) +
                nrings/2 * nrings * d + 1);			 
}
/*}}}  */

/*{{{  cti_open */
/*******************************************************************************
	cti_open - open a matrix file and return its pointer.
	
	fname - file to open
	fmode - how it should be opened
*******************************************************************************/
FILE *cti_open (const char* fname, const char *fmode)
{
	FILE *fptr;
	
	fptr = fopen (fname, fmode);
	return (fptr);
}
/*}}}  */

/*{{{  cti_close */
/*******************************************************************************
	cti_close - close a matrix file given its pointer.
	
	fptr - file to close
*******************************************************************************/
void cti_close (FILE *fptr)
{
	fflush (fptr);
	fclose (fptr);
}
/*}}}  */

/*{{{  cti_rblk */
/*******************************************************************************
	cti_rblk - read from a matrix file starting at the given block.
		returns EXIT_SUCCESS if all went well.
	
	fptr - file pointer
	blkno - first block to read
	nblks - number of blocks to read
*******************************************************************************/
int cti_rblk (FILE *fptr, int blkno, void *bufr, int nblks)
{
    int n, err;

    if (!fptr || !bufr) {
	#ifdef _DEBUG_
	printf ("cti_rblk: file pointer or buffer pointer was null\n");
	#endif
	return EXIT_FAILURE;
    }
    #ifdef _DEBUG_
    printf ("cti_rblk: seeking to %ld\n", (long) (blkno - 1) * MatBLKSIZE);
    #endif
    err = fseek (fptr, (long) (blkno - 1) * MatBLKSIZE, 0);
    if (err) {
	#ifdef _DEBUG_
	printf ("cti_rblk: fseek() returned %d instead of 0\n", err);
	#endif
	return (EXIT_FAILURE);
    }
    #ifdef _DEBUG_
    printf ("cti_rblk: reading %d blocks\n", nblks);
    #endif
    n = fread (bufr, sizeof (char), nblks * MatBLKSIZE, fptr);
    if (n != nblks * MatBLKSIZE) {
	#ifdef _DEBUG_
	printf ("cti_rblk: fread() returned %d instead of %d\n",
		n, nblks * MatBLKSIZE);
	#endif
	return (EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}
/*}}}  */

/*{{{  cti_wblk */
/*******************************************************************************
	cti_wblk - write blocks from buffer into file.  returns EXIT_SUCCESS
	if successful.

	fptr - pointer to file.
	blkno - position in file of first block to write.
	nblks - number of blocks to write.
*******************************************************************************/
int cti_wblk (FILE *fptr, int blkno, void *bufr, int nblks)
{
    int err;

    if (!fptr || !bufr) {
	#ifdef _DEBUG_
	printf ("cti_wblk: null argument.\n");
	#endif
	return EXIT_FAILURE;
    }

    /* seek to position in file */
    #ifdef _DEBUG_
    printf ("cti_wblk: seeking to %ld\n", (long) (blkno - 1) * MatBLKSIZE);
    #endif
    err = fseek (fptr, (long) (blkno - 1) * MatBLKSIZE, 0);
    if (err) {
	#ifdef _DEBUG_
	printf ("cti_wblk: fseek() error code = %d \n", err);
	#endif
	return (EXIT_FAILURE);
    }

    #ifdef _DEBUG_
    printf ("cti_wblk: writing %d blocks\n", nblks);
    #endif
    err = fwrite (bufr, sizeof (char), nblks * MatBLKSIZE, fptr);
    if (err != nblks * MatBLKSIZE){
	#ifdef _DEBUG_
	printf ("cti_wblk: fwrite() returns %d but n = %d\n",
		err, nblks * MatBLKSIZE);
	#endif
	return EXIT_FAILURE;
    }
		
    return EXIT_SUCCESS;
}
/*}}}  */

/*{{{  cti_read_main_header */
/*******************************************************************************
	cti_read_main_header - read header data from a file and place it into a 
		Main_header struct.  returns EXIT_SUCCESS if no error.
	
	fptr - file containing the header data
	h - struct to fill with header info
*******************************************************************************/
int cti_read_main_header (FILE *fptr, Main_header *h)
{
    short *b;       /* buffer for read */
    char *bb;
    int status, i;

    b = (short *) malloc (MatBLKSIZE);
    if (!b) {
	#ifdef _DEBUG_
	printf ("cti_read_main_header: calloc failed\n");
	#endif
	return EXIT_FAILURE;
    }
    bb = (char *) b;

    /* read main header at block 1 into buf */
    status = cti_rblk (fptr, 1,  bb, 1);

    if (status != EXIT_SUCCESS) {
	#ifdef _DEBUG_
	printf ("cti_read_main_header: cti_rblk failed\n");
	#endif
	return EXIT_FAILURE;
    }

    /* copy all the strings */
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

#ifdef _SWAPEM_
    /* we have to swap bytes in order to read ints and floats */
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
    for (i = 0; i < 15; i++)
	h->bed_offset [i] = get_vax_float ((unsigned short *) b, 194 + 2 * i);
    h->plane_separation = get_vax_float ((unsigned short *) b, 224);
    h->lwr_sctr_thres = b [226];
    h->lwr_true_thres = b [227];
    h->upr_true_thres = b [228];
    h->collimator = get_vax_float ((unsigned short *) b, 229);

    free (b);
    return EXIT_SUCCESS;
}
/*}}}  */

/*{{{  cti_read_scan_subheader */
/*******************************************************************************
	cti_read_scan_subheader - read header data from a file and place it into a 
		Scan_subheader struct.  returns EXIT_SUCCESS if no error.
	
	fptr - file containing the header data
	blknum - block number at which to begin reading
	h - struct to fill
*******************************************************************************/
int cti_read_scan_subheader (FILE *fptr, int blknum, Scan_subheader *h)
{
    short   *b;         /* our buffer */
    int     i, status ;

    b = (short *) malloc (MatBLKSIZE);
    if (!b) {
	#ifdef _DEBUG_
	printf ("cti_read_scan_subheader: malloc failed\n");
	#endif
	return EXIT_FAILURE;
    }

    status = cti_rblk (fptr, blknum, (char *) b, 1);   /* read the block */
    if (status != EXIT_SUCCESS) {
	#ifdef _DEBUG_
	printf ("cti_read_scan_subheader: read block #%d failed\n", blknum);
	#endif
	free (b);
	return (EXIT_FAILURE);
    }

#ifdef _SWAPEM_
    /* we have to swap bytes in order to read the ints */
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
    for (i = 0; i < 16; i++) {
	h->cor_singles [i] = get_vax_float ((unsigned short *) b, 158 + 2 * i);
	h->uncor_singles [i] = get_vax_float ((unsigned short *) b, 190 + 2 * i);
    }
    h->tot_avg_cor = get_vax_float ((unsigned short *) b, 222);
    h->tot_avg_uncor = get_vax_float ((unsigned short *) b, 224);
    h->total_coin_rate = get_vax_long ((unsigned short *) b, 226);
    h->frame_start_time = get_vax_long ((unsigned short *) b, 228);
    h->frame_duration = get_vax_long ((unsigned short *) b, 230);
    h->loss_correction_fctr = get_vax_float ((unsigned short *) b, 232);
    free (b);
    return EXIT_SUCCESS;
}
/*}}}  */

/*{{{  cti_read_image_subheader*/
/*******************************************************************************
	cti_read_image_subheader - fill in various parts of the
		image subheader from the cti file pointed to

	header - header to fill in
*******************************************************************************/
int cti_read_image_subheader (FILE *fptr,
			int blknum,
			Image_subheader *ihead)
{
    int i;
    int status;
    short *b;
    char  *bb;

    /*{{{  alloc buffer*/
    b = (short *) malloc (MatBLKSIZE);// CL 090398 Add unsigned
    if (!b) {
	#ifdef _DEBUG_
	printf ("cti_read_image_subheader: malloc failed\n");
	#endif
	return EXIT_FAILURE;
    }
    /*}}}  */

    /*{{{  read block into buffer*/
    status = cti_rblk (fptr, blknum, (char *) b, 1);   /* read the block */
    if (status != EXIT_SUCCESS) {
	#ifdef _DEBUG_
	printf ("cti_read_image_subheader: read block #%d failed\n", blknum);
	#endif
	free (b);
	return (EXIT_FAILURE);
    }
    /*}}}  */

    bb = (char *) b;
    strncpy (ihead->annotation, bb + 420, 40);

#ifdef _SWAPEM_
    /* we have to swap bytes in order to read the ints */
    swab ((char *) b, (char *) b, MatBLKSIZE);
#endif
	
    /*{{{  fill in the image subheader*/
    ihead->x_origin = get_vax_float ((unsigned short *)b, 80); // CL 090398 Add (unsigned short *)
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
    
    ihead->image_rotation = get_vax_float ((unsigned short *)b, 148); // CL 090398 Add (unsigned short *)
    ihead->plane_eff_corr_fctr = get_vax_float ((unsigned short *)b, 150);
    ihead->decay_corr_fctr = get_vax_float ((unsigned short *)b, 152);
    ihead->loss_corr_fctr = get_vax_float ((unsigned short *)b, 154);
    ihead->ecat_calibration_fctr = get_vax_float ((unsigned short *)b, 194);
    ihead->well_counter_cal_fctr = get_vax_float ((unsigned short *)b, 196);
    for (i = 0; i < 6; i++)
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
    /*}}}  */

    free (b);
    return (EXIT_SUCCESS);
}
/*}}}  */

/*{{{  cti_create */
/*******************************************************************************
	cti_create - open a file and write main header to it.  returns a pointer
			to the file, or 0 if unsuccessful.

	fname - name of file to open
	mhead - pointer to main header struct to copy into the file
*******************************************************************************/
FILE *cti_create (const char *fname, const Main_header *mhead)
{
    FILE *fptr;
    int status;
    long *bufr;

    /* open the file and write the header into it. */
    fptr = cti_open (fname, "w+");
    if (!fptr) {
	#ifdef _DEBUG_
	printf ("cti_create: open failed\n");
	#endif
	return fptr;
    }
    status = cti_write_main_header (fptr, mhead);
    if (status != EXIT_SUCCESS) {
	#ifdef _DEBUG_
	printf ("cti_create: cti_write_main_header returns %d\n", status);
	#endif
	fclose (fptr);
	return NULL;
    }
	
    /* create a First Directory Block in the file */
    bufr = (long *) calloc (MatBLKSIZE / sizeof (long), sizeof (long));
    if (!bufr) {
	#ifdef _DEBUG_
	printf ("cti_create:  calloc failed\n");
	#endif
	fclose (fptr);
	return NULL;
    }

    bufr [0] = 31;          /* mystery number */
    bufr [1] = 2;           /* next block */
	
#ifdef _SWAPEM_
    /* we must do some swapping about */
    swaw ((short *) bufr, (short *) bufr, MatBLKSIZE/2);
    swab ((char *) bufr, (char *) bufr, MatBLKSIZE);
#endif

    #ifdef _DEBUG_
    printf ("cti_create: writing First Directory Block at %d.\n", MatFirstDirBlk);
    printf ("            bufr [0] = %ld;  bufr [1] = %ld\n", bufr[0], bufr[1]);
    #endif
    status = cti_wblk (fptr, MatFirstDirBlk, (char *) bufr, 1);
    if (status != EXIT_SUCCESS) {
	#ifdef _DEBUG_
	printf ("cti_create: write first dir block: cti_wblk returns %d\n", status);
	#endif
	fclose (fptr);
	free (bufr);
	return NULL;
    }
    free (bufr);
	
    return (fptr);
}
/*}}}  */

/*{{{  cti_enter */
/*******************************************************************************
	cti_enter - create entry in file corresponding to matnum, and return 
		offset of next block.  or some such.  Returns 0 if there was an
		error.

	fptr - pointer to file.
	matnum - desired matnum.
	nblks - number of blocks
*******************************************************************************/
int cti_enter (FILE *fptr, long matnum, int nblks)
{
    int     i, dirblk, nxtblk, busy, oldsize;
    long    *dirbufr;           /* buffer for directory block */
    int     status;

    /*{{{  set up buffer for directory block*/
    dirbufr = (long *) calloc (MatBLKSIZE / sizeof (long), sizeof (long));
    if (!dirbufr) {
	#ifdef _DEBUG_
	printf ("cti_enter: calloc failed.\n");
	#endif
	return 0;
    }
    /*}}}  */

    /*{{{  read first directory block from file*/
    dirblk = MatFirstDirBlk;
    #ifdef _DEBUG_
    printf( "cti_enter: read first directory block at %d\n", dirblk);
    #endif
    status = cti_rblk (fptr, dirblk, (char *) dirbufr, 1);
    if (status != EXIT_SUCCESS) {
	#ifdef _DEBUG_
	printf ("cti_enter: failed to read first directory block.\n");
	#endif
	free (dirbufr);
	return 0;
    }
    /*}}}  */

    #ifdef _SWAPEM_
    /*{{{  swap bytes*/
    /* we must swap about */
    swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
    swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE / 2);
    /*}}}  */
    #endif

    status = EXIT_SUCCESS;
    busy = 1;

    /*{{{  search for our matnum in directory blocks*/
    while (busy && status == EXIT_SUCCESS) {
    
	nxtblk = dirblk + 1;
    
	/*{{{  see if matnum entry is in this block*/
	
	for (i = 4; i < MatBLKSIZE / sizeof (long); i += sizeof (long)) {
	    if (dirbufr [i] == 0) {         /* no mo' go */
		/*{{{  skip to next block*/
		#ifdef _DEBUG_
		printf ("cti_enter: found null at block %d [%d]\n", dirblk, i);
		#endif
		busy = 0;
		break;
		/*}}}  */
	    }
	    else if (dirbufr [i] == matnum) {     /* found it */
		/*{{{  see if this entry has reserved enough space for us*/
		
		#ifdef _DEBUG_
		printf ("cti_enter: found matnum %#lX at offset %d\n", matnum, i);
		#endif
		
		/* see if there's enough room */
		oldsize = dirbufr [i + 2] - dirbufr [i + 1] + 1;
		if (oldsize < nblks) {
		    /*{{{  delete old entry and create new one*/
		    
		    dirbufr [i] = 0xFFFFFFFF;
		    
		    #ifdef _SWAPEM_
		    swaw ((short *) dirbufr,
			  (short *) dirbufr,
			  MatBLKSIZE / sizeof (short));
		    swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
		    #endif
		    #ifdef _DEBUG_
		    printf ("cti_enter: deleting entry at block %d [%d]\n", dirblk, i);
		    #endif
		    status = cti_wblk (fptr, dirblk, (char *) dirbufr, 1);
		    
		    #ifdef _SWAPEM_
		    swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
		    swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE / sizeof (short));
		    #endif
		    nxtblk = dirbufr [i + 2] + 1;
		    /*}}}  */
		}
		else {    /* enough room here */
		    /*{{{  get pointer to next block from entry*/
		    nxtblk = dirbufr [i + 1];
		    dirbufr [0] ++;
		    dirbufr [3] --;
		    busy = 0;
		    break;
		    /*}}}  */
		}
		/*}}}  */
	    }
	    else {
		nxtblk = dirbufr [i + 2] + 1;
		#ifdef _DEBUG_
		printf ("cti_enter: reached end of block, next =%d\n", nxtblk);
		#endif
	    }
	} /* end of for */
	/*}}}  */
    
	if (!busy) break;  /* hit end of block, or found it */
    
	if (dirbufr [1] != MatFirstDirBlk) {
	    /*{{{  hop to next block*/
	    /* hop to next block */
	    dirblk = dirbufr [1];
	    #ifdef _DEBUG_
	    printf( "cti_enter: read next directory block at %d\n", dirblk);
	    #endif
	    status = cti_rblk (fptr, dirblk, (char *) dirbufr, 1);
	    if (status != EXIT_SUCCESS) {
		#ifdef _DEBUG_
		printf ("cti_enter: failed to read next directory block.\n");
		#endif
		status = EXIT_FAILURE;     /* get out */
		break;
	    /*}}}  */
	    }
    #ifdef _SWAPEM_
	    swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
	    swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE / 2);
    #endif
	} else {
	    /*{{{  add a new block*/
	    
	    /* modify this block to point to next block */
	    dirbufr [1] = nxtblk;
	    
	    /* do some swapping for good measure */
	    #ifdef _SWAPEM_
	    swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE/2);
	    swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
	    #endif
	    #ifdef _DEBUG_
	    printf ("cti_enter: writing new directory block at %d\n", dirblk);
	    #endif
	    status = cti_wblk (fptr, dirblk, (char *) dirbufr, 1);
	    
	    /* prepare new directory block */
	    dirbufr [0] = 31;
	    dirbufr [1] = MatFirstDirBlk;
	    dirbufr [2] = dirblk;
	    dirbufr [3] = 0;
	    dirblk = nxtblk;
	    for (i = 4; i < MatBLKSIZE / 4; i++)
		dirbufr [i] = 0;
	    /*}}}  */
    
	}
    }       /* end of busy loop */
    /*}}}  */

    if (status == EXIT_SUCCESS) {
	 /*{{{  add new entry*/
	 #ifdef _DEBUG_
	 printf ("cti_enter: creating new entry, next = %d, size = %d\n",
		 nxtblk, nblks);
	 #endif
	 dirbufr [i] = matnum;
	 dirbufr [i+1] = nxtblk;
	 dirbufr [i+2] = nxtblk + nblks;
	 dirbufr [i+3] = 1;
	 dirbufr [0] --;
	 dirbufr [3] ++;
	 
	 #ifdef _SWAPEM_
	 swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE/2);
	 swab ((char *) dirbufr, (char *) dirbufr, MatBLKSIZE);
	 #endif
	 
	 /*}}}  */
	 /*{{{  write to directory block*/
	 #ifdef _DEBUG_
	 printf ("cti_enter: writing entry for matnum %#lX at block %d [%d]\n",
		 matnum, dirblk, i);
	 #endif
	 cti_wblk (fptr, dirblk, (char *) dirbufr, 1);
	 /*}}}  */
    }

    fflush (fptr);
    free (dirbufr);

    if (status != EXIT_SUCCESS) {
	#ifdef _DEBUG_
	printf ("cti_enter: failed.\n");
	#endif
	return 0;
    }
    return (nxtblk);
}
/*}}}  */

/*{{{  cti_lookup*/
/*******************************************************************************
	cti_lookup - look up a "matrix number" in the file and return the MatDir
		entry for it.  returns 0 if the lookup was NOT successful.
	
	fptr - file containing all the tabulated information.
	matnum - index.
	entry - where to put the result.
*******************************************************************************/
int cti_lookup (FILE *fptr, long matnum, MatDir *entry)
{
    int blk, i, status;
    int nfree, nxtblk, prvblk, nused, matnbr, strtblk, endblk, matstat;
    long *dirbufr;

    #ifdef _SWAPEM_
    /*{{{  set up byte buffer*/
    char *bytebufr;
    bytebufr = (char *) malloc (MatBLKSIZE);
    if (!bytebufr) {
	#ifdef _DEBUG_
	printf ("cti_lookup: malloc failed\n");
	#endif
	return 0;
    }
    /*}}}  */
    #endif

    /*{{{  set up buffer for directory block*/
    
    dirbufr = (long *) malloc (MatBLKSIZE);
    if (!dirbufr) {
	#ifdef _DEBUG_
	printf ("cti_lookup: malloc failed\n");
	#endif
	return 0;
    }
    /*}}}  */

    blk = MatFirstDirBlk;
    status = EXIT_SUCCESS;
    while (status == EXIT_SUCCESS) { /* look through the blocks in the file */
	#if 0
	printf ("cti_lookup: reading directory block %d\n", blk);
	#endif
	/*{{{  read a block and examine the matrix numbers in it*/
	#ifdef _SWAPEM_   /* read into byte buffer */
	/*{{{  read into byte buffer and swap*/
	status = cti_rblk (fptr, blk, bytebufr, 1);
	if (status != EXIT_SUCCESS)
	    break;
	
	/* we have to swap bytes in order to read the ints */
	swab (bytebufr, (char *) dirbufr, MatBLKSIZE);
	/*}}}  */
	#else   /* read into directory buffer */
	/*{{{  just read*/
	status = cti_rblk (fptr, blk, (char *) dirbufr, 1);
	if (status != EXIT_SUCCESS)
	    break;
	/*}}}  */
	#endif
	
	#ifdef _SWAPEM_
	/*{{{  swap words*/
	swaw ((short *) dirbufr, (short *) dirbufr, MatBLKSIZE / sizeof (short));
	/*}}}  */
	#endif
	
	/*{{{  get directory block info*/
	nfree  = dirbufr [0];
	nxtblk = dirbufr [1];
	prvblk = dirbufr [2];
	nused  = dirbufr [3];
	
	#ifdef _DEBUG_
	printf ("        %#X free, next = %#X, prev = %#X, used = %#X\n",
		nfree, nxtblk, prvblk, nused);
	#endif
	/*}}}  */
	
	/*{{{  look through entries in the block*/
	/* look through the entries in this block */
	for (i = 4; i < MatBLKSIZE / sizeof (long); i += sizeof (long)) {
	    matnbr  = dirbufr [i];
	    strtblk = dirbufr [i + 1];
	    endblk  = dirbufr [i + 2];
	    matstat = dirbufr [i + 3];
	    if (matnum == matnbr) { /* got our entry */
		#ifdef _DEBUG_
		printf ("cti_lookup: found entry at [%d] for matnum %#lX\n",
			i, matnum);
		printf ("            matnbr = %#X, start = %#X, end = %#X, stat = %#X\n",
			matnbr, strtblk, endblk, matstat);
		#endif
		entry->matnum  = matnbr;
		entry->strtblk = strtblk;
		entry->endblk  = endblk;
		entry->matstat = matstat;
		free (dirbufr);
		#ifdef _SWAPEM_
		free (bytebufr);
		#endif
		return 1;     /* we were successful */
	    }
	    #ifdef _DEBUG_
	    else if (matnbr) {
		printf ("cti_lookup: ----  entry [%d] does not match:\n", i);
		printf ("            matnbr = %#X, start = %#X, end = %#X, stat = %#X\n",
			matnbr, strtblk, endblk, matstat);
	    }
	    #endif
	}
	/*}}}  */
	
	blk = nxtblk;       /* point to next block */
	if (blk <= MatFirstDirBlk) break;
	/*}}}  */
    }
    #ifdef _SWAPEM_
    free (bytebufr);
    #endif
    free (dirbufr);

    #ifdef _DEBUG_
    printf ("cti_lookup: failed to find matnum\n");
    #endif
    return 0;       /* we were unsuccessful */
}
/*}}}  */


/*{{{  cti_write_idata */
/*******************************************************************************
    cti_write_idata - write data in blocks from buffer into file.

    fptr - pointer to file.
    blk - offset (in blocks) in file of first block to write.
    data - buffer to write
    ibytes - number of bytes to write.  (should be multiple of MatBLKSIZE)
*******************************************************************************/
int cti_write_idata (FILE *fptr, int blk, const short *data, int ibytes)
{
    unsigned int nblks;
    char *dataptr;
    int status;

#ifdef _SWAPEM_
    char *bufr;
    int i;

    /* allocate intermediate buffer */
    bufr = (char *)calloc (MatBLKSIZE, sizeof (char));
    if (!bufr)
	return (EXIT_FAILURE);

    dataptr = (char *) data;    /* point into data buffer */

    /* we'll use cti_wblk to write the data via another buffer.
     * this way, if we need to transform the data as we went,
     * we can do it. */
    nblks = toblocks (ibytes);
    for (i = 0; i < nblks; i++) {
	bcopy (dataptr, bufr, MatBLKSIZE);
	/* swap the bytes */
	swab (bufr, bufr, MatBLKSIZE);
	if ((status = cti_wblk (fptr, blk + i, bufr, 1)) != EXIT_SUCCESS) {
	    free (bufr);
	    return (EXIT_FAILURE);
	}
	dataptr += MatBLKSIZE;
    }
    fflush (fptr);
    free (bufr);
#else
    /* write the data in blocks */
    nblks = toblocks (ibytes);
    dataptr = (char *) data;
    #ifdef _DEBUG_
    printf ("cti_write_idata: writing %d blocks at block %d\n", nblks, blk);
    #endif
    if ((status = cti_wblk (fptr, blk, dataptr, nblks)) != EXIT_SUCCESS)
	return (EXIT_FAILURE);
    fflush (fptr);
#endif
    return (EXIT_SUCCESS);
}
/*}}}  */

/*{{{  cti_write_image_subheader*/
/*******************************************************************************
    cti_write_image_subheader - write an image subheader into a matrix file.
	returns 0 if successful.

    fptr - pointer to file.
    blknum - offset (in blocks) in file of first block to write.
    header - header to write
*******************************************************************************/
int cti_write_image_subheader (FILE *fptr, int blknum, 
    const Image_subheader *header)
{
    int i, status;
    char *bbufr;
    short *bufr = 0;
  
    bufr = (short *) calloc (MatBLKSIZE / sizeof (short), sizeof (short));
    if (!bufr)
	return EXIT_FAILURE;

    bbufr = (char *) bufr;
#ifdef _DEBUG_
printf ("cti_write_image_subheader: data_type = %d, num_dimensions = %d\n",
    header->data_type, header->num_dimensions);
printf ("    image_min = %d, image_max = %d, pixel_size = %g, quant_scal = %g\n",
    header->image_min, header->image_max,
    header->pixel_size, header->quant_scale);
#endif
    /* transfer subheader information */
    bufr [63] = header->data_type;
    bufr [64] = header->num_dimensions;
    bufr [66] = header->dimension_1;
    bufr [67] = header->dimension_2;
    sunftovaxf (header->x_origin, (unsigned short *) &bufr [80]);
    sunftovaxf (header->y_origin, (unsigned short *) &bufr [82]);
    sunftovaxf (header->recon_scale, (unsigned short *) &bufr [84]);
    sunftovaxf (header->quant_scale, (unsigned short *) &bufr [86]);
    bufr [88] = header->image_min;
    bufr [89] = header->image_max;
    sunftovaxf (header->pixel_size, (unsigned short *) &bufr [92]);
    sunftovaxf (header->slice_width, (unsigned short *) &bufr [94]);
    sunltovaxl (header->frame_duration, (unsigned short *) &bufr [96]);
    sunltovaxl (header->frame_start_time, (unsigned short *) &bufr [98]);
    bufr [100] = header->slice_location;
    bufr [101] = header->recon_start_hour;
    bufr [102] = header->recon_start_minute;
    bufr [103] = header->recon_start_sec;
    sunltovaxl (header->recon_duration, (unsigned short *) &bufr [104]);
    bufr [118] = header->filter_code;
    sunltovaxl (header->scan_matrix_num, (unsigned short *) &bufr [119]);
    sunltovaxl (header->norm_matrix_num, (unsigned short *) &bufr [121]);
    sunltovaxl (header->atten_cor_matrix_num, (unsigned short *) &bufr [123]);
    sunftovaxf (header->image_rotation, (unsigned short *) &bufr [148]);
    sunftovaxf (header->plane_eff_corr_fctr, (unsigned short *) &bufr [150]);
    sunftovaxf (header->decay_corr_fctr, (unsigned short *) &bufr [152]);
    sunftovaxf (header->loss_corr_fctr, (unsigned short *) &bufr [154]);
    bufr [188] = header->processing_code;
    bufr [190] = header->quant_units;
    bufr [191] = header->recon_start_day;
    bufr [192] = header->recon_start_month;
    bufr [193] = header->recon_start_year;
    sunftovaxf (header->ecat_calibration_fctr, (unsigned short *) &bufr [194]);
    sunftovaxf (header->well_counter_cal_fctr, (unsigned short *) &bufr [196]);

    for (i = 0; i < 6; i++)
      sunftovaxf (header->filter_params [i], (unsigned short *) &bufr [198+2*i]);

#ifdef _SWAPEM_
    /* swap the bytes */
    swab ((char *) bufr, (char *) bufr, MatBLKSIZE);
#endif

#ifdef _DEBUG_
printf ("cti_write_image_subheader: bufr[data_type] = %d, bufr[num_dimensions] = %d\n",
	bufr[63], bufr[64]);
#endif
    bcopy (header->annotation, bbufr + 420, 40);
    /* write to matrix file */
    status = cti_wblk (fptr, blknum, bbufr, 1);
    free (bufr);
	
    if (status != EXIT_SUCCESS)
	return (EXIT_FAILURE);
    return (EXIT_SUCCESS);
}
/*}}}  */

/*{{{  cti_write_main_header */
/*******************************************************************************
	cti_write_main_header - write an image main header into a matrix file.  
		returns 0 if successful.

	fptr - pointer to file.
	header - header to write
*******************************************************************************/
int cti_write_main_header (FILE *fptr, const Main_header *header)
{
    char *bbufr;
    short *bufr;
    int status, i;

    bufr = (short *) calloc (MatBLKSIZE / sizeof (short), sizeof (short));
    if (!bufr) {
	#ifdef _DEBUG_
	printf ("cti_write_main_header: calloc failed\n");
	#endif
	return EXIT_FAILURE;
    }

#ifdef _DEBUG_
printf ("cti_write_main_subheader: data_type = %d, file_type = %d\n",
    header->data_type, header->file_type);
#endif
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
    sunftovaxf (header->isotope_halflife, (unsigned short *) &bufr [43]);
    sunftovaxf (header->gantry_tilt, (unsigned short *) &bufr [61]);
    sunftovaxf (header->gantry_rotation, (unsigned short *) &bufr [63]);
    sunftovaxf (header->bed_elevation, (unsigned short *) &bufr [65]);
    bufr [67] = header->rot_source_speed;
    bufr [68] = header->wobble_speed;
    bufr [69] = header->transm_source_type;
    sunftovaxf (header->axial_fov, (unsigned short *) &bufr [70]);
    sunftovaxf (header->transaxial_fov, (unsigned short *) &bufr [72]);
    bufr [74] = header->transaxial_samp_mode;
    bufr [75] = header->coin_samp_mode;
    bufr [76] = header->axial_samp_mode;
    sunftovaxf (header->calibration_factor, (unsigned short *) &bufr [77]);
    bufr [79] = header->calibration_units;
    bufr [80] = header->compression_code;
    bufr [175] = header->acquisition_type;
    bufr [176] = header->bed_type;
    bufr [177] = header->septa_type;
    bufr [188] = header->num_planes;
    bufr [189] = header->num_frames;
    bufr [190] = header->num_gates;
    bufr [191] = header->num_bed_pos;
    sunftovaxf (header->init_bed_position, (unsigned short *) &bufr [192]);
    for (i = 0; i < 15; i ++)
	sunftovaxf (header->bed_offset [i], (unsigned short *) &bufr [194 + 2 * i]);
    sunftovaxf (header->plane_separation, (unsigned short *) &bufr [224]);
    bufr [226] = header->lwr_sctr_thres;
    bufr [227] = header->lwr_true_thres;
    bufr [228] = header->upr_true_thres;
    sunftovaxf (header->collimator, (unsigned short *) &bufr [229]);

#ifdef _SWAPEM_
    /* swap the bytes */
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

    /* write main header at block 1 */
    #ifdef _DEBUG_
    printf ("cti_write_main_header: writing to block 1\n");
    #endif
    status = cti_wblk (fptr, 1, (char *) bufr, 1);

    fflush (fptr);
    free (bufr);
    if (status != EXIT_SUCCESS) {
	#ifdef _DEBUG_
	printf ("cti_write_main_header: cti_wblk returns %d\n", status);
	#endif
	return (status);
    }
    else
	return (EXIT_SUCCESS);
}
/*}}}  */

/*{{{  cti_write_scan_subheader */
/*******************************************************************************
    cti_write_scan_subheader - write a scan subheader into a matrix file.

    fptr - pointer to file.
    blknum - block offset at which to begin writing
    header - header to write
*******************************************************************************/
int cti_write_scan_subheader (FILE *fptr, int blknum, 
	const Scan_subheader *header)
{
    int i, status;
    short *bufr;
  
    /*{{{  calloc bufr*/
    bufr = (short *) calloc (MatBLKSIZE / sizeof (short), sizeof (short));
    if (!bufr) {
	#ifdef _DEBUG_
	printf ("cti_write_scan_subheader: calloc failed.\n");
	#endif
	return EXIT_FAILURE;
    }
    /*}}}  */

    /*{{{  fill in bufr*/
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
    bufr[66] = header->dimension_1;         /* x dimension */
    bufr[67] = header->dimension_2;         /* y_dimension */
    bufr[68] = header->smoothing;
    bufr[69] = header->processing_code;
    sunftovaxf (header->sample_distance, (unsigned short *) &bufr[73]);
    sunftovaxf (header->isotope_halflife, (unsigned short *) &bufr[83]);
    bufr[85] = header->frame_duration_sec;
    sunltovaxl (header->gate_duration, (unsigned short *) &bufr[86]);
    sunltovaxl (header->r_wave_offset, (unsigned short *) &bufr[88]);
    sunftovaxf (header->scale_factor, (unsigned short *) &bufr[91]);
    bufr[96] = header->scan_min;
    bufr[97] = header->scan_max;
    sunltovaxl (header->prompts, (unsigned short *) &bufr[98]);
    sunltovaxl (header->delayed, (unsigned short *) &bufr[100]);
    sunltovaxl (header->multiples, (unsigned short *) &bufr[102]);
    sunltovaxl (header->net_trues, (unsigned short *) &bufr[104]);
    for (i = 0; i < 16; i++) {
      sunftovaxf (header->cor_singles [i], (unsigned short *) &bufr [158 + 2 * i]);
      sunftovaxf (header->uncor_singles [i], (unsigned short *) &bufr [190 + 2 * i]);
    };
    sunftovaxf (header->tot_avg_cor, (unsigned short *) &bufr[222]);
    sunftovaxf (header->tot_avg_uncor, (unsigned short *) &bufr[224]);
    sunltovaxl (header->total_coin_rate, (unsigned short *) &bufr[226]);
    sunltovaxl (header->frame_start_time, (unsigned short *) &bufr[228]);
    sunltovaxl (header->frame_duration,(unsigned short *)  &bufr[230]);
    sunftovaxf (header->loss_correction_fctr, (unsigned short *) &bufr[232]);
    /*}}}  */

#ifdef _SWAPEM_
    /* swap the bytes */
    swab ((char *) bufr, (char *) bufr, MatBLKSIZE);
#endif

    #ifdef _DEBUG_
    printf ("cti_write_scan_subheader: writing to block %d\n", blknum);
    #endif

    status = cti_wblk (fptr, blknum, (char *) bufr, 1);
    free (bufr);

    return status;
}
/*}}}  */

/*{{{  cti_write_image */
/*******************************************************************************
    cti_write_image - write an image, including headers, into a
		matrix file.  

	fptr - pointer to file.
	matnum - matnum to use
	header - header to write
	data - data buffer containing image
	data_size - number of bytes in image
*******************************************************************************/
int cti_write_image (FILE *fptr, long matnum, const Image_subheader *header,
		    const short *data, int data_size)
{
    int nxtblk, nblocks;
    int status;

    nblocks = (data_size + (MatBLKSIZE - 1)) / MatBLKSIZE;
    nxtblk = cti_enter (fptr, matnum, nblocks);
    if (nxtblk <= 0)
	return (EXIT_FAILURE);
    status = cti_write_image_subheader (fptr, nxtblk, header);
    if (status != EXIT_SUCCESS)
	return (EXIT_FAILURE);
    /* swab ((char *) data, (char *) data, data_size);  */ /* ??? */
    status = cti_write_idata (fptr, nxtblk + 1, data, data_size);
    return status;
}
/*}}}  */

/*{{{  cti_write_scan */
/*******************************************************************************
    cti_write_scan - write a scan, including headers, into a matrix file.

    fptr - pointer to file.
    matnum - matnum to use
    header - header to write
    data - data buffer containing image
    data_size - number of bytes in image
*******************************************************************************/
int cti_write_scan (FILE *fptr, long matnum, const Scan_subheader *header,
		    const short *data, int data_size)
{
    int nxtblk, nblocks;
    int status;

    nblocks = (data_size + (MatBLKSIZE - 1)) / MatBLKSIZE;
    nxtblk = cti_enter (fptr, matnum, nblocks);
    if (nxtblk <= 0)
	return (EXIT_FAILURE);
    status = cti_write_scan_subheader (fptr, nxtblk, header);
    if (status != EXIT_SUCCESS)
	return (EXIT_FAILURE);
    status = cti_write_idata (fptr, nxtblk + 1, data, data_size);
    return status;
}
/*}}}  */

/*{{{  quantize16*/
/*******************************************************************************
	quantize16 - quantize a buffer of floats to fit inside a buffer of
		16-bit shorts, rescaling as necessary.

	in - buffer to rescale, containing count elements
	out - where to put the scaled result
	count - length of the buffer
	min, max - point to where to store the scaled min. and max. value.
*******************************************************************************/
void quantize16 (float *in, short *out, int count,
		 short *scaled_min, short *scaled_max, float *absmax)
{
    register int i;
    float    found_min, found_max, f_absmax;
    float    f_shrt_max = (float) SHRT_MAX;

    ffind_minmax (in, &found_min, &found_max, count);
    f_absmax = ((float) fabs (found_max) > (float) fabs (found_min)) ?
	      (float) fabs (found_max) : (float) fabs (found_min);
    if (f_absmax != 0) {
	for (i = 0; i < count; i++) {
	    if (in [i] >= 0)
		out [i] = (short) ((f_shrt_max / f_absmax) * in [i] + 0.5);
	    else
		out [i] = (short) ((f_shrt_max / f_absmax) * in [i] - 0.5);
	}
	*scaled_min = (short) (found_min * f_shrt_max / f_absmax);
	*scaled_max = (short) (found_max * f_shrt_max / f_absmax);
    }
    else { /* they must all be zero */
	for (i = 0; i < count; i++)
	    out [i] = (short) 0;
	*scaled_min = (short) 0;
	*scaled_max = (short) 0;
    }

    *absmax = f_absmax;

#if 0
printf ("\nquantize: floats -- absmax =%g, min = %g, max=%g\n", f_absmax,
	  found_min, found_max);
#endif

}
/*}}}  */

/*{{{  fill_teststring */
/*******************************************************************************
	fill_teststring - fill up a string with alphabetic characters

	str - string to fill
	len - size of the string (result will contain len - 1 chars plus a null)
*******************************************************************************/
void fill_teststring (char *str, int len)
{
	int i;
	char c = 'a';
	
	for (i = 0; i < len - 2; i++) {
		if (c > 'z')
			c = 'a';
		str [i] = c ++;
		}
	str [i] = '\0';
}
/*}}}  */

/*{{{  fill_testscan */
/*******************************************************************************
	fill_testscan - fill a scan buffer with test numbers

	buf - buffer to fill
	nblks - size of the buffer in blocks
*******************************************************************************/
void fill_testscan (short *buf, int nblks)
{
    short  i;
    short *b;

    b =  buf;
    for (i = 0; i < nblks * MatBLKSIZE / sizeof (short); i++)
	*b++ = i;  /* rand (); */
}
/*}}}  */

/*{{{  fill_main_header */
/*******************************************************************************
	fill_main_header - fill in various parts of a main subheader with 
		dummy values for a test

	shead - header to fill in
	mtype - matScanFile or matImageFile
*******************************************************************************/
void fill_main_header (Main_header *mhead, MatFileType mtype)
{
	int i;
	
	/* fill it in */
	mhead->isotope_halflife = 111.111;
	mhead->gantry_tilt = 222.222;
	mhead->gantry_rotation = 333.333;
	mhead->bed_elevation = 444.444;
	mhead->axial_fov = 555.555;
	mhead->transaxial_fov = 666.666;
	mhead->calibration_factor = 777.777;
	mhead->init_bed_position = 888.888;
	mhead->plane_separation = 999.999;
	mhead->collimator =1010.1010;
	for (i = 0; i < 15; i++)
		mhead->bed_offset[i] = 9876.54321;

	mhead->num_planes = 123;
	mhead->num_frames = 456;
	mhead->num_gates = 789;
	mhead->num_bed_pos = 10;
	mhead->sw_version = 11;
	mhead->data_type = 12;
	mhead->system_type = 13;
	mhead->file_type = (long) mtype;
	mhead->scan_start_day = 15;
	mhead->scan_start_month = 3;
	mhead->scan_start_year = 1962;
	mhead->scan_start_hour = 24;
	mhead->scan_start_minute = 60;
	mhead->scan_start_second = 00;
	mhead->rot_source_speed = 24; 
	mhead->wobble_speed = 25;
	mhead->transm_source_type = 26;
	mhead->transaxial_samp_mode = 27;
	mhead->coin_samp_mode = 28;
	mhead->axial_samp_mode = 29;
	mhead->calibration_units = 30;
	mhead->compression_code = 31; 
	mhead->acquisition_type = 32; 
	mhead->bed_type = 33;
	mhead->septa_type = 34;
	mhead->lwr_sctr_thres = 35;
	mhead->lwr_true_thres = 36;
	mhead->upr_true_thres = 37;
	
	fill_teststring (mhead->original_file_name, 20);
	fill_teststring (mhead->node_id, 10);
	fill_teststring (mhead->isotope_code, 8);
	fill_teststring (mhead->radiopharmaceutical, 32);
	fill_teststring (mhead->study_name, 12);
	fill_teststring (mhead->patient_id, 16);
	fill_teststring (mhead->patient_name, 32);
	fill_teststring (mhead->patient_age, 10);
	fill_teststring (mhead->patient_height, 10);
	fill_teststring (mhead->patient_weight, 10);
	fill_teststring (mhead->physician_name, 32);
	fill_teststring (mhead->operator_name, 32);
	fill_teststring (mhead->study_description, 32);
	fill_teststring (mhead->facility_name, 20);
	fill_teststring (mhead->user_process_code, 10);
	mhead->patient_sex = 1;                 /* ?? */
	mhead->patient_dexterity = 2;   /* ?? */
}
/*}}}  */

/*{{{  fill_scan_subheader */
/*******************************************************************************
	fill_scan_subheader - fill in various parts of a scan subheader with 
		dummy values for a test

	shead - header to fill in
	nprojs - 
	nviews -
*******************************************************************************/
void fill_scan_subheader (Scan_subheader *shead, int nprojs, int nviews)
{
	int i;
	
	/* fill it in */
	shead->sample_distance = 1234.5;                                                /* ?? */
	shead->isotope_halflife = 67.89;                                                /* ?? */
	shead->scale_factor = 123.45;                                                   /* ?? */
	shead->loss_correction_fctr = 6.7;                                              /* ?? */
	shead->tot_avg_cor = 89.012;                                                    /* ?? */
	shead->tot_avg_uncor = 34.56;                                                   /* ?? */
	
	for (i = 0; i < 16; i++)
		shead->cor_singles [i] = 123.45;                                        /* ?? */
	for (i = 0; i < 16; i++)
		shead->uncor_singles [i] = 678.9;                                       /* ?? */

	shead->gate_duration = 123;                                                             /* ?? */
	shead->r_wave_offset = 456;                                                             /* ?? */
	shead->prompts = 789;                                                                   /* ?? */
	shead->delayed = 1011;                                                                  /* ?? */
	shead->multiples = 1213;                                                                /* ?? */
	shead->net_trues = 1415;                                                                /* ?? */
	shead->total_coin_rate = 1617;                                                  /* ?? */
	shead->frame_start_time = 1819;                                                 /* ?? */
	shead->frame_duration = 2021;                                                   /* ?? */
	
	shead->data_type = 22;                                                          /* ??? */
	shead->dimension_1 = nprojs;
	shead->dimension_2 = nviews;
	shead->smoothing = 23;                                                          /* ?? */
	shead->processing_code = 24;                                            /* ?? */
	shead->frame_duration_sec = 25;                                         /* ?? */
	shead->scan_min = 26;                                                           /* ?? */
	shead->scan_max = 32767;                                                        /* ?? */
}
/*}}}  */

/*{{{  fill_image_subheader */
/*******************************************************************************
	fill_image_subheader - fill in various parts of the image subheader with 
		plausible values

	header - header to fill in
	scale - scaling factor applied to quantize the image
	image_max - max. value of the unquantized image
*******************************************************************************/
void fill_image_subheader (Image_subheader *ihead, float zoom, int isize, 
			float image_max)
{
	int i;
	
	/* fill in some values */
	ihead->x_origin = 0.0;
	ihead->y_origin = 0.0;
	ihead->recon_scale = zoom;
	ihead->quant_scale = image_max / 32767.;
	ihead->pixel_size = 1.0;
	ihead->slice_width = 1.0;
	
	/* fill in some more values */
	ihead->data_type = 2;                           /* ?? */
	ihead->num_dimensions = 2;
	ihead->dimension_1 = ihead->dimension_2 = isize;
	ihead->image_min = 0;
	ihead->image_max = 32767;

	/** don't know about these, set them to 0 **/
	ihead->image_rotation = 0.0;
	ihead->plane_eff_corr_fctr = 0.0;
	ihead->decay_corr_fctr = 0.0;
	ihead->loss_corr_fctr = 0.0;
	ihead->ecat_calibration_fctr = 0.0;
	ihead->well_counter_cal_fctr = 0.0;
	for (i = 0; i < 6; i++)
		ihead->filter_params[i] = 0.0;
	ihead->frame_duration = 0;
	ihead->frame_start_time = 0;
	ihead->recon_duration = 0;
	ihead->scan_matrix_num = 0;
	ihead->norm_matrix_num = 0;
	ihead->atten_cor_matrix_num = 0;
	ihead->slice_location = 0;
	ihead->recon_start_hour = 0;
	ihead->recon_start_minute = 0;
	ihead->recon_start_sec = 0;
	ihead->filter_code = 0;
	ihead->processing_code = 0;
	ihead->quant_units = 0;
	ihead->recon_start_day = 0;
	ihead->recon_start_month = 0;
	ihead->recon_start_year = 0;
}
/*}}}  */

/*{{{  dump_main_header */
/*******************************************************************************
	dump_main_header - dump various parts of a main subheader into file

	fptr - file to write into 
	shead - header to view
*******************************************************************************/
void dump_main_header (FILE *fptr, const Main_header *mhead)
{
	int i;
	FILE *dptr;
	
	if (!fptr) {
		printf ("dumping to mainheader.dmp\n");
		dptr = fopen ("mainheader.dmp", "w+");
	} else
		dptr = fptr;
	
	/* print out all the fields */
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
	for (i = 0; i < 15; i++)
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

	if (!fptr)
		fclose (dptr);
}
/*}}}  */

/*{{{  dump_scan_subheader */
/*******************************************************************************
	dump_scan_subheader - dump scan subheader

	fptr - file to write into 
	shead - header to fill in
*******************************************************************************/
void dump_scan_subheader (FILE *fptr, const Scan_subheader *shead)
{
	int i;
	FILE *dptr;
	
	if (!fptr) {
		printf ("dumping to scansubheader.dmp\n");
		dptr = fopen ("scansubheader.dmp", "w+");
	} else
		dptr = fptr;
	
	fprintf (dptr, "    SCAN SUBHEADER\n    --------------\n");
	fprintf (dptr, "sample_distance = %g\n", shead->sample_distance);
	fprintf (dptr, "isotope_halflife = %g\n", shead->isotope_halflife);
	fprintf (dptr, "scale_factor = %g\n", shead->scale_factor);
	fprintf (dptr, "loss_correction_fctr = %g\n", shead->loss_correction_fctr);
	fprintf (dptr, "tot_avg_cor = %g\n", shead->tot_avg_cor);
	fprintf (dptr, "tot_avg_uncor = %g\n\n", shead->tot_avg_uncor);
	
	for (i = 0; i < 16; i++)
		fprintf (dptr, "cor_singles [%d] = %g\n", i, shead->cor_singles [i]);
	for (i = 0; i < 16; i++)
		fprintf (dptr, "uncor_singles [%d] = %g\n", i, shead->uncor_singles [i]);

	fprintf (dptr, "\ngate_duration = %ld\n", shead->gate_duration);
	fprintf (dptr, "r_wave_offset = %ld\n", shead->r_wave_offset);
	fprintf (dptr, "prompts = %ld\n", shead->prompts);
	fprintf (dptr, "delayed = %ld\n", shead->delayed);
	fprintf (dptr, "multiples = %ld\n", shead->multiples);
	fprintf (dptr, "net_trues = %ld\n", shead->net_trues);
	fprintf (dptr, "total_coin_rate = %ld\n", shead->total_coin_rate);
	fprintf (dptr, "frame_start_time = %ld\n", shead->frame_start_time);
	fprintf (dptr, "frame_duration = %ld\n\n", shead->frame_duration);
	
	fprintf (dptr, "data_type = %d\n", shead->data_type);
	fprintf (dptr, "dimension_1 = %d \t", shead->dimension_1);
	fprintf (dptr, "dimension_2 = %d\n", shead->dimension_2);
	fprintf (dptr, "smoothing = %d\n", shead->smoothing);
	fprintf (dptr, "processing_code = %d\n", shead->processing_code);
	fprintf (dptr, "frame_duration_sec = %d\n", shead->frame_duration_sec);
	fprintf (dptr, "scan_min = %d \t", shead->scan_min);
	fprintf (dptr, "scan_max = %d\n", shead->scan_max);
	fprintf (dptr, "    -- end --\n");

	fflush (dptr);
	if (!fptr)
		fclose (dptr);
}
/*}}}  */

/*{{{  dump_image_subheader */
/*******************************************************************************
	dump_image_subheader - print image subheader to file

	fptr - file to write into 
	header - header to dump
*******************************************************************************/
void dump_image_subheader (FILE *fptr, const Image_subheader *ihead)
{
	int i;
	FILE *dptr;
	
	if (!fptr) {
		printf ("dumping to imagesubheader.dmp\n");
		dptr = fopen ("imagesubheader.dmp", "w+");
	} else
		dptr = fptr;
	
	fprintf (dptr, "    IMAGE SUBHEADER\n    ---------------\n");
	fprintf (dptr, "x_origin = %g\t", ihead->x_origin);
	fprintf (dptr, "y_origin = %g\n", ihead->y_origin);
	fprintf (dptr, "recon_scale = %g\n", ihead->recon_scale);
	fprintf (dptr, "quant_scale = %g\n", ihead->quant_scale);
	fprintf (dptr, "pixel_size = %g\n", ihead->pixel_size);
	fprintf (dptr, "slice_width = %g\n", ihead->slice_width);
	
	fprintf (dptr, "image_rotation = %g\n", ihead->image_rotation);
	fprintf (dptr, "plane_eff_corr_fctr = %g\n", ihead->plane_eff_corr_fctr);
	fprintf (dptr, "decay_corr_fctr = %g\n", ihead->decay_corr_fctr);
	fprintf (dptr, "loss_corr_fctr = %g\n", ihead->loss_corr_fctr);
	fprintf (dptr, "ecat_calibration_fctr = %g\n", ihead->ecat_calibration_fctr);
	fprintf (dptr, "well_counter_cal_fctr = %g\n", ihead->well_counter_cal_fctr);

	for (i = 0; i < 6; i++)
		fprintf (dptr, "filter_params [%d] = %g\n", i, ihead->filter_params [i]);
		
	fprintf (dptr, "data_type = %d\n", ihead->data_type);
	fprintf (dptr, "num_dimensions = %d\n", ihead->num_dimensions);
	fprintf (dptr, "dimension_1 = %d \t", ihead->dimension_1);
	fprintf (dptr, "dimension_2 = %d\n", ihead->dimension_2);
	fprintf (dptr, "image_min = %d \t", ihead->image_min);
	fprintf (dptr, "image_max = %d\n", ihead->image_max);

	fprintf (dptr, "frame_duration = %ld\n", ihead->frame_duration);
	fprintf (dptr, "frame_start_time = %ld\n", ihead->frame_start_time);
	fprintf (dptr, "recon_duration = %ld\n", ihead->recon_duration);
	fprintf (dptr, "scan_matrix_num = %ld\n", ihead->scan_matrix_num);
	fprintf (dptr, "norm_matrix_num = %ld\n", ihead->norm_matrix_num);
	fprintf (dptr, "atten_cor_matrix_num = %ld\n",
			ihead->atten_cor_matrix_num);
	fprintf (dptr, "slice_location = %d\n", ihead->slice_location);
	
	fprintf (dptr, "recon_start_day = %d\t", ihead->recon_start_day);
	fprintf (dptr, "recon_start_month = %d\t", ihead->recon_start_month);
	fprintf (dptr, "recon_start_year = %d\n", ihead->recon_start_year);
	fprintf (dptr, "recon_start_hour = %d\t", ihead->recon_start_hour);
	fprintf (dptr, "recon_start_minute = %d\t", ihead->recon_start_minute);
	fprintf (dptr, "recon_start_sec = %d\n", ihead->recon_start_sec);
	fprintf (dptr, "filter_code = %d\n", ihead->filter_code);
	fprintf (dptr, "processing_code = %d\n", ihead->processing_code);
	fprintf (dptr, "quant_units = %d\n", ihead->quant_units);
	fprintf (dptr, "    -- end --\n");

	if (!fptr)
		fclose (dptr);
}
/*}}}  */

/*{{{  lfind_minmax */
/*******************************************************************************
	lfind_minmax - find minimum and maximum values in a buffer of long ints
	
	buf - array of bufsize long ints
	min - will be set to the smallest positive int in buf
	max - will be set to the largest int contained in buf
	bufsize - number of elements in buf
*******************************************************************************/
void lfind_minmax (long *buf, long *min, long *max, int bufsize)
{
	register int i;
	register long foundmax, foundmin, *b;
	
	foundmax = 0;
	foundmin = 0;
	b = buf;
	for (i = 0; i < bufsize; i++) {
		if (*b > foundmax) 
		foundmax = *b;
		if ((*b < foundmin ) && (*b > 0)) 
		foundmin = *b;
	    ++b;
	}
	*min = foundmin;
	*max = foundmax;
}
/*}}}  */

/*{{{  sfind_minmax */
/*******************************************************************************
	sfind_minmax - find minimum and maximum values in a buffer of shorts
	
	buf - array of bufsize shorts
	min - will be set to the smallest positive short in buf
	max - will be set to the largest short contained in buf
	bufsize - number of elements in array
*******************************************************************************/
void sfind_minmax (short *buf, short *min, short *max, int bufsize)
{
	register int i;
	register short  *b, foundmax, foundmin;
	
	foundmax = SHRT_MIN;
	foundmin = SHRT_MAX;
	b = buf;
	for (i = 0; i < bufsize; i++) {
		if (*b > foundmax) 
		foundmax = *b;
		if (*b < foundmin) 
		foundmin = *b;
	    ++b;
	}
	*min = foundmin;
	*max = foundmax;
}
/*}}}  */

/*{{{  ffind_minmax*/
/*******************************************************************************
	ffind_minmax - find minimum and maximum values in a buffer of floats
	
	buf - array of count floats
	min - will be set to the smallest positive float in buf
	max - will be set to the largest float contained in buf
	count - number of elements in buf
	
*******************************************************************************/
void ffind_minmax (float *buf, float *min, float *max, int count)
{
	register int i;
	float foundmax, foundmin, *b;
	
	foundmax = FLT_MIN;
	foundmin = FLT_MAX;
	b = buf;
	for (i = 0; i < count; i++, b++) {
	    if (*b > foundmax)
		foundmax = *b;
	    if (*b < foundmin)
		foundmin = *b;
	}
	*min = foundmin;
	*max = foundmax;
}
/*}}}  */

#ifdef _PLATFORM_TRANSPUTER_
/*{{{  bcopy */
/*******************************************************************************
	bcopy - copy an array to another array, a byte at a time.  Arrays must
		not overlap.
	
	from - input array.
	to - output array.
*******************************************************************************/
void bcopy (const char *from, char *to, int length)
{
	memcpy (to, from, length);
}
/*}}}  */
#endif

#ifdef _PLATFORM_TRANSPUTER_
/*{{{  sincos */
/*******************************************************************************
	sincos - sin and cos.
	
	t - input angle
	sint - sine of angle
	cost - cosine of angle
*******************************************************************************/
void sincos (float t, double *sint, double *cost)
{
	*sint = sin (t);
	*cost = cos (t);
}
/*}}}  */
#endif

/*{{{  swab */
/*******************************************************************************
	swab - copy array, swapping bytes as we go
	
	from - input array
	to - output array (may be same as input array)ibytes
	length - total number of bytes to copy
*******************************************************************************/
void swab (char *from, char *to, int length)
{
    register int i;
    register char temp;

    for (i = 0; i < length; i+= 2) {
	temp = from [i + 1];
	to [i + 1] = from [i];
	to [i] = temp;
    }
#if 0
    register char *f, *t, *f1, *t1;
    register int i;

    f = from;       f1 = from + 1;
    t = to;         t1 = t + 1;

    for (i = 0; i < length; i += 2) {
	temp = *f1++;
	*t1++ = *f++;
	*t++ = temp;
    }
#endif
}
/*}}}  */

/*{{{  swaw */
/*******************************************************************************
	swaw - copy array, swapping 16-bit words as we go.  
	
	from - data buffer to copy.
	to - where to copy the data.  may be same as (or overlap) from.
	length - number of 16-bit words to swap
*******************************************************************************/
void swaw (short *from, short *to, int length)
{
    register short temp;
    register int i;

    for (i = 0; i < length; i +=2) {
	temp = from [i + 1];
	to [i + 1] = from [i];
	to [i] = temp;
    }
#if 0
    register short *f, *t, *f1, *t1;
    register int i;
    f = from;       f1 = from + 1;
    t = to;         t1 = t + 1;

    for (i = 0; i < length; i += 2) {
	temp = *f1++;
	*t1++ = *f++;
	*t++ = temp;
    }
#endif
}
/*}}}  */

/*{{{  get_vax_float */
/*******************************************************************************
	get_vax_float - get indexed value from buffer, a vax float, and return it
		as an IEEE float.
	
	bufr - input data buffer.
	off - offset into buffer of first 16-bit half of the 32-bit value to convert.
*******************************************************************************/

float get_vax_float (const unsigned short *bufr, int off)
{
	unsigned short t1, t2;
	union {unsigned long t3; float t4;} test;

	if (bufr [off] == 0 && bufr [off + 1] == 0) 
		return ((float) 0.0);
	t1 = bufr [off] & 0x80ff;
	t2 = (((bufr [off]) & 0x7f00) + 0xff00) & 0x7f00;
	test.t3 = ((long) t1 + (long) t2) << 16;
	test.t3 = test.t3 + bufr [off + 1];
	
	return (test.t4);
}

/*}}}  */

/*{{{  get_vax_long */
/*******************************************************************************
	get_vax_long - get the indexed value from a buffer, a 32-bit vax long, and
		convert it by swapping the words.
		(vax int = vax long int = 32 bits; vax short = 16 bits)
	
	bufr - input data buffer.
	off - index into buffer of first 16-bit word of the 32-bit value to convert.
*******************************************************************************/
long get_vax_long (const unsigned short *bufr, int off)
{
  /* KT 30/11/98 new ifdef */
#ifdef _SWAPEM_
	return ((bufr [off + 1] << 16) + bufr [off]);
#else
	/* KT 30/11/98 new code */
	return *(long *) (&bufr[off]);
#endif
}
/*}}}  */

/*{{{  sunltovaxl */
/*******************************************************************************
	sunltovaxl - convert a sun long int to a vax long int -- i.e. swap the
		16-bit words of the 32-bit long.
		(sun long = sun int = 32 bits)
	
	in - value to convert.
	out - result.
*******************************************************************************/
void sunltovaxl (const long in, unsigned short out [2])
{
  /* KT 30/11/98 new ifdef */
#ifdef _SWAPEM_
	out [0] = (in & 0x0000FFFF);
	out [1] = (in & 0xFFFF0000) >> 16;
#else
	/* KT 30/11/98 new code */
        out[0] = (in & 0xFFFF0000) >> 16;
        out [1] = (in & 0x0000FFFF);
#endif
}
/*}}}  */

/*{{{  sunftovaxf */
/*******************************************************************************
	sunftovaxf - convert a sun float to a vax float
	
	in - value to convert.
	out - result.
*******************************************************************************/
void sunftovaxf (const float in, unsigned short out [2])
{


	union {
		unsigned short t [2]; 
		float t4;
	      } test;	unsigned short exp;

	out [0] = 0;
	out [1] = 0;

	#ifdef _SWAPEM_
	swaw ((short *) &in,(short int *) &test.t[0], 2);
	#else
	test.t4 = in;
	#endif
	if (test.t4 == 0.0)     /* all set, it is zero!! */
		return;
	exp = ((test.t [0] & 0x7f00) + 0x0100) & 0x7f00; 
	test.t [0] = (test.t [0] & 0x80ff) + exp;  
	out [0] = ((test.t [1] + 256) & 0xff00) | (test.t [1] & 0x00ff);
	out [1] = ((test.t [0] - 256) & 0xff00) | (test.t [0] & 0x00ff);

}
/*}}}  */

/*{{{  usage*/
void usage (const char *name, const char *msg)
{
	fprintf (stderr, "%s%s", name, msg);
	exit (1);
}
/*}}}  */


//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
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
*/

/*! 
  \file
  \ingroup ECAT
  \brief Declaration of ECAT 6 CTI functions to access data
  \author Damien Sauge
  \author Kris Thielemans
  \author PARAPET project
*/
/* History:
  KT 18/08/2000 added  file_data_to_host, get_attnheaders
  KT 11/01/2002 added  normalisation file things, removed get_attndata
  KT 13/01/2008 replace original CTI code with calls to LLN matrix library:
                introduced mhead_ptr in various functions
                have #define STIR_ORIGINAL_ECAT6 to be able to switch between old and new version
*/

#ifndef __stir_IO_ecat6_utils_h
#define __stir_IO_ecat6_utils_h

#include "stir/IO/ecat6_types.h"

#include <stdlib.h>

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT6

// CTI UTILITIES

/*! \brief Encode scan information into a single, incomprehensible number.
  \ingroup ECAT
*/
long    cti_numcod (int frame, int plane, int gate, int data, int bed);
/*!
  \brief Unpack encoded data into a nice struct. Reverse of cti_numcod ().
  \ingroup ECAT

  \param matnum   the thingy to decode
  \param matval   struct containing the decoded values from matnum
*/
void    cti_numdoc (long matnum, Matval *matval);
/*!
  \brief Get sinogram plane from ring pair.
  \ingroup ECAT

  \param ring0   first ring in ring pair
  \param ring1   second ring in ring pair
*/
int     cti_rings2plane (short nrings, short ring0, short ring1);

// FILE ACCESS

/*!
  \brief Open a file and write main header to it. Returns a pointer to the file, or 0 if unsuccessful.
  \ingroup ECAT

  \param fname   name of file to open
  \param mhead   pointer to main header struct to copy into the file
*/
FILE    *cti_create (const char *fname, const ECAT6_Main_header *mhead);

/*!
  \brief Read from a matrix file starting at the given block. Returns EXIT_SUCCESS if all went well.
  \ingroup ECAT
	
  \param fptr    file pointer
  \param blkno   first block to read
  \param nblks   number of blocks to read
*/
int     cti_rblk (FILE *fptr, int blkno, void *bufr, int nblks);
/*!
  \brief Write blocks from buffer into file. Returns EXIT_SUCCESS if successful.
  \ingroup ECAT

  \param fptr    pointer to file.
  \param blkno   position in file of first block to write.
  \param nblks   number of blocks to write.
*/
int     cti_wblk (FILE *fptr, int blkno, void *bufr, int nblks);
/*!
  \brief Read header data from a file and place it into a ECAT6_Main_header struct. Returns EXIT_SUCCESS if no error.
  \ingroup ECAT

  \param fptr  file containing the header data
  \param h     struct to fill with header info
*/
int     cti_read_ECAT6_Main_header (FILE *fptr, ECAT6_Main_header *h);
/*!
  \brief Read header data from a file and place it into a Scan_subheader 
  struct. Returns EXIT_SUCCESS if no error.
  \ingroup ECAT

  \param fptr    file containing the header data
  \param blknum  block number at which to begin reading
  \param h       struct to fill
*/
int     cti_read_scan_subheader (FILE *fptr, const ECAT6_Main_header *, int blknum, Scan_subheader *);


/*!
  \brief Read header data from a file and place it into a Attn_subheader 
  struct. Returns EXIT_SUCCESS if no error.
  \ingroup ECAT

  \param fptr    file containing the header data
  \param blknum  block number at which to begin reading
  \param header_ptr       struct to fill
*/
int cti_read_attn_subheader(FILE* fptr, const ECAT6_Main_header *h, int blknum, Attn_subheader *header_ptr);


/*!
  \brief Read header data from a file and place it into a Norm_subheader 
  struct. Returns EXIT_SUCCESS if no error.
  \ingroup ECAT

  \param fptr    file containing the header data
  \param blknum  block number at which to begin reading
  \param header_ptr       struct to fill
*/
int cti_read_norm_subheader(FILE* fptr, const ECAT6_Main_header *h, int blknum, Norm_subheader *header_ptr);

/*!
  \brief Read header data from a file and place it into an Image_subheader 
  struct. Returns EXIT_SUCCESS if no error.
  \ingroup ECAT

  \param fptr    file containing the header data
  \param blknum  block number at which to begin reading
  \param header_ptr       struct to fill
*/
int	cti_read_image_subheader (FILE *fptr, const ECAT6_Main_header *h, int   blknum, Image_subheader *header_ptr);

/*!
  \brief Create entry in file corresponding to matnum, and return offset of 
  next block. Or some such. Returns 0 if there was an error.
  \ingroup ECAT

  \param fptr    pointer to file.
  \param mhead_ptr pointer to a main header
  \param matnum  desired matnum.
  \param nblks   number of blocks
*/
int     cti_enter (FILE *fptr, const ECAT6_Main_header* mhead_ptr, long matnum, int nblks);
/*! 
  \brief Look up a "matrix number" in the file and return the MatDir entry 
  for it. Returns 0 if the lookup was NOT successful.
  \ingroup ECAT

  \param fptr    file containing all the tabulated information.
  \param mhead_ptr pointer to a main header
  \param matnum  index.
  \param entry   where to put the result.
*/
int     cti_lookup (FILE *fptr, const ECAT6_Main_header* mhead_ptr, long matnum, MatDir *entry);
/*!
  \brief Write data in blocks from buffer into file.
  \ingroup ECAT
  \internal 
  \param fptr    pointer to file.
  \param blk     offset (in blocks) in file of first block to write.
  \param data    buffer to write
  \param isize  number of bytes to write.  (should be multiple of MatBLKSIZE)
*/
int     cti_write_idata (FILE *fptr, int blk, const short *data, int isize);
/*!
  \brief Write an image subheader into a matrix file. Returns 0 if successful.
  \ingroup ECAT

  \param fptr    pointer to file.
  \param blknum  offset (in blocks) in file of first block to write.
  \param header_ptr  header to write
*/
int     cti_write_image_subheader (FILE *fptr, const ECAT6_Main_header *h, int blknum, const Image_subheader *header_ptr);
/*!
  \brief Write an image main header into a matrix file. Returns 0 if successful.
  \ingroup ECAT
	
  \param fptr    pointer to file.
  \param header_ptr  header to write
*/
int     cti_write_ECAT6_Main_header (FILE *fptr, const ECAT6_Main_header *header_ptr);
/*!
   \brief Write a scan subheader into a matrix file.
  \ingroup ECAT

   \param fptr    pointer to file.
   \param blknum  block offset at which to begin writing
   \param header_ptr  header to write
*/
int     cti_write_scan_subheader (FILE *fptr, const ECAT6_Main_header *h, int blknum, const Scan_subheader *header_ptr);

/*!
  \brief Write an image, including headers, into a matrix file.
  \ingroup ECAT
  \internal

  \param fptr       pointer to file.
  \param mhead_ptr pointer to main header
  \param matnum     matnum to use
  \param header_ptr     header to write
  \param data       data buffer containing image
  \param data_size  number of bytes in image

  \warning  \a data_size has to be a multiple of MatBLKSIZE, even if there is 
  actually less data in the sinogram. Similarly, the buffer has to be
  allocated to accomodate for this size.
*/
int     cti_write_image (FILE *fptr, long matnum, const ECAT6_Main_header *mhead_ptr, const Image_subheader *header_ptr,
			 const short *data, int data_size);
/*!
  \brief Write a scan, including headers, into a matrix file.
  \ingroup ECAT
  \internal

  \param fptr       pointer to file.
  \param mhead_ptr pointer to main header
  \param matnum     matnum to use
  \param header_ptr     header to write
  \param data       data buffer containing sinogram
  \param data_size  number of bytes in sinogram

  \warning  \a data_size has to be a multiple of MatBLKSIZE, even if there is 
  actually less data in the sinogram. Similarly, the buffer has to be
  allocated to accomodate for this size.
*/
int     cti_write_scan (FILE *fptr, long matnum, const ECAT6_Main_header *mhead_ptr, const Scan_subheader *header_ptr,
			const short *data, int data_size);
/*!
  \brief Read main header and subheader from scan file. Returns EXIT_SUCCESS if no error.
  \ingroup ECAT

  \param fptr       pointer to scan file
  \param matnum     matnum for scan       
  \param mhead      where to put the main header
  \param shead      where to put the subheader
  \param scanParams where to put the scan parameters
*/
int     get_scanheaders (FILE *fptr, long matnum, ECAT6_Main_header *mhead, 
			 Scan_subheader *shead, ScanInfoRec *scanParams);
/*!
  \brief Read scan data from file. Returns EXIT_FAILURE if the data could not be read.
  \ingroup ECAT

  \param fptr       scan file
  \param scan       buffer for the data, must be preallocated
  \param scanParams data parameters

  The data will be stored according to the data_type in scanParams, but 
  converted (i.e. byte-swapped etc.) to the hosts native format.

  \warning  the \a scan buffer has to be allocated with size a multiple of 
  MatBLKSIZE, even if there is actually less data in the sinogram.
*/
// KT 18/05/2000 changed type from short to char

int     get_scandata (FILE *fptr, char *scan, ScanInfoRec *scanParams);


/*!
  \brief Read main header and subheader from attn file. Returns EXIT_SUCCESS if no error.
  \ingroup ECAT

  \param fptr       pointer to attn file
  \param matnum     matnum for attn       
  \param mhead      where to put the main header
  \param shead      where to put the subheader
  \param attnParams where to put the attn parameters
*/
int     get_attnheaders (FILE *fptr, long matnum, ECAT6_Main_header *mhead, 
			 Attn_subheader *shead, ScanInfoRec *attnParams);


/*!
  \brief Read main header and subheader from normalisation file. Returns EXIT_SUCCESS if no error.
  \ingroup ECAT

  \param fptr       pointer to nrm file
  \param matnum     matnum for nrm       
  \param mhead      where to put the main header
  \param shead      where to put the subheader
  \param nrmParams where to put the nrm parameters
*/
int     get_normheaders (FILE *fptr, long matnum, ECAT6_Main_header *mhead, 
			 Norm_subheader *shead, ScanInfoRec *nrmParams);

// OTHER UTILITIES

//! Convert data in buffer dptr to native host format
int file_data_to_host(char *dptr, int nblks, int dtype);

#ifdef STIR_ORIGINAL_ECAT6

/*
  \brief Find minimum and maximum values in a buffer of floats
  \ingroup ECAT
	
  \param buf    array of count floats
  \param min    will be set to the smallest positive float in buf
  \param max    will be set to the largest float contained in buf
  \param count  number of elements in buf
*/
void    ffind_minmax (float *buf, float *min, float *max, int bufsize);
/*
  \brief Find minimum and maximum values in a buffer of shorts
	
  \param buf      array of bufsize shorts
  \param min      will be set to the smallest positive short in buf
  \param max      will be set to the largest short contained in buf
  \param bufsize  number of elements in array
*/
void    sfind_minmax (short *buf, short *min, short *max, int bufsize);
/*
  \brief Copy array, swapping bytes as we go
	
  \param from    input array
  \param to      output array (may be same as input array)ibytes
  \param length  total number of bytes to copy
*/
void    swab (char *from, char *to, int length);
/*
  \brief Copy array, swapping 16-bit words as we go.  
	
  \param from    data buffer to copy.
  \param to      where to copy the data.  may be same as (or overlap) from.
  \param length  number of 16-bit words to swap
*/
void    swaw (short *from, short *to, int length);
/*
  \brief Get indexed value from buffer, a vax float, and return it as an IEEE float.
  (a swab has to be performed first on bigendian machines)

  \param bufr  input data buffer.
  \param off   offset into buffer of first 16-bit half of the 32-bit value to convert.
*/
float   get_vax_float (const unsigned short *bufr, int off);
/*
  \brief Get the indexed value from a buffer, a 32-bit vax long, and convert it by swapping the words.
  (a swab has to be performed first on bigendian machines)

  \param bufr  input data buffer.
  \param off   index into buffer of first 16-bit word of the 32-bit value to convert.
*/
long    get_vax_long (const unsigned short *bufr, int off);
/*!
  \brief Dump various parts of a main subheader into file

  \param fptr   file to write into
  \param mhead  header to view
*/
void    dump_ECAT6_Main_header (FILE *fptr, const ECAT6_Main_header *mhead);
/*
  \brief Convert a sun long int to a vax long int, i.e. swap the 16-bit words of the 32-bit long. 
  (sun long = sun int = 32 bits)
  (a swab has to be performed afterwards on bigendian machines)

  \param in   value to convert.
  \param out  result.
*/
void    hostltovaxl (const long in, unsigned short out [2]);
/*
  \brief Convert a host float to a vax float
  (a swab has to be performed afterwards on bigendian machines)

  \param in   value to convert.
  \param out  result.
*/
void    hostftovaxf (const float in, unsigned short out [2]);


/*! \brief Fill main header with negative or default values.*/
ECAT6_Main_header     main_zero_fill();
#endif // STIR_ORIGINAL_ECAT6
/*! \brief Fill scan subheader with negative or default values.*/
Scan_subheader  scan_zero_fill();
/*! \brief Fill image subheader with negative or default values.*/
Image_subheader img_zero_fill();

END_NAMESPACE_ECAT6
END_NAMESPACE_ECAT
END_NAMESPACE_STIR
#endif

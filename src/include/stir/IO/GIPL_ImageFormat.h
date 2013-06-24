//
/*
 Copyright (C) 2012 - 2013, King's College London
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
 (at your option) any later version.
 
 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 See STIR/LICENSE.txt for details
 */
/*!
 \file
 \ingroup IO
 \brief Applies the dilation filter (i.e. voxel=max(neighbours))
 \author Charalampos Tsoumpas [extension of original by Buerger, see below]
 
 $Date$
 $Revision$
 */

/*-----
 Original Copyright (c) 2012, Division of Imaging Sciences & Biomedical Engineering,
King's College London

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, 
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation 
      and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-----*/

// -------------------------------------------------------------------------
/**
*	\class			Image
*	\brief			Reading and writing gipl files
*
*	\date			2010
*	\author			Colin Studholme, (original VoxObj concept)
*					Derek Hill,
*					Jason Zhao,
*					Cliff Ruff,
*					Abhir H. Bhalerao,
*					modified by Christian Buerger
*					adapted for STIR by Charalampos Tsoumpas
*	\note			Copyright (c) King's College London, The Rayne Institute. 
*					Division of Imaging Sciences, 4th floow Lambeth Wing, St Thomas' Hospital.
*					All rights reserved.
*/
// -------------------------------------------------------------------------

#ifndef __GIPL_IMAGE_H
#define __GIPL_IMAGE_H

#include <vector>
#include <stdio.h>
#include <fstream>

#define _SHORT									0
#define _FLOAT									1

//#include "ByteSwap.h"

// -------------------------------------------------------------------------
//   Image class
// -------------------------------------------------------------------------

class Image
{
public:
	
	Image();
	Image(const int num_voxels, const short data_type_case);

	~Image();

	// Image data types
	short* vData;
	float* vData_f;

	// Dimensions of image in voxels and time (x,y,z,t). Bytes: 0 to 7.
	short m_dim[4];

	// Image data type (binary, char, short etc). Bytes: 8 and 9.
	short m_image_type;

	// Dimensions of individual image voxels (x,y,z,t). Bytes: 10 to 25.
	float m_pixdim[4];

	// Patient description string. Bytes: 26 to 105.
	char m_patDesc[80];

	// Matrix (transformation?) stored with image. Bytes: 106 to 153.
	float m_matrixElements[12];

	// Image identification number. Bytes: 154 to 157.
	int m_identifier;

	// Spare storage string. Bytes: 158 to 185.
	char m_spare[28];

	// Orientation of the image (AP, lateral, axial etc). Byte 186.
	char m_orientationFlag;

	// GIPL version number. Byte 187.
	char m_flag1;

	// Minimum data value. Bytes: 188 to 195.
	double m_min;

	// Maximum data value. Bytes: 196 to 203.
	double m_max;

	// Image origin in mm. Bytes: 204 to 235.
	double m_origin[4];

	// Pixel value offset. Bytes: 236 to 239.
	float m_pixval_offset;

	// Pixel value calibration. Bytes: 240 to 243.
	float m_pixval_cal;

	// User defined float 1. Bytes: 244 to 247.
	float m_user_def1;

	// User defined float 2. Bytes: 248 to 251.
	float m_user_def2;

	// GIPL magic number. Bytes: 252 to 255.
	unsigned int m_magic_number;

	// Data dimension
	int ImageDimension;

	// Parameter dimension
	int ParametersDimension;

	// Data vector length
	int MaxLength;

	// Max and min gray value
	float iMax;
	float iMin;

	// Offset vector
	int ImageOffset[2];

	// Center of rotation
	float vCenter[4];

	// Downscaling properties
	short vDownsample[3];

	// Input output functions
	void GiplRead(char* filename);
	void ReadGiplHeader(std::fstream* myFile);
	void GiplWrite(const char* filename);
	void WriteGiplHeader(std::fstream* myFile);
	void ByteSwapHeader();
	
	// Get minimum and maximum gray values
	void GetMinMaxValue();

	// Initialize and copy images
	void Initialize(Image* Input, short type);
	void Initialize(Image* Input);
	void Zeros(Image* Input, short type);
	void Zeros();
	void Ones(Image* Input, short type);
	void Copy(Image* Input, short type);

	// Swap bytes if necessary
	void ByteSwap(int *i);
	void ByteSwap(short *s);
	void ByteSwap(float *f);
	void ByteSwap(double *d);
};

#endif

// -------------------------------------------------------------------------
//   EOF
// -------------------------------------------------------------------------

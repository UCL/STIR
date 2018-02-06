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
 \brief Class for reading GIPL data
 \author Charalampos Tsoumpas [extension of original by Buerger, see below]
 
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

#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace std;

#include <cstdlib>

#include <math.h>
#include "stir/IO/GIPL_ImageFormat.h"

// -------------------------------------------------------------------------
//   Constructor
// -------------------------------------------------------------------------

Image::Image()
{
	vData = NULL;
	vData_f = NULL;

	// Initially no downscaling
	vDownsample[0] = 1;
	vDownsample[1] = 1;
	vDownsample[2] = 1;

	vCenter[0] = 0;
	vCenter[1] = 0;
	vCenter[2] = 0;
	vCenter[3] = 0;
}
Image::Image(const int num_voxels, const short data_type_case)
{
  this->m_image_type=data_type_case;
  this->MaxLength=num_voxels;
  if(this->m_image_type == 64)  
    {
      vData_f = new float[this->MaxLength];
      for (int i = 0; i < this->MaxLength; i++)
	this->vData_f[i] = 0.F;
      vData = NULL;
    }
  else if(this->m_image_type == 15)  
    {
      vData = new short[this->MaxLength];
      for (int i = 0; i < this->MaxLength; i++)
	this->vData[i] = 0;
      vData_f = NULL;
    }
  else
    {
	vData = NULL;
	vData_f = NULL;
    }
      vDownsample[0] = 1;
      vDownsample[1] = 1;
      vDownsample[2] = 1;
      
	vCenter[0] = 0;
	vCenter[1] = 0;
	vCenter[2] = 0;
	vCenter[3] = 0;
}

// -------------------------------------------------------------------------
//   Destructor
// -------------------------------------------------------------------------

Image::~Image()
{
	if (vData)
		delete[] vData;
	if (vData_f)
		delete[] vData_f;	
}

// -------------------------------------------------------------------------
//   Public functions
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
/**
*	\brief	Read data from GIPL filename.
*
*	\param	filename		Input filename
*/
// -------------------------------------------------------------------------
void Image::GiplRead(char* filename)
{
	printf("Read %s\n",filename);

	// Open input binary file
    std::fstream myFile(filename, std::ios_base::in | std::ios_base::binary);

	// Initialize image dimensions by zero
	m_dim[0] = 0;
	m_dim[1] = 0;
	m_dim[2] = 0;

	// open file for reading
	/*char *buf;
   try {
      buf = new char[512];
      if( buf == 0 )
         throw "Memory allocation failure!";
   }
   catch( char * str ) {
      cout << "Exception raised: " << str << '\n';
   }*/
	
	// Read gipl header
	ReadGiplHeader(&myFile);
	
	// Require big endian format
	ByteSwapHeader();

	// Length of the data to be read in
	MaxLength = m_dim[0]*m_dim[1]*m_dim[2];

	// File cannot be read, wrong path or filename
	if (MaxLength == 0)
	{
		printf("Error: File %s cannot be found\n",filename);
		exit(1);
	}

	// Update image dimension
	ImageDimension = 2;
	if(m_dim[2]>1) ImageDimension = 3;

	// Update parameters dimension
	ParametersDimension = ImageDimension*ImageDimension+ImageDimension;

	// Set offset vector
	ImageOffset[0] = m_dim[0]; // XLen
	ImageOffset[1] = m_dim[0]*m_dim[1]; // XLen*YLen

	// Set center of rotation of the image
	for (int i = 0; i < ImageDimension; i++)
		vCenter[i] = 0;//m_dim[i]/2.0*m_pixdim[i] - m_origin[i];

	// Delete possible old data vectors
	if (vData)
		delete[] vData;
	if (vData_f)
		delete[] vData_f;

	

	// Check data type
	switch(m_image_type)
	{
		case 15:	// shorts (default)
				
			// Initialize image vector
			vData = new short[MaxLength];

			// Read image data
			myFile.read((char*)vData, sizeof(short)*MaxLength);

			// Swap hi lo bytes
			for( int i = 0; i < MaxLength; i++)
				ByteSwap(&vData[i]);
			
			// Get min and max gray value
			GetMinMaxValue();

			break;

		case  64:	// floats

			// Initialize image vector
			vData_f = new float[MaxLength];

			// Read image data
			myFile.read((char*)vData_f, sizeof(float)*MaxLength);

			// Swap hi lo bytes
			for( int i = 0; i < MaxLength; i++)
				ByteSwap(&vData_f[i]);
			
			break;

		default:
			break;
	}

	//int SourceIndex = ((int)(5 + 10*ImageOffset[0] + 20*ImageOffset[1]));
	//float test = vData_f[SourceIndex];

	// Initially no downscaling
	vDownsample[0] = 1;
	vDownsample[1] = 1;
	vDownsample[2] = 1;

	// Close file
	myFile.close();
}

// -------------------------------------------------------------------------
/**
*	\brief	Get minimum and maximum gray values in image.
*/
// -------------------------------------------------------------------------
void Image::GetMinMaxValue()
{
	iMax = -16959;
	iMin = +16959;
	
	// Check data type
	if(m_image_type == 15)
	{
		for( int i = 0; i < MaxLength; i++)
		{
			if (vData[i] > iMax)
				iMax = vData[i];
			if (vData[i] < iMin)
				iMin = vData[i];
		}
	}
	if(m_image_type == 64)
	{
		for( int i = 0; i < MaxLength; i++)
		{
			if (vData_f[i] > iMax)
				iMax = vData_f[i];
			if (vData_f[i] < iMin)
				iMin = vData_f[i];
		}
	}
}

// -------------------------------------------------------------------------
/**
*	\brief	Read GIPL header.
*
*	\param	myfile		Input file
*/
// -------------------------------------------------------------------------
void Image::ReadGiplHeader(std::fstream* myFile)
{
	int i;
	for(i=0;i<4;i++) 
		myFile->read(reinterpret_cast < char * > (&m_dim[i]), sizeof(m_dim[i]));
	myFile->read(reinterpret_cast < char * > (&m_image_type), sizeof(m_image_type));
	for(i=0;i<4;i++) 
		myFile->read(reinterpret_cast < char * > (&m_pixdim[i]), sizeof(m_pixdim[i]));
	for(i=0;i<80;i++) 
		myFile->read(reinterpret_cast < char * > (&m_patDesc[i]), sizeof(m_patDesc[i]));
	for(i=0;i<12;i++) 
		myFile->read(reinterpret_cast < char * > (&m_matrixElements[i]), sizeof(m_matrixElements[i]));
	myFile->read(reinterpret_cast < char * > (&m_identifier), sizeof(m_identifier));
	for(i=0;i<28;i++) 
		myFile->read(reinterpret_cast < char * > (&m_spare[i]), sizeof(m_spare[i]));
	myFile->read(reinterpret_cast < char * > (&m_orientationFlag), sizeof(m_orientationFlag));
	myFile->read(reinterpret_cast < char * > (&m_flag1), sizeof(m_flag1));
	myFile->read(reinterpret_cast < char * > (&m_min), sizeof(m_min));
	myFile->read(reinterpret_cast < char * > (&m_max), sizeof(m_max));
	for(i=0;i<4;i++) 
		myFile->read(reinterpret_cast < char * > (&m_origin[i]), sizeof(m_origin[i]));
	myFile->read(reinterpret_cast < char * > (&m_pixval_offset), sizeof(m_pixval_offset));
	myFile->read(reinterpret_cast < char * > (&m_pixval_cal), sizeof(m_pixval_cal));
	myFile->read(reinterpret_cast < char * > (&m_user_def1), sizeof(m_user_def1));
	myFile->read(reinterpret_cast < char * > (&m_user_def2), sizeof(m_user_def2));
	myFile->read(reinterpret_cast < char * > (&m_magic_number), sizeof(m_magic_number));
	
}

// -------------------------------------------------------------------------
/**
*	\brief	Write image data to GIPL output file.
*
*	\param	filename		Output filename
*/
// -------------------------------------------------------------------------
void Image::GiplWrite(const char* filename)
{
	// Open file for writing
	std::fstream myFile(filename, std::ios_base::out | std::ios_base::binary);
	
	// Require big endian format
	ByteSwapHeader();

	// Write gipl header
	WriteGiplHeader(&myFile);

	// Go back to correct format
	ByteSwapHeader();

	// Check data type
	switch(m_image_type)
	{
		case 15:	// shorts (default)
				
			// Swap bytes of data vector before writing to file
			for( int i = 0; i < MaxLength; i++)
				ByteSwap(&vData[i]);

			// Read image data
			myFile.write((char*)vData, sizeof(short)*MaxLength);

			// Swap hi lo bytes
			for( int i = 0; i < MaxLength; i++)
				ByteSwap(&vData[i]);
			break;

			// Swap bytes back
			for(int i = 0; i < MaxLength; i++)
				ByteSwap(&vData[i]);

		case  64:	// floats

			// Swap bytes of data vector before writing to file
			for( int i = 0; i < MaxLength; i++)
				ByteSwap(&vData_f[i]);

			// Read image data
			myFile.write((char*)vData_f, sizeof(float)*MaxLength);

			// Swap hi lo bytes
			for( int i = 0; i < MaxLength; i++)
				ByteSwap(&vData_f[i]);
			break;

			// Swap bytes back
			for(int i = 0; i < MaxLength; i++)
				ByteSwap(&vData_f[i]);

		default:
			break;
	}
	 
	// Close file
	myFile.close();
}

// -------------------------------------------------------------------------
/**
*	\brief	Write GIPL header to output file.
*
*	\param	myfile		Output file
*/
// -------------------------------------------------------------------------
void Image::WriteGiplHeader(fstream* myFile)
{
	int i;
	for(i=0;i<4;i++) 
		myFile->write(reinterpret_cast < char * > (&m_dim[i]), sizeof(m_dim[i]));
	myFile->write(reinterpret_cast < char * > (&m_image_type), sizeof(m_image_type));
	for(i=0;i<4;i++) 
		myFile->write(reinterpret_cast < char * > (&m_pixdim[i]), sizeof(m_pixdim[i]));
	for(i=0;i<80;i++) 
		myFile->write(reinterpret_cast < char * > (&m_patDesc[i]), sizeof(m_patDesc[i]));
	for(i=0;i<12;i++) 
		myFile->write(reinterpret_cast < char * > (&m_matrixElements[i]), sizeof(m_matrixElements[i]));
	myFile->write(reinterpret_cast < char * > (&m_identifier), sizeof(m_identifier));
	for(i=0;i<28;i++) 
		myFile->write(reinterpret_cast < char * > (&m_spare[i]), sizeof(m_spare[i]));
	myFile->write(reinterpret_cast < char * > (&m_orientationFlag), sizeof(m_orientationFlag));
	myFile->write(reinterpret_cast < char * > (&m_flag1), sizeof(m_flag1));
	myFile->write(reinterpret_cast < char * > (&m_min), sizeof(m_min));
	myFile->write(reinterpret_cast < char * > (&m_max), sizeof(m_max));
	for(i=0;i<4;i++) 
		myFile->write(reinterpret_cast < char * > (&m_origin[i]), sizeof(m_origin[i]));
	myFile->write(reinterpret_cast < char * > (&m_pixval_offset), sizeof(m_pixval_offset));
	myFile->write(reinterpret_cast < char * > (&m_pixval_cal), sizeof(m_pixval_cal));
	myFile->write(reinterpret_cast < char * > (&m_user_def1), sizeof(m_user_def1));
	myFile->write(reinterpret_cast < char * > (&m_user_def2), sizeof(m_user_def2));
	myFile->write(reinterpret_cast < char * > (&m_magic_number), sizeof(m_magic_number));
}

// -------------------------------------------------------------------------
/**
*	\brief	Swap bytes (little/big endian conversion).
*/
// -------------------------------------------------------------------------
void Image::ByteSwapHeader() // for PC little endian
{ 
	int i;
	for(i=0;i<4;i++) ByteSwap(&(m_dim[i]));

	ByteSwap(&(m_image_type));

	for(i=0;i<4;i++) ByteSwap(&(m_pixdim[i]));	

	for(i=0;i<12;i++) ByteSwap(&(m_matrixElements[i]));	

	ByteSwap(&(m_min));
	ByteSwap(&(m_max));

	for(i=0;i<4;i++) ByteSwap(&(m_origin[i]));	
  
	ByteSwap(&(m_pixval_offset));
	ByteSwap(&(m_pixval_cal));
	ByteSwap(&(m_user_def1));
	ByteSwap(&(m_user_def2));
	ByteSwap((int*)&(m_magic_number));

	return;
}

// -------------------------------------------------------------------------
/**
*	\brief	Initialize zero image
*
*	\param	Input		Input image
*/
// -------------------------------------------------------------------------
void Image::Zeros(Image* Input, short iType)
{
	// Initialize image with dimensions
	this->Initialize(Input, iType);

	if (iType == _SHORT)
	{
		// Initialize with zeros
		for (int i = 0; i < this->MaxLength; i++)
			this->vData[i] = 0;

		// Update min max values
		//this->GetMinMaxValue();
	}

	if (iType == _FLOAT)
	{	
		// Initialize with zeros
		for (int i = 0; i < this->MaxLength; i++)
			this->vData_f[i] = 0;
	}
	
	iMin = 0;
	iMax = 0;
}

// -------------------------------------------------------------------------
/**
*	\brief	Initialize zero image
*
*	\param	Input		Input image
*/
// -------------------------------------------------------------------------
void Image::Zeros()
{
	// Set image to 0
	if (this->m_image_type == 15)
	{
		// Initialize with zeros
		for (int i = 0; i < this->MaxLength; i++)
			this->vData[i] = 0;

		// Update min max values
		//this->GetMinMaxValue();
	}

	if (this->m_image_type == 64)
	{	
		// Initialize with zeros
		for (int i = 0; i < this->MaxLength; i++)
			this->vData_f[i] = 0;
	}
	
	iMin = 0;
	iMax = 0;
}

// -------------------------------------------------------------------------
/**
*	\brief	Initialize one image
*
*	\param	Input		Input image
*/
// -------------------------------------------------------------------------
void Image::Ones(Image* Input, short iType)
{
	// Initialize image with dimensions
	this->Initialize(Input, iType);

	if (iType == _SHORT)
	{
		// Initialize with zeros
		for (int i = 0; i < this->MaxLength; i++)
			this->vData[i] = 1;
	}

	if (iType == _FLOAT)
	{	
		// Initialize with zeros
		for (int i = 0; i < this->MaxLength; i++)
			this->vData_f[i] = 1;
	}
	
	iMax = 1;
	iMin = 1;
}

// -------------------------------------------------------------------------
/**
*	\brief	Initialize image data by with the dimension of the input image
*
*	\param	Input		Input image
*/
// -------------------------------------------------------------------------
void Image::Initialize(Image* Input, short iType)
{
	// Set new data properties
	for (int i = 0; i < 4; i++)
	{
		this->m_dim[i] = Input->m_dim[i];
		this->m_pixdim[i] = Input->m_pixdim[i];
		this->vCenter[i] = Input->vCenter[i];
		this->m_origin[i] = Input->m_origin[i];
	}

	this->m_orientationFlag = Input->m_orientationFlag;
	this->m_flag1 = Input->m_flag1;
	this->m_min = Input->m_min;
	this->m_max = Input->m_max;	
	this->m_pixval_offset = Input->m_pixval_offset;
	this->m_pixval_cal = Input->m_pixval_cal;
	this->m_user_def1 = Input->m_user_def1;
	this->m_user_def2 = Input->m_user_def2;
	this->m_magic_number = Input->m_magic_number;
	for (int i = 0; i < 2; i++)
		this->ImageOffset[i] = Input->ImageOffset[i]; // XLen and XLen*YLen
	this->MaxLength = Input->MaxLength;

	// Initialize data vactor
	if (iType == _SHORT)
	{
		// Delete old vector and initialize data vector
		if (this->vData)
			delete[] this->vData;

		this->m_image_type = 15;
		this->vData = new short[Input->MaxLength];
	}

	if (iType == _FLOAT)
	{
		// Delete old vector and initialize data vector
		if (this->vData_f)
			delete[] this->vData_f;

		this->m_image_type = 64;
		this->vData_f = new float[Input->MaxLength];
	}
}

// -------------------------------------------------------------------------
/**
*	\brief	Initialize image data by with the dimension of the input image
*
*	\param	Input		Input image
*/
// -------------------------------------------------------------------------
void Image::Initialize(Image* Input)
{
	// Set new data properties
	for (int i = 0; i < 4; i++)
	{
		this->m_dim[i] = Input->m_dim[i];
		this->m_pixdim[i] = Input->m_pixdim[i];
		this->vCenter[i] = Input->vCenter[i];
		this->m_origin[i] = Input->m_origin[i];
	}

	this->m_orientationFlag = Input->m_orientationFlag;
	this->m_flag1 = Input->m_flag1;
	this->m_min = Input->m_min;
	this->m_max = Input->m_max;	
	this->m_pixval_offset = Input->m_pixval_offset;
	this->m_pixval_cal = Input->m_pixval_cal;
	this->m_user_def1 = Input->m_user_def1;
	this->m_user_def2 = Input->m_user_def2;
	this->m_magic_number = Input->m_magic_number;
	for (int i = 0; i < 2; i++)
		this->ImageOffset[i] = Input->ImageOffset[i]; // XLen and XLen*YLen
	this->MaxLength = Input->MaxLength;

	// Initialize data vactor
	if (this->vData)
		delete[] this->vData;

	if (this->vData_f)
		delete[] this->vData_f;
}

// -------------------------------------------------------------------------
/**
*	\brief	Copy image
*
*	\param	Input		Input image
*/
// -------------------------------------------------------------------------
void Image::Copy(Image* Input, short iType)
{
	// Initialize image with dimensions
	this->Initialize(Input, iType);

	if (iType == _SHORT)
	{
		// Initialize with zeros
		for (int i = 0; i < Input->MaxLength; i++)
			this->vData[i] = Input->vData[i];

		// Update min max values
		this->GetMinMaxValue();
	}

	if (iType == _FLOAT)
	{	
		// Initialize with zeros
		for (int i = 0; i < Input->MaxLength; i++)
			this->vData_f[i] = Input->vData_f[i];
	}
}

// -------------------------------------------------------------------------
//   Byte swap functions
// -------------------------------------------------------------------------

void Image::ByteSwap(int *i) // for PC little endian
{ 
  typedef struct {
    unsigned char byte1;
    unsigned char byte2;
    unsigned char byte3;
    unsigned char byte4;
  } Bytes;

  typedef union {
    int integer;
    Bytes bytes;
  } intUnion;

  intUnion intU, intUS;
  intU.integer = *i;

  intUS.bytes.byte1 = intU.bytes.byte4;
  intUS.bytes.byte2 = intU.bytes.byte3;
  intUS.bytes.byte3 = intU.bytes.byte2;
  intUS.bytes.byte4 = intU.bytes.byte1;
  
  *i = intUS.integer;
  return;
}

void Image::ByteSwap(short *s) // for PC little endian
{ 
  typedef struct {
    unsigned char byte1;
    unsigned char byte2;
  } Bytes;

  typedef union {
    short shortint;
    Bytes bytes;
  } shortUnion;

  shortUnion shortU, shortUS;
  shortU.shortint = *s;

  shortUS.bytes.byte1 = shortU.bytes.byte2;
  shortUS.bytes.byte2 = shortU.bytes.byte1;

  *s = shortUS.shortint;
  return;
}

void Image::ByteSwap(float *f) // for PC little endian
{ 
  typedef struct {
    unsigned char byte1;
    unsigned char byte2;
    unsigned char byte3;
    unsigned char byte4;
  } Bytes;

  typedef union {
    float floatnum;
    Bytes bytes;
  } floatUnion;

  floatUnion floatU, floatUS;
  floatU.floatnum = *f;

  floatUS.bytes.byte1 = floatU.bytes.byte4;
  floatUS.bytes.byte2 = floatU.bytes.byte3;
  floatUS.bytes.byte3 = floatU.bytes.byte2;
  floatUS.bytes.byte4 = floatU.bytes.byte1;

  *f = floatUS.floatnum;
  return;
}

void Image::ByteSwap(double *d) // for PC little endian
{ 
  typedef struct {
    unsigned char byte1;
    unsigned char byte2;
    unsigned char byte3;
    unsigned char byte4;
    unsigned char byte5;
    unsigned char byte6;
    unsigned char byte7;
    unsigned char byte8;
  } Bytes;

  typedef union {
    double doublenum;
    Bytes bytes;
  } doubleUnion;

  doubleUnion doubleU, doubleUS;
  doubleU.doublenum = *d;

  doubleUS.bytes.byte1 = doubleU.bytes.byte8;
  doubleUS.bytes.byte2 = doubleU.bytes.byte7;
  doubleUS.bytes.byte3 = doubleU.bytes.byte6;
  doubleUS.bytes.byte4 = doubleU.bytes.byte5;
  doubleUS.bytes.byte5 = doubleU.bytes.byte4;
  doubleUS.bytes.byte6 = doubleU.bytes.byte3;
  doubleUS.bytes.byte7 = doubleU.bytes.byte2;
  doubleUS.bytes.byte8 = doubleU.bytes.byte1;

  *d = doubleUS.doublenum;
  return;
}

// -------------------------------------------------------------------------
//   EOF
// -------------------------------------------------------------------------

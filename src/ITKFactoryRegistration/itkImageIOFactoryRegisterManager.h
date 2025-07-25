/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef itkImageIOFactoryRegisterManager_h
#define itkImageIOFactoryRegisterManager_h

namespace itk {

class ImageIOFactoryRegisterManager
{
  public:
  explicit ImageIOFactoryRegisterManager(void (* const list[])(void))
    {
    for(;*list != nullptr; ++list)
      {
      (*list)();
      }
    }
};


//
//  The following code is intended to be expanded at the end of the
//  itkImageFileReader.h and itkImageFileWriter.h files.
//
void ITK_ABI_IMPORT BMPImageIOFactoryRegister__Private();void ITK_ABI_IMPORT BioRadImageIOFactoryRegister__Private();void ITK_ABI_IMPORT Bruker2dseqImageIOFactoryRegister__Private();void ITK_ABI_IMPORT GDCMImageIOFactoryRegister__Private();void ITK_ABI_IMPORT GE4ImageIOFactoryRegister__Private();void ITK_ABI_IMPORT GE5ImageIOFactoryRegister__Private();void ITK_ABI_IMPORT GiplImageIOFactoryRegister__Private();void ITK_ABI_IMPORT HDF5ImageIOFactoryRegister__Private();void ITK_ABI_IMPORT JPEGImageIOFactoryRegister__Private();void ITK_ABI_IMPORT JPEG2000ImageIOFactoryRegister__Private();void ITK_ABI_IMPORT LSMImageIOFactoryRegister__Private();void ITK_ABI_IMPORT MGHImageIOFactoryRegister__Private();void ITK_ABI_IMPORT MINCImageIOFactoryRegister__Private();void ITK_ABI_IMPORT MRCImageIOFactoryRegister__Private();void ITK_ABI_IMPORT MetaImageIOFactoryRegister__Private();void ITK_ABI_IMPORT NiftiImageIOFactoryRegister__Private();void ITK_ABI_IMPORT NrrdImageIOFactoryRegister__Private();void ITK_ABI_IMPORT PNGImageIOFactoryRegister__Private();void ITK_ABI_IMPORT StimulateImageIOFactoryRegister__Private();void ITK_ABI_IMPORT TIFFImageIOFactoryRegister__Private();void ITK_ABI_IMPORT VTKImageIOFactoryRegister__Private();

//
// The code below registers available IO helpers using static initialization in
// application translation units. Note that this code will be expanded in the
// ITK-based applications and not in ITK itself.
//

void (* const ImageIOFactoryRegisterRegisterList[])(void) = {
  BMPImageIOFactoryRegister__Private,BioRadImageIOFactoryRegister__Private,Bruker2dseqImageIOFactoryRegister__Private,GDCMImageIOFactoryRegister__Private,GE4ImageIOFactoryRegister__Private,GE5ImageIOFactoryRegister__Private,GiplImageIOFactoryRegister__Private,HDF5ImageIOFactoryRegister__Private,JPEGImageIOFactoryRegister__Private,JPEG2000ImageIOFactoryRegister__Private,LSMImageIOFactoryRegister__Private,MGHImageIOFactoryRegister__Private,MINCImageIOFactoryRegister__Private,MRCImageIOFactoryRegister__Private,MetaImageIOFactoryRegister__Private,NiftiImageIOFactoryRegister__Private,NrrdImageIOFactoryRegister__Private,PNGImageIOFactoryRegister__Private,StimulateImageIOFactoryRegister__Private,TIFFImageIOFactoryRegister__Private,VTKImageIOFactoryRegister__Private,
  nullptr};
const ImageIOFactoryRegisterManager ImageIOFactoryRegisterManagerInstance(ImageIOFactoryRegisterRegisterList);

}

#endif

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

#ifndef itkTransformIOFactoryRegisterManager_h
#define itkTransformIOFactoryRegisterManager_h

namespace itk {

class TransformIOFactoryRegisterManager
{
  public:
  explicit TransformIOFactoryRegisterManager(void (* const list[])(void))
    {
    for(;*list != nullptr; ++list)
      {
      (*list)();
      }
    }
};


//
//  The following code is intended to be expanded at the end of the
//  itkTransformFileReader.h and itkTransformFileWriter.h files.
//
void ITK_ABI_IMPORT HDF5TransformIOFactoryRegister__Private();void ITK_ABI_IMPORT MINCTransformIOFactoryRegister__Private();void ITK_ABI_IMPORT MatlabTransformIOFactoryRegister__Private();void ITK_ABI_IMPORT TxtTransformIOFactoryRegister__Private();

//
// The code below registers available IO helpers using static initialization in
// application translation units. Note that this code will be expanded in the
// ITK-based applications and not in ITK itself.
//

void (* const TransformIOFactoryRegisterRegisterList[])(void) = {
  HDF5TransformIOFactoryRegister__Private,MINCTransformIOFactoryRegister__Private,MatlabTransformIOFactoryRegister__Private,TxtTransformIOFactoryRegister__Private,
  nullptr};
const TransformIOFactoryRegisterManager TransformIOFactoryRegisterManagerInstance(TransformIOFactoryRegisterRegisterList);

}

#endif

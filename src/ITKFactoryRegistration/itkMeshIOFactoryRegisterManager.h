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

#ifndef itkMeshIOFactoryRegisterManager_h
#define itkMeshIOFactoryRegisterManager_h

namespace itk {

class MeshIOFactoryRegisterManager
{
  public:
  explicit MeshIOFactoryRegisterManager(void (* const list[])(void))
    {
    for(;*list != nullptr; ++list)
      {
      (*list)();
      }
    }
};


//
//  The following code is intended to be expanded at the end of the
//  itkMeshFileReader.h and itkMeshFileWriter.h files.
//
void ITK_ABI_IMPORT BYUMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT BYUMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT FreeSurferAsciiMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT FreeSurferAsciiMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT FreeSurferBinaryMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT FreeSurferBinaryMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT GiftiMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT GiftiMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT OBJMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT OBJMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT OFFMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT OFFMeshIOFactoryRegister__Private();void ITK_ABI_IMPORT VTKPolyDataMeshIOFactoryRegister__Private();

//
// The code below registers available IO helpers using static initialization in
// application translation units. Note that this code will be expanded in the
// ITK-based applications and not in ITK itself.
//

void (* const MeshIOFactoryRegisterRegisterList[])(void) = {
  BYUMeshIOFactoryRegister__Private,BYUMeshIOFactoryRegister__Private,FreeSurferAsciiMeshIOFactoryRegister__Private,FreeSurferAsciiMeshIOFactoryRegister__Private,FreeSurferBinaryMeshIOFactoryRegister__Private,FreeSurferBinaryMeshIOFactoryRegister__Private,GiftiMeshIOFactoryRegister__Private,GiftiMeshIOFactoryRegister__Private,OBJMeshIOFactoryRegister__Private,OBJMeshIOFactoryRegister__Private,OFFMeshIOFactoryRegister__Private,OFFMeshIOFactoryRegister__Private,VTKPolyDataMeshIOFactoryRegister__Private,
  nullptr};
const MeshIOFactoryRegisterManager MeshIOFactoryRegisterManagerInstance(MeshIOFactoryRegisterRegisterList);

}

#endif

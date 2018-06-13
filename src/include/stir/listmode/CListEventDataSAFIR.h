/* CListEventDataSAFIR.h

 Coincidence Event Class for SAFIR: Header File
 Jannis Fischer
 jannis.fischer@cern.ch

	Copyright 2015 ETH Zurich, Institute of Particle Physics

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.

 */

/*!

  \file
  \ingroup listmode
  \brief Declaration of class stir::CListEventDataSAFIR supporting class

  \author Jannis Fischer
*/

#ifndef __stir_listmode_CListEventDataSAFIR_H__
#define __stir_listmode_CListEventDataSAFIR_H__

#include "stir/listmode/CListEventSAFIR.h"
#include "stir/DetectionPositionPair.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrderDefine.h"

START_NAMESPACE_STIR
//! Class for record with coincidence data
class CListEventDataSAFIR
{
public:
	//! Writes detection position pair to reference given as argument.
	inline void get_detection_position_pair(DetectionPositionPair<>& det_pos_pair);

	//! Returns 0 if event is prompt and 1 if random/delayed
	inline bool is_prompt()
		const { return !isRandom; }

	//! Can be used to set "promptness" of event.
	inline Succeeded set_prompt( const bool prompt = true ) { 
		isRandom = !prompt;
		return Succeeded::yes; 
	}


private:
	friend class CListRecordSAFIR;

#if STIRIsNativeByteOrderBigEndian
	unsigned type : 1;
	unsigned isRandom : 1;
	unsigned reserved : 6;
	unsigned layerB : 4;
	unsigned layerA : 4;
	unsigned detB : 16;
	unsigned detA : 16;
	unsigned ringB : 8;
	unsigned ringA : 8;
#else
	unsigned ringA : 8;
	unsigned ringB : 8;
	unsigned detA : 16;
	unsigned detB : 16;
	unsigned layerA : 4;
	unsigned layerB : 4;
	unsigned reserved : 6;
	unsigned isRandom : 1;
	unsigned type : 1;
#endif
}; 

END_NAMESPACE_STIR

#endif

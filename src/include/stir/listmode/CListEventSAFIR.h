/* CListEventSAFIR.h

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
  \brief Declaration of class stir::CListEventSAFIR supporting class

  \author Jannis Fischer
*/

#ifndef __stir_listmode_CListEventSAFIR_H__
#define __stir_listmode_CListEventSAFIR_H__

#include "stir/listmode/CListEvent.h"
#include "stir/listmode/DetectorCoordinateMapFromFile.h"

START_NAMESPACE_STIR

/*!
Provides interface of the record class to STIR by implementing get_LOR(). It uses a map from detector indices to coordinates to specify LORAs2Points from given detection pair indices.

The record has the following format (for little-endian byte order)
\code
	unsigned ringA : 8;
	unsigned ringB : 8;
	unsigned detA : 16;
	unsigned detB : 16;
	unsigned layerA : 4;
	unsigned layerB : 4;
	unsigned reserved : 6;
	unsigned isRandom : 1;
	unsigned type : 1;
\endcode
*/
template <class Derived>
class CListEventSAFIR : public CListEvent
{
public:
	/*! Constructor which initializes map upon construction.
	*/
	inline CListEventSAFIR( shared_ptr<DetectorCoordinateMapFromFile> map ) : map(map) {}
	//! Returns LOR corresponding to the given event.
	inline virtual LORAs2Points<float> get_LOR() const;
  //! This method checks if the template is valid for LmToProjData
  /*! Used before the actual processing of the data (see issue #61), before calling get_bin()
   *  Most scanners have listmode data that correspond to non arc-corrected data and
   *  this check avoids a crash when an unsupported template is used as input.
   */
	inline virtual bool is_valid_template(const ProjDataInfo&) const {return true;}

	//! Returns 0 if event is prompt and 1 if random/delayed
	inline bool is_prompt()
		const { return !(static_cast<const Derived*>(this)->is_prompt()); }
	//! Function to set map for detector indices to coordinates.
	inline void set_map( shared_ptr<DetectorCoordinateMapFromFile> new_map ) { map = new_map; }

private:
	friend class CListRecordSAFIR;
	/*! Default constructor will not work as it does not initialize a map to relate
	detector indices and space coordinates. Always use other constructor with a map pointer. Or use set_map( shared_ptr<DetectorCoordinateMapFromFile> new_map ) after default construction.
	*/
	inline CListEventSAFIR( ) {}
	shared_ptr<DetectorCoordinateMapFromFile> map;
};

END_NAMESPACE_STIR

#endif

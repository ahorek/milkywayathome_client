/*
 *  Copyright (c) 2010-2011 Rensselaer Polytechnic Institute
 *  Copyright (c) 2010-2011 Matthew Arsenault
 *
 *  This file is part of Milkway@Home.
 *
 *  Milkway@Home is free software: you may copy, redistribute and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation, either version 3 of the License, or (at your
 *  option) any later version.
 *
 *  This file is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _NBODY_POTENTIAL_H_
#define _NBODY_POTENTIAL_H_

#include "nbody_types.h"

#ifdef __cplusplus
extern "C" {
#endif

mwvector nbExtAcceleration(const Potential* pot, mwvector pos, real time);
mwvector LMCAcceleration(const int lmcfunction, const mwvector pos, const mwvector pos1, const real mass, const real scale, const real scale2);
//mwvector pointLmcAccel(const mwvector pos, const mwvector pos1, const real mass);
//mwvector plummerLmcAccel(const mwvector pos, const mwvector pos1, const real mass, const real scale);
//mwvector hernquistLmcAccel(const mwvector pos, const mwvector pos1, const real mass, const real scale);

/* String-name helpers for the potential components, used by both
 * nbPrintPotentialModel (CPU-side init log) and the CUDA backend's
 * unsupported-component rejection messages. */
const char* nbCUDASphericalName(spherical_t t);
const char* nbCUDADiskName(disk_t t);
const char* nbCUDAHaloName(halo_t t);

/* Print one-line summary of the WU's potential model. Called once
 * during simulation startup, regardless of CUDA / CPU / fallback path. */
void nbPrintPotentialModel(const NBodyCtx* ctx);

/* Print the program's argv as a single [nbody] argv: line so a WU is
 * exactly replayable from stderr.txt. Called once at main() entry. */
void nbPrintArgv(int argc, const char* const* argv);

/* Print the derived numeric simulation parameters (nStep, timestep,
 * eps2, theta, etc.) — useful for triaging anomalous runs at a glance
 * even without re-deriving them from argv + lua. */
void nbPrintRunParams(const NBodyCtx* ctx, int nbody);

#ifdef __cplusplus
}
#endif

#endif /* _NBODY_POTENTIAL_H_ */


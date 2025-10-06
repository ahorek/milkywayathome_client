/* Copyright 2010 Matthew Arsenault, Travis Desell, Boleslaw
Szymanski, Heidi Newberg, Carlos Varela, Malik Magdon-Ismail and
Rensselaer Polytechnic Institute.

This file is part of Milkway@Home.

Milkyway@Home is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Milkyway@Home is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Milkyway@Home.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _NBODY_CHECKPOINT_H_
#define _NBODY_CHECKPOINT_H_

#include "nbody_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_FCNTL_H
  #include <fcntl.h>
#endif

#if HAVE_WINDOWS_H
  #include <windows.h>
#endif

#if HAVE_SYS_MMAN_H
  #include <sys/mman.h>
#endif

#if HAVE_SYS_TYPES_H
  #include <sys/types.h>
#endif

#if HAVE_SYS_STAT_H
  #include <sys/stat.h>
#endif

#ifndef _WIN32

typedef struct
{
    int fd;            /* File descriptor for checkpoint file */
    char* mptr;        /* mmap'd pointer for checkpoint file */
    size_t cpFileSize; /* For checking how big the file should be for expected bodies */
} CheckpointHandle;

#define EMPTY_CHECKPOINT_HANDLE { 1, NULL, 0 }

#else

typedef struct
{
    HANDLE file;
    HANDLE mapFile;
    char* mptr;
    DWORD cpFileSize;
} CheckpointHandle;

#define EMPTY_CHECKPOINT_HANDLE { INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE, NULL, 0 }

#endif /* _WIN32 */

typedef struct
{
    char header[128];                     /* "mwnbody" */
    uint32_t majorVersion, minorVersion;  /* Version check */
    uint32_t nbody;
    uint32_t step;
    uint32_t realSize;                   /* Does the checkpoint use float or double */
    uint32_t ptrSize;
    uint32_t nOrbitTrace;
    uint32_t nShiftLMC;
    uint32_t treeIncest;
    real rsize;
    NBodyCtx ctx;
} NBodyCheckpointHeader;

void nbReadCheckpointHeader(NBodyCheckpointHeader* cp, NBodyCtx* ctx, NBodyState* st);
int verifyCheckpointHeader(const NBodyCheckpointHeader* cpHdr, const CheckpointHandle* cp, const NBodyState* st, size_t supposedCheckpointSize);
int nbThawState(NBodyCtx* ctx, NBodyState* st, CheckpointHandle* cp);
int nbResolveCheckpoint(NBodyState* st, const char* checkpointFileName);
int nbResolvedCheckpointExists(const NBodyState* st);
int nbReadCheckpoint(NBodyCtx* ctx, NBodyState* st);
int nbWriteCheckpoint(const NBodyCtx* ctx, const NBodyState* st);
int nbWriteCheckpointWithTmpFile(const NBodyCtx* ctx, const NBodyState* st, const char* tmpFile);
NBodyStatus nbWriteFinalCheckpoint(const NBodyCtx* ctx, NBodyState* st);
int nbTimeToCheckpoint(const NBodyCtx* ctx, NBodyState* st);

#ifdef __cplusplus
}
#endif

#endif /* _NBODY_CHECKPOINT_H_ */


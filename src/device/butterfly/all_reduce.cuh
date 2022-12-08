/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.cuh"
#include "collectives.h"

//segmented butterfly - lyz
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_BUTTERFLY_YZ, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads-WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    //struct ncclButterfly* butterfly = &channel->butterfly;
    const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
    //const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
    const ssize_t size = args->coll.count;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    //help keep track of the offsets and size
    int reducedPeerRanks[1024]; 
    //int peerRecvOffsets[1024];
    //int peerFirstHalfSizes[1024];
    //int peerSendOffsets[1024];
    int peerHalfSizes[1024];
    int reducedPeerCount = 0;

    /////////////// begin Segmented Butterfly steps ///////////////
    int commOffset = 0;
    int commSize = size;
    struct ncclButterfly* butterfly = &channel->butterfly;
    int myRank = comm->rank;

    ////// Scatter ////
    for (int p = 0; p < butterfly->peerCount; p++) {
        ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
          prims(tid, nthreads, &(butterfly->peerRanks[p]), &(butterfly->peerRanks[p]), thisOutput, stepSize, channel, comm);

        int peerRank = butterfly->peerRanks[p];

        //if(myRank == 0) printf("tid %d:LYZ - Scatter, comm with %d, size %d, loopSize %d, compressed %d\n",tid,peerRank, commSize, loopSize,butterfly->lastoneCompressed);
        if (p == (butterfly->peerCount - 1) && butterfly->lastoneCompressed == 1) continue; 
        //send the entire data block to the neighbor
        if (p==0 && butterfly->lastoneCompressed == 1) {
    //if(myRank == 0) printf("LYZ - BUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
            ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
            ssize_t chunkOffset = gridOffset + bid*realChunkSize;
            ssize_t offset;
            int nelem;
            offset = chunkOffset;
            nelem = min(realChunkSize, size-offset);
            if (myRank < peerRank) {
              prims.send(thisInput+offset, nelem);
            }
            else{
              prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
            }
          }
        }
        else{
          //looping
          int halfSize = commSize/2;

          //modification required - lyz
          //send first half
          //int finishedSize = 0; //help determine if the second half requires an extra round
          int loopCount = 0;
          for (ssize_t gridOffset = commOffset; gridOffset < commOffset + halfSize; gridOffset += loopSize) {
            //finishedSize += loopSize;
            loopCount++;
            ssize_t realChunkSize = min(chunkSize, DIVUP(commOffset + halfSize - gridOffset,nChannels));
            ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
            ssize_t chunkOffset_first = gridOffset + bid*realChunkSize;
            ssize_t chunkOffset_second = chunkOffset_first + halfSize;

            /////////////// begin Butterfly steps ///////////////
            int nelem;
            nelem = min(realChunkSize, commOffset + halfSize - chunkOffset_first); //nelems for the first and second half are the same
            
            //First both send
            if (myRank < peerRank) {
              prims.send(thisInput+chunkOffset_second, nelem);
            }
            else{
              prims.send(thisInput+chunkOffset_first, nelem);
            }

            //if(myRank == 0) printf("tid %d:LYZ - 1Send done - loop %d \n",tid,loopCount);
            //Then both recv
            if (myRank < peerRank) {
              prims.recvReduceCopy(thisInput+chunkOffset_first, thisOutput+chunkOffset_first, nelem);
            }
            else{
              prims.recvReduceCopy(thisInput+chunkOffset_second, thisOutput+chunkOffset_second, nelem);
            }
      //if(myRank == 0) printf("tid %d:LYZ - 1Recv done - loop %d \n",tid,loopCount);
          }


          if (myRank > peerRank) commOffset += halfSize;
          commSize = halfSize;
          peerHalfSizes[reducedPeerCount] = halfSize;
          reducedPeerRanks[reducedPeerCount] = peerRank;
          reducedPeerCount++;
        }
    }
    
    ////// Gather ////
    for (int p = reducedPeerCount -1 ; p >= 0; p--) {
        ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
          prims(tid, nthreads, &(reducedPeerRanks[p]), &(reducedPeerRanks[p]), thisOutput, stepSize, channel, comm);

        int peerRank = reducedPeerRanks[p];
        //if(myRank == 0) printf("tid %d:LYZ - Gather, comm with %d, size %d, loopSize %d\n",tid,peerRank, commSize, loopSize);
        //looping
        int halfSize = peerHalfSizes[p];
        //if(myRank == 0) printf("tid %d:LYZ - Gather, comm with %d, --halfsize %d, loopSize %d\n",tid,peerRank, halfSize, loopSize);
        if (myRank > peerRank) commOffset -= halfSize;
        int loopCount = 0;
        for (ssize_t gridOffset = commOffset; gridOffset < commOffset + halfSize; gridOffset += loopSize) {
            //finishedSize += loopSize;
            loopCount++;
            ssize_t realChunkSize = min(chunkSize, DIVUP(commOffset + halfSize - gridOffset,nChannels));
            ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
            ssize_t chunkOffset_first = gridOffset + bid*realChunkSize;
            ssize_t chunkOffset_second = chunkOffset_first + halfSize;

            /////////////// begin Butterfly steps ///////////////
            int nelem;
            nelem = min(realChunkSize, commOffset + halfSize - chunkOffset_first); //nelems for the first and second half are the same
            
            //First both send
            if (myRank < peerRank) {
              prims.send(thisInput+chunkOffset_first, nelem);
            }
            else{
              prims.send(thisInput+chunkOffset_second, nelem);
            }

      //if(myRank == 0) printf("tid %d:LYZ - Send done - loop %d \n",tid,loopCount);

            //Then both recv
            if (myRank < peerRank) {
              prims.recv(thisOutput+chunkOffset_second, nelem);
        //prims.recvReduceCopy(thisInput+chunkOffset_second, thisOutput+chunkOffset_second, nelem);
            }
            else{
              prims.recv(thisOutput+chunkOffset_first, nelem);
        //prims.recvReduceCopy(thisInput+chunkOffset_first, thisOutput+chunkOffset_first, nelem);
            }
      //if(myRank == 0) printf("tid %d:LYZ - Recv done - loop %d \n",tid,loopCount);
        }
      }

    //finally, send to the reduced ranks
    if (butterfly->lastoneCompressed == 1) {
      int peerRank = butterfly->peerRanks[0];

      ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
          prims(tid, nthreads, &(butterfly->peerRanks[0]), &(butterfly->peerRanks[0]), thisOutput, stepSize, channel, comm);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*realChunkSize;

        /////////////// begin Butterfly steps ///////////////
        ssize_t offset;
        int nelem;
        offset = chunkOffset;
        nelem = min(realChunkSize, size-offset);
        if (myRank < peerRank) {
          //printf("0Stage 1");
          prims.recv(thisOutput+offset, nelem);
        }
        else{
          //printf("1Stage 1");
          prims.send(thisInput+offset, nelem);
        }
      }
    }
  }
};

//butterfly - lyz
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_BUTTERFLY_YZ, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    //struct ncclRing* butterfly = &channel->butterfly;
    const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);
    //const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;



    /////////////// begin Butterfly steps ///////////////
    struct ncclButterfly* butterfly = &channel->butterfly;
    int myRank = comm->rank;
    for (int p = 0; p < butterfly->peerCount; p++) {
        ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &(butterfly->peerRanks[p]), &(butterfly->peerRanks[p]), stepLines, channel, comm);

        int peerRank = butterfly->peerRanks[p];
        printf("LL Communicating %d <-> %d \n", myRank, peerRank);      
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t realChunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);

            /////////////// begin AllReduce steps ///////////////
            ssize_t offset;
            int nelem;

            offset = gridOffset + bid * realChunkSize;
            nelem = min(realChunkSize, size-offset);
            
            if (p==0 && butterfly->lastoneCompressed == 1) {
              if (myRank < peerRank) {
                LLprims.send(thisInput+offset, nelem);
              }
              else{
                LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
              }
            }
            else if (p < butterfly->peerCount - 1 || (p == (butterfly->peerCount - 1) && butterfly->lastoneCompressed == 0)) {
              if (myRank < peerRank) {
                printf("I'm sending data \n");
                LLprims.send(thisInput+offset, nelem);
          printf("Sending done. start recving \n");
                LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
          printf("Recving done \n");
              }
              else {
                LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
              }
            }
            else {
              if (myRank < peerRank) {
                LLprims.recv(thisOutput+offset, nelem);
              }
              else {
                LLprims.send(thisOutput+offset, nelem);
              }
            }
        }
    }
    printf("Kernel finished 0.\n");

  }
};

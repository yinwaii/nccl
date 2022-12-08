/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.cuh"
#include "collectives.h"

//butterfly - lyz
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclCollBroadcast, NCCL_ALGO_BUTTERFLY_YZ, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct CollectiveArgs* args) {
    //printf("Launching kernel ---!!--- \n");


    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads-WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    //struct ncclButterfly* butterfly = &channel->butterfly;
    const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    const int chunkSize = stepSize * BROADCAST_CHUNKSTEPS;
    //const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
    const ssize_t size = args->coll.count;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;


    struct ncclButterfly* butterfly = &channel->butterfly;
    /** HD broadcast basing on the structure of butter fly **/
    //determine the peer ranks
    int recvPeerRank = -1;
    int sendPeerRanks[32];
    int sendPeerCount = 0;

    int adjsize = 1;
    int nRanks = comm->nRanks;
    while (adjsize <= nRanks) adjsize <<= 1;
    adjsize >>= 1;

    int myRank = comm->rank;
    int myNewRank = myRank;
    

    int rootRank = args->coll.root;
    int rootNewRank = args->coll.root;
    //cases for nRanks non-power of 2
    int extra_ranks = nRanks - adjsize;
    if (myRank < (2 * extra_ranks)) {
    if ((myRank % 2) == 0) myNewRank = -1;
    else myNewRank >>= 1;
    }
    else myNewRank -= extra_ranks;
                  /***root***/
    if (rootRank < (2 * extra_ranks)) {
    if ((rootRank % 2) == 0){  //root is in extra ranks
      if (myRank == rootRank) {
        sendPeerRanks[sendPeerCount++] = rootRank + 1;
      }
      if (myRank == rootRank + 1) recvPeerRank = rootRank;
    }
    rootNewRank >>= 1;
    }
    else rootNewRank -= extra_ranks;
    /***root***/


    int roundStepSize = adjsize/2;
    //find the recv peer
    if (myNewRank == -1) {
        if (myRank != rootRank) recvPeerRank = myRank + 1; // no send peer
        roundStepSize = 0;
    }
    else{
    //starting from the root to my rank
    if (myNewRank != rootNewRank) {
      int currentRank = rootNewRank;
      while (currentRank != myNewRank) {
        int myRankLoc = myNewRank/roundStepSize;
        int currentRankLoc = currentRank/roundStepSize;

        if ( myRankLoc > currentRankLoc){
                if ((currentRank + roundStepSize) == myNewRank) recvPeerRank = currentRank;
                currentRank += roundStepSize;
        }
        if ( myRankLoc < currentRankLoc){
                if ((currentRank - roundStepSize) == myNewRank) recvPeerRank = currentRank;
                currentRank -= roundStepSize;
        }
        roundStepSize >>= 1;
      }
      recvPeerRank = (recvPeerRank < extra_ranks)? (recvPeerRank * 2 + 1):(recvPeerRank + extra_ranks);
    }
    }

    //remaining are send peers
    while (roundStepSize > 0) {
      int sendPeerRank;
      int loc = (myNewRank)%(roundStepSize*2);
      if ( (loc+1) > roundStepSize) sendPeerRank= myNewRank - roundStepSize;
      else sendPeerRank = myNewRank + roundStepSize;
      sendPeerRank = (sendPeerRank < extra_ranks)? (sendPeerRank * 2 + 1):(sendPeerRank + extra_ranks);
      sendPeerRanks[sendPeerCount++] = sendPeerRank;
      roundStepSize >>=1;
    }
    if (myNewRank != -1 && myNewRank < extra_ranks && (myRank - 1) != rootRank) sendPeerRanks[sendPeerCount++] = (myRank - 1);
    /** Peer calculation done **/


    /**Start comm**/
    //First Recv 
    if (recvPeerRank != -1) {
        ncclPrimitives<UNROLL, BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS, T, 1, 1, 0, FUNC>
          prims(tid, nthreads, &(recvPeerRank), &(recvPeerRank), thisOutput, stepSize, channel, comm);

        //printf("Recving1 from %d \n",recvPeerRank);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
            ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
            ssize_t chunkOffset = gridOffset + bid*realChunkSize;

            ssize_t offset;
            int nelem;

            offset = chunkOffset;
            nelem = min(realChunkSize, size-offset);
            //prims.recv(thisOutput+offset, nelem);
            prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        }
    }
    //Then send
    for (int p = 0; p < sendPeerCount; p++) {
        ncclPrimitives<UNROLL, BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS, T, 1, 1, 0, FUNC>
          prims(tid, nthreads, &(sendPeerRanks[p]), &(sendPeerRanks[p]), thisOutput, stepSize, channel, comm);

        //printf("Sending1 to %d \n",sendPeerRanks[p]);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
            ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
            ssize_t chunkOffset = gridOffset + bid*realChunkSize;

            ssize_t offset;
            int nelem;

            offset = chunkOffset;
            nelem = min(realChunkSize, size-offset);
            if (myRank == rootRank) {
              if (thisInput == thisOutput) {
                prims.send(thisInput+offset, nelem);
              } else {
                prims.copySend(thisInput+offset, thisOutput+offset, nelem);
              }
            }
            else {
              prims.send(thisOutput+offset, nelem);
              //prims.send(thisInput+offset, nelem);
            }
        }
    }
  }
};

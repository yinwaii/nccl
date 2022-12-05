/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceRingKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads-WARP_SIZE;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const int nranks = comm->nRanks;
  const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
  const ssize_t size = args->coll.count;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
    prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm);

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
    ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    prims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.directRecvCopySend(thisOutput+offset, offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    // Final wait/copy.
    prims.directRecv(thisOutput+offset, offset, nelem);
  }
}


//segmented butterfly - lyz
template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceButterflyKernel(struct CollectiveArgs* args) {
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

/*
//segmented butterfly - lyz
template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceButterflyKernel(struct CollectiveArgs* args) {
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
  int peerFirstHalfSizes[1024];
  //int peerSendOffsets[1024];
  int peerSecondHalfSizes[1024];
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

      if (p == (butterfly->peerCount - 1) && butterfly->lastoneCompressed == 1) continue; 
      //send the entire data block to the neighbor
      if (p==0 && butterfly->lastoneCompressed == 1) {
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
          ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
          ssize_t chunkOffset = gridOffset + bid*realChunkSize;
          ssize_t offset;
          int nelem;
          offset = chunkOffset;
          nelem = min(realChunkSize, size-offset);
          if (myRank < peerRank) {
            //printf("0Stage 1");
            prims.send(thisInput+offset, nelem);
          }
          else{
            //printf("1Stage 1");
            prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
          }
        }
      }
      else{
        //looping
        int firstHalfSize = commSize/2;
        int secondHalfSize = commSize/2 + (commSize%2);

        //modification required - lyz
        //send first half
        for (ssize_t gridOffset = commOffset; gridOffset < commOffset + firstHalfSize; gridOffset += loopSize) {
          ssize_t realChunkSize = min(chunkSize, DIVUP(commOffset + firstHalfSize-gridOffset,nChannels));
          ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
          ssize_t chunkOffset = gridOffset + bid*realChunkSize;

          /////////////// begin Butterfly steps ///////////////
          ssize_t offset;
          int nelem;
          offset = chunkOffset;
          nelem = min(realChunkSize, commOffset + firstHalfSize - offset);
          if (myRank < peerRank) {
            //printf("0Stage 1");
            prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
          }
          else{
            //printf("1Stage 1");
            prims.send(thisInput+offset, nelem);
          }
        }

        //then the second half
        for (ssize_t gridOffset = commOffset + firstHalfSize; gridOffset < commOffset + firstHalfSize + secondHalfSize; gridOffset += loopSize) {
          ssize_t realChunkSize = min(chunkSize, DIVUP(commOffset + firstHalfSize + secondHalfSize - gridOffset , nChannels));
          ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
          ssize_t chunkOffset = gridOffset + bid*realChunkSize;

          /////////////// begin Butterfly steps ///////////////
          ssize_t offset;
          int nelem;
          offset = chunkOffset;
          nelem = min(realChunkSize, commOffset + firstHalfSize + secondHalfSize -offset);
          if (myRank < peerRank) {
            //printf("0Stage 1");
            prims.send(thisInput+offset, nelem);
          }
          else{
            //printf("1Stage 1");
            prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
          }
        }

        if (myRank > peerRank) {
          commSize = secondHalfSize;
          commOffset += firstHalfSize;
        }
        else commSize = firstHalfSize;
        peerSecondHalfSizes[reducedPeerCount] = secondHalfSize;
        peerFirstHalfSizes[reducedPeerCount] = firstHalfSize;
        reducedPeerRanks[reducedPeerCount] = peerRank;
        reducedPeerCount++;
      }
  }

  ////// Gather ////
  for (int p = reducedPeerCount -1 ; p >= 0; p--) {
      ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &(reducedPeerRanks[p]), &(reducedPeerRanks[p]), thisOutput, stepSize, channel, comm);

      int peerRank = reducedPeerRanks[p];
      //looping
      int firstHalfSize = peerFirstHalfSizes[p];
      int secondHalfSize = peerSecondHalfSizes[p];
      if (myRank > peerRank) commOffset -= firstHalfSize;

      //send first half
      for (ssize_t gridOffset = commOffset; gridOffset < commOffset + firstHalfSize; gridOffset += loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(commOffset + firstHalfSize-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*realChunkSize;

        /////////////// begin Butterfly steps ///////////////
        ssize_t offset;
        int nelem;
        offset = chunkOffset;
        nelem = min(realChunkSize, commOffset + firstHalfSize - offset);        
        if (myRank < peerRank) {
          //printf("0Stage 1");
          prims.send(thisInput+offset, nelem);
        }
        else{
          //printf("1Stage 1");
          prims.recv(thisOutput+offset, nelem);
        }
      }

      //then the second half
      for (ssize_t gridOffset = commOffset + firstHalfSize; gridOffset < commOffset + firstHalfSize + secondHalfSize; gridOffset += loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(commOffset + firstHalfSize + secondHalfSize - gridOffset , nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*realChunkSize;

        /////////////// begin Butterfly steps ///////////////
        ssize_t offset;
        int nelem;
        offset = chunkOffset;
        nelem = min(realChunkSize, commOffset + firstHalfSize + secondHalfSize -offset);
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
*/

/*
//butterfly - lyz
template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceButterflyKernel(struct CollectiveArgs* args) {
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



  /////////////// begin Butterfly steps ///////////////
  struct ncclButterfly* butterfly = &channel->butterfly;
  int myRank = comm->rank;
  for (int p = 0; p < butterfly->peerCount; p++) {
      ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &(butterfly->peerRanks[p]), &(butterfly->peerRanks[p]), thisOutput, stepSize, channel, comm);

      int peerRank = butterfly->peerRanks[p];
      //printf("Communicating %d <-> %d \n", myRank, peerRank);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
          ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
          ssize_t chunkOffset = gridOffset + bid*realChunkSize;

          /////////////// begin Butterfly steps ///////////////
          ssize_t offset;
          int nelem;
          //int chunk;

          offset = chunkOffset;
          nelem = min(realChunkSize, size-offset);

          
          if (p==0 && butterfly->lastoneCompressed == 1) {
            if (myRank < peerRank) {
	      //printf("0Stage 1");
              prims.send(thisInput+offset, nelem);
            }
            else{
              //printf("1Stage 1");
              prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
            }
          }
          else if (p < butterfly->peerCount - 1 || (p == (butterfly->peerCount - 1) && butterfly->lastoneCompressed == 0)) {
            if (myRank < peerRank) {
              //printf("1Stage 2\n");
              //printf("I'm sending data to %d \n", butterfly->peerRanks[p]);
              prims.send(thisInput+offset, nelem);
              //printf("Sending done. start recving \n");
	      prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
              //printf("Recving done \n");
	      //printf("1Stage 2-done\n");
	    }
            else {
              //printf("Doing RCS\n");
              prims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
	      //printf("RCS done\n");
            }
          }
          else {
            if (myRank < peerRank) {
              //printf("0Stage 2\n");
              prims.directRecv(thisOutput+offset, offset, nelem);
	      //prims.recv(thisInput+offset, nelem);
              //prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
	      //printf("0Stage 2-done\n");
            }
            else {
	      //printf("1Stage 3\n");
              prims.directSend(thisOutput+offset, offset, nelem);
              //prims.send(thisInput+offset, nelem);
	      //printf("1Stage 3 done\n");            
            }
          }
      }
  }
  //printf("Kernel done.\n");

}
*/

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceTreeKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads-WARP_SIZE;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
  int chunkSize = args->coll.lastChunkSize;
  const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = nChannels*chunkSize;
  const ssize_t size = args->coll.count;

  if (loopSize > size) {
    chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclPrimitives<UNROLL/2, 1, 1, T, NCCL_MAX_TREE_ARITY, 1, 0, FUNC> prims(tid, nthreads, tree->down, &tree->up, NULL, stepSize, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.send(thisInput+offset, nelem);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclPrimitives<UNROLL/2, 1, 1, T, 1, NCCL_MAX_TREE_ARITY, 1, FUNC> prims(tid, nthreads, &tree->up, tree->down, thisOutput, stepSize, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.directSend(thisOutput+offset, offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.directRecv(thisOutput+offset, offset, nelem);
      } else {
        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }
    }
  } while(0);
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceCollNetKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads-WARP_SIZE;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
  int chunkSize = args->coll.lastChunkSize;
  const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = nChannels*chunkSize;
  const ssize_t size = args->coll.count;

  if (loopSize > size) {
    chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  if (blockIdx.x < nChannels) { // first half of the channels do reduce
    struct ncclTree* tree = &channel->collTreeUp;
    ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC> prims(tid, nthreads, tree->down, &tree->up, NULL, stepSize, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.send(thisInput+offset, nelem);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  }

  if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
    struct ncclTree* tree = &channel->collTreeDn;
    ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC> prims(tid, nthreads, &tree->up, tree->down, NULL, stepSize, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.recv(thisOutput+offset, nelem);
      } else {
        prims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceRingLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
  ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);
  const int nranks = comm->nRanks;
  const ssize_t loopSize = nChannels*nranks*chunkSize;
  const ssize_t size = args->coll.count;

  ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepLines, channel, comm);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(thisOutput+offset, nelem);
  }
}


//butterfly - lyz
template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceButterflyLLKernel(struct CollectiveArgs* args) {
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


template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceTreeLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
  ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = nChannels*chunkSize;
  const ssize_t size = args->coll.count;

  if (loopSize > size) {
    chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclLLPrimitives<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreads, tree->down, &tree->up, stepLines, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclLLPrimitives<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, &tree->up, tree->down, stepLines, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceCollNetLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
  ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = nChannels*chunkSize;
  const ssize_t size = args->coll.count;

  if (loopSize > size) {
    chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  if (blockIdx.x < nChannels) { // first half of the channels do reduce
    struct ncclTree* tree = &channel->collTreeUp;
    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, tree->down, &tree->up, stepLines, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  }

  if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
    struct ncclTree* tree = &channel->collTreeDn;
    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &tree->up, tree->down, stepLines, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  }
}

#include "prims_ll128.h"
template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceRingLL128Kernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
  ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
  // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
  const int nranks = comm->nRanks;
  const ssize_t loopSize = nChannels*nranks*chunkSize;
  const ssize_t size = args->coll.count;

  ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepSize, channel, comm);

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(thisOutput+offset, nelem);
  }
}

//butterfly - lyz
template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceButterflyLL128Kernel(struct CollectiveArgs* args) { }

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceTreeLL128Kernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclTree* treeUp = &channel->treeUp;
  struct ncclTree* treeDn = &channel->treeDn;
  const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
  ssize_t chunkSize = args->coll.lastChunkSize;
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/8;
  const ssize_t loopSize = nChannels*chunkSize;
  int nthreadsSplit = NCCL_LL128_SPLIT(nthreads);
  const ssize_t size = args->coll.count;

  if (loopSize > size) {
    chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  if (treeUp->up == -1) {
    // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
    ncclLL128Primitives<T, FUNC, NCCL_MAX_TREE_ARITY, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, treeUp->down, treeDn->down, stepSize, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
    }
  } else {
    if (tid < nthreadsSplit) {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclLL128Primitives<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreadsSplit, treeUp->down, &treeUp->up, stepSize, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeUp->down[0] == -1) {
          LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    } else {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclLL128Primitives<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid-nthreadsSplit, nthreads-nthreadsSplit, &treeDn->up, treeDn->down, stepSize, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeDn->down[0] == -1) {
          LLprims.recv(thisOutput+offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceCollNetLL128Kernel(struct CollectiveArgs* args) { }

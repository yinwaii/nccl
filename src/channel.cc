/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel.h"
#include "param.h"

ncclResult_t initChannel(struct ncclComm* comm, int channelid) {
  struct ncclChannel* channel = comm->channels+channelid;
  if (channel->id != -1) return ncclSuccess;
  channel->id = channelid;

  // Ring index to user rank table.
  NCCLCHECK(ncclCudaCalloc(&channel->ring.devUserRanks, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->ring.userRanks, comm->nRanks));
  
  NCCLCHECK(ncclCudaCalloc(&channel->butterfly.devPeerRanks, log2i(comm->nRanks)));
  NCCLCHECK(ncclCalloc(&channel->butterfly.peerRanks, log2i(comm->nRanks)));

  NCCLCHECK(ncclCudaCalloc(&channel->meshCross.devIntraRanks, comm->nSubRanks * comm->nPartitions));
  NCCLCHECK(ncclCalloc(&channel->meshCross.intraRanks, comm->nSubRanks * comm->nPartitions));

  NCCLCHECK(ncclCudaCalloc(&channel->meshCross.devInterRanks, comm->nPartitions));
  NCCLCHECK(ncclCalloc(&channel->meshCross.interRanks, comm->nPartitions));

  NCCLCHECK(ncclCudaCalloc(&channel->butterfly2d.devIntraRanks, comm->localRanks));
  NCCLCHECK(ncclCalloc(&channel->butterfly2d.intraRanks, comm->localRanks));

  NCCLCHECK(ncclCudaCalloc(&channel->butterfly2d.devPeerRanks, log2i(comm->nNodes)));
  NCCLCHECK(ncclCalloc(&channel->butterfly2d.peerRanks, log2i(comm->nNodes)));

  NCCLCHECK(ncclCudaCalloc(&channel->ring2d.devIntraRanks, comm->localRanks));
  NCCLCHECK(ncclCalloc(&channel->ring2d.intraRanks, comm->localRanks));

  NCCLCHECK(ncclCudaCalloc(&channel->ring2d.devInterRanks, comm->nNodes));
  NCCLCHECK(ncclCalloc(&channel->ring2d.interRanks, comm->nNodes));

  // Communication structures with peers.
  NCCLCHECK(ncclCudaCalloc(&channel->devPeers, comm->nRanks+1)); // The extra one rank is for collnet root (i.e. network)
  NCCLCHECK(ncclCalloc(&channel->peers, comm->nRanks+1));
  for (size_t i=0; i<comm->nRanks+1; ++i) {
    channel->peers[i].send.comm = comm;
    channel->peers[i].recv.comm = comm;
  }

  // Per-channel operation list.
  NCCLCHECK(ncclCudaHostCalloc(&channel->collectives, NCCL_MAX_OPS));
  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks) {
  if (channel->id == -1) return ncclSuccess;
  // Operation list
  NCCLCHECK(ncclCudaHostFree(channel->collectives));

  // Free Ring index to rank tables
  free(channel->ring.userRanks);
  CUDACHECK(cudaFree(channel->ring.devUserRanks));

  free(channel->butterfly.peerRanks);
  CUDACHECK(cudaFree(channel->butterfly.devPeerRanks));

  free(channel->meshCross.intraRanks);
  CUDACHECK(cudaFree(channel->meshCross.devIntraRanks));

  free(channel->meshCross.interRanks);
  CUDACHECK(cudaFree(channel->meshCross.devInterRanks));

  free(channel->butterfly2d.peerRanks);
  CUDACHECK(cudaFree(channel->butterfly2d.devPeerRanks));

  free(channel->butterfly2d.intraRanks);
  CUDACHECK(cudaFree(channel->butterfly2d.devIntraRanks));

  free(channel->ring2d.interRanks);
  CUDACHECK(cudaFree(channel->ring2d.devInterRanks));

  free(channel->ring2d.intraRanks);
  CUDACHECK(cudaFree(channel->ring2d.devIntraRanks));

  // Free transport proxy resources
  // Note: free all send resources first due to CollNet arrangement
  for (int r=0; r<nRanks+1; r++) {
    struct ncclPeer* peer = channel->peers+r;
    if (peer->send.transportResources) NCCLCHECK(peer->send.transportComm->free(peer->send.transportResources));
  }
  for (int r=0; r<nRanks+1; r++) {
    struct ncclPeer* peer = channel->peers+r;
    if (peer->recv.transportResources) NCCLCHECK(peer->recv.transportComm->free(peer->recv.transportResources));
  }

  // Free the peer structures.
  CUDACHECK(cudaFree(channel->devPeers));
  free(channel->peers);

  return ncclSuccess;
}

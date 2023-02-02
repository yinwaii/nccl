#include "algo_interface.h"
#include "collectives.h"
#include <assert.h>

// Topo

ncclTopoButterfly_yz::ncclTopoButterfly_yz(struct ncclComm *comm): ncclTopoBase(NCCL_ALGO_BUTTERFLY_YZ, comm, ncclParamCrossNic(), 0) {}

ncclResult_t ncclTopoButterfly_yz::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank, nranks = comm->nRanks;
  WARN("xxx Preset");
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
	int* butterflyIntra = graph.intra+c*localRanks;
    for (int i=0; i<localRanks; i++) {
      //butterfly_yz - lyz
      if (butterflyIntra[i] == rank) {
        topoRanks->butterflyRecv[c] = butterflyIntra[0];
        topoRanks->butterflySend[c] = butterflyIntra[localRanks-1];
        //channel->butterfly_yz.prev = (i == 0) ? -1 : butterflyIntra[i-1];
        //channel->butterfly_yz.next = (i == localRanks-1) ? -1 : butterflyIntra[i+1];
        //topoRanks->butterflyPrev[c] = channel->butterfly_yz.prev;
        //topoRanks->butterflyNext[c] = channel->butterfly_yz.next;
      }
	}
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoButterfly_yz::connectButterfly(struct ncclComm* comm, int* butterflyRecv, int* butterflySend, int* firstRanks) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  int nRanks = comm->nRanks;
  int myRank = comm->rank;

  for (int c=0; c< 2 * nChannels; c++) {
    //not used for now
    int* recv = butterflyRecv+c*comm->nRanks;
    int* send = butterflySend+c*comm->nRanks;

    struct ncclChannel* channel = comm->channels+c;
    channel->butterfly_yz.peerCount = 0;
    channel->butterfly_yz.lastoneCompressed = 0;
    channel->butterfly_yz.myRank = myRank;

    //algorithm
    if (nRanks == 1) {
      channel->butterfly_yz.peerRanks[channel->butterfly_yz.peerCount++] = myRank;
      return ncclSuccess;
    }

    int adjsize = 1;
    while (adjsize <= nRanks) adjsize <<= 1;
    adjsize >>= 1;

    int newRank = myRank;
    //cases for nRanks non-power of 2
    int extra_ranks = nRanks - adjsize;
    if (myRank < (2 * extra_ranks)) {
      if ((myRank % 2) == 0) {
        channel->butterfly_yz.peerRanks[channel->butterfly_yz.peerCount++] = myRank + 1;
        newRank = -1;
      }
      else {
        channel->butterfly_yz.peerRanks[channel->butterfly_yz.peerCount++] = myRank - 1;
        newRank >>= 1;
      }
    }
    else{
      newRank -= extra_ranks;
    }

    //determine peerRanks
    for (int dist = 1; dist < adjsize; dist <<= 1) {
      if (newRank < 0) break;
      int newRemote = newRank ^ dist;
      int remote = (newRemote < extra_ranks)? (newRemote * 2 + 1):(newRemote + extra_ranks);
      channel->butterfly_yz.peerRanks[channel->butterfly_yz.peerCount++] = remote;
    }

    //deal with those compressed ranks
    if (myRank < 2 * extra_ranks) {
      channel->butterfly_yz.lastoneCompressed = 1;
      if ((myRank % 2) == 0) {
        channel->butterfly_yz.peerRanks[channel->butterfly_yz.peerCount++] = myRank + 1;
      }
      else {
        channel->butterfly_yz.peerRanks[channel->butterfly_yz.peerCount++] = myRank - 1;
      }
    }

    //char line[1024];
    for (int p = 0; p < channel->butterfly_yz.peerCount; p++) {
      ///int offset = strlen(line);
      //sprintf(line+offset, "%2d ", channel->butterfly_yz.peerRanks[p]);
      INFO(NCCL_INIT,"Butterfly Rank %d, communicates with : %d (%d)", myRank, channel->butterfly_yz.peerRanks[p], channel->butterfly_yz.peerCount);
    }
    //INFO(NCCL_INIT, "Butterfly Rank %d, communicates with : %s", myRank, line);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoButterfly_yz::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  WARN("xxx Postset");
  // Gather data from all ranks
  int *butterflyRecv, *butterflySend;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  NCCLCHECK(ncclCalloc(&butterflyRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&butterflySend, nranks*MAXCHANNELS));
  for (int i=0; i<nranks; i++) {
    for (int c=0; c<nChannels;c++) {
      //butterfly_yz - lyz
      butterflyRecv[c*nranks+i] = allTopoRanks[i]->butterflyRecv[c];
      butterflySend[c*nranks+i] = allTopoRanks[i]->butterflySend[c];
    }
  }
  //butterfly_yz - lyz
  NCCLCHECK(connectButterfly(comm, butterflyRecv, butterflySend, firstRanks));
  free(butterflyRecv);
  free(butterflySend);
  return ncclSuccess;
}

ncclResult_t ncclTopoButterfly_yz::transportSetup() {
  WARN("xxx Transport");
  INFO(NCCL_INIT, "Setting up butterfly_yz connection ...");
  for (int c=0; c<comm->nChannels; c++) {
	struct ncclChannel* channel = comm->channels+c;
	if (comm->nRanks == 1) continue;
	if (channel->butterfly_yz.lastoneCompressed == 0) {
	  NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, channel->butterfly_yz.peerCount, channel->butterfly_yz.peerRanks, channel->butterfly_yz.peerCount, channel->butterfly_yz.peerRanks));
	}
	else {
	  NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, channel->butterfly_yz.peerCount - 1, channel->butterfly_yz.peerRanks, channel->butterfly_yz.peerCount - 1, channel->butterfly_yz.peerRanks));
	}
	INFO(NCCL_INIT, "Butterfly connection established!");
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly_yz::getPattern(int coll, int *pattern) const {
  switch (coll) {
    case ncclCollBroadcast:
      *pattern = ncclPatternHalfDoubling;
      break;
    case ncclCollAllReduce:
      *pattern = ncclPatternButterfly;
      break;
    default:
      *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly_yz::proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo* info) const {
  WARN("xxx Save");
  int pattern = info->pattern;
  struct ncclButterfly_yz *butterfly_yz = &args->channel->butterfly_yz;
  int nRanks = info->comm->nRanks;
  //butterfly_yz - lyz
  if (pattern == ncclPatternButterfly) {
	//printf("Save proxy: butterfly_yz\n");
	struct ncclButterfly_yz* butterfly_yz = &args->channel->butterfly_yz;
	int myRank = butterfly_yz->myRank;
	
	// lyz - segmented - different nsteps for different peers
	int commSize = info->nBytes;
	//int commOffset = 0;

	int peerSteps[1024];
	int reducedPeerRanks[1024];
	int reducedPeerCount = 0;

	for (int p = 0; p < butterfly_yz->peerCount; p++) {
	int peerRank = butterfly_yz->peerRanks[p];
	int nsteps = 0;
	if (p == (butterfly_yz->peerCount - 1) && butterfly_yz->lastoneCompressed == 1 || p == 0 && butterfly_yz->lastoneCompressed == 1) continue;
	int halfSize = commSize/2;

	/*calculate steps*/
	struct ncclPeer* peerComm = args->channel->peers+peerRank;
	struct ncclConnector* connector = &peerComm->recv;

	int stepSize   = connector->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
	int chunkSteps = ALLREDUCE_CHUNKSTEPS;
	//int sliceSteps = ALLREDUCE_SLICESTEPS;
	int chunkSize  = stepSize*chunkSteps;
	int nchunksPerLoop = 1;
	int nstepsPerLoop = 1;

printf("Enqueue  size: %d\n", commSize);

	int nLoops = (int)(DIVUP(halfSize, (((size_t)(connector->comm->nChannels))*nchunksPerLoop*chunkSize)));
	nsteps += nstepsPerLoop * nLoops * chunkSteps;
	nsteps += nstepsPerLoop * nLoops * chunkSteps;

	commSize = halfSize;
	peerSteps[reducedPeerCount] = nsteps;
	reducedPeerRanks[reducedPeerCount] = peerRank;
	reducedPeerCount++;
	}
	/** Enqueue proxies **/
	if (butterfly_yz->lastoneCompressed == 1) {
	  NCCLCHECK(SaveProxy<proxySend>(butterfly_yz->peerRanks[0], args));
	  NCCLCHECK(SaveProxy<proxyRecv>(butterfly_yz->peerRanks[0], args));
	}
	for (int p = 0; p < reducedPeerCount; p++) {
	  NCCLCHECK(SaveProxy<proxySend>(reducedPeerRanks[p], args, peerSteps[p]));
	  NCCLCHECK(SaveProxy<proxyRecv>(reducedPeerRanks[p], args, peerSteps[p]));
	}
  }
  if (pattern == ncclPatternHalfDoubling) {
	INFO(NCCL_INIT, "Butterfly broadcast !!!!");
	// determine the peer ranks
	int recvPeerRank = -1;
	int sendPeerRanks[32];
	int sendPeerCount = 0;

	int nRanks = info->comm->nRanks;
	int adjsize = 1;
	while (adjsize <= nRanks)
		adjsize <<= 1;
	adjsize >>= 1;

	int myRank = args->channel->butterfly_yz.myRank;
	int myNewRank = myRank;

	int rootRank = info->root;
	int rootNewRank = info->root;

	// cases for nRanks non-power of 2
	int extra_ranks = nRanks - adjsize;
	if (myRank < (2 * extra_ranks))
	{
		if ((myRank % 2) == 0)
			myNewRank = -1;
		else
			myNewRank >>= 1;
	}
	else
		myNewRank -= extra_ranks;
	/***root***/
	if (rootRank < (2 * extra_ranks))
	{
		if ((rootRank % 2) == 0)
		{ // root is in extra ranks
			if (myRank == rootRank)
			{
				sendPeerRanks[sendPeerCount++] = rootRank + 1;
			}
			if (myRank == rootRank + 1)
				recvPeerRank = rootRank;
		}
		rootNewRank >>= 1;
	}
	else
		rootNewRank -= extra_ranks;
	/***root***/

	int roundStepSize = adjsize / 2;
	// find the recv peer
	if (myNewRank == -1)
	{
		if (myRank != rootRank)
			recvPeerRank = myRank + 1; // no send peer
		roundStepSize = 0;
	}
	else
	{
		// starting from the root to my rank
		if (myNewRank != rootNewRank)
		{
			int currentRank = rootNewRank;
			while (currentRank != myNewRank)
			{
				int myRankLoc = myNewRank / roundStepSize;
				int currentRankLoc = currentRank / roundStepSize;

				if (myRankLoc > currentRankLoc)
				{
					if ((currentRank + roundStepSize) == myNewRank)
						recvPeerRank = currentRank;
					currentRank += roundStepSize;
				}
				if (myRankLoc < currentRankLoc)
				{
					if ((currentRank - roundStepSize) == myNewRank)
						recvPeerRank = currentRank;
					currentRank -= roundStepSize;
				}
				roundStepSize >>= 1;
			}
			assert(recvPeerRank != -1);
			recvPeerRank = (recvPeerRank < extra_ranks) ? (recvPeerRank * 2 + 1) : (recvPeerRank + extra_ranks);
		}
	}

	// remaining are send peers
	while (roundStepSize > 0)
	{
		int sendPeerRank;
		int loc = (myNewRank) % (roundStepSize * 2);
		if ((loc + 1) > roundStepSize)
			sendPeerRank = myNewRank - roundStepSize;
		else
			sendPeerRank = myNewRank + roundStepSize;
		sendPeerRank = (sendPeerRank < extra_ranks) ? (sendPeerRank * 2 + 1) : (sendPeerRank + extra_ranks);
		sendPeerRanks[sendPeerCount++] = sendPeerRank;
		roundStepSize >>= 1;
	}
	if (myNewRank != -1 && myNewRank < extra_ranks && (myRank - 1) != rootRank)
		sendPeerRanks[sendPeerCount++] = (myRank - 1);

	// check
	if (myRank == rootRank)
		assert(recvPeerRank == -1);
	INFO(NCCL_INIT, "Butterfly broadcast %d, recv from : %d", myRank, recvPeerRank);
	for (int p = 0; p < sendPeerCount; p++)
	{
		INFO(NCCL_INIT, "Butterfly broadcast %d, send to : %d (%d)", myRank, sendPeerRanks[p], sendPeerCount);
	}

	// enqueue all the proxies
	if (recvPeerRank > -1)
		NCCLCHECK(SaveProxy<proxyRecv>(recvPeerRank, args));
	for (int i = 0; i < sendPeerCount; i++)
	{
		NCCLCHECK(SaveProxy<proxySend>(sendPeerRanks[i], args));
	}
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly_yz::enqueueLoopInfo(struct ncclInfo *info) const {
  switch (info->pattern) {
  case ncclPatternHalfDoubling:
	  info->nchunksPerLoop = 1;
	  info->nstepsPerLoop = 1;
	  break;
  case ncclPatternButterfly:
    info->nchunksPerLoop = 1;
    info->nstepsPerLoop = 1;
    break;
  default:
    WARN("Unknown pattern %d\n", info->pattern);
    return ncclInternalError;
  }
  return ncclSuccess;
}

#ifndef __MACROS_H__
#define __MACROS_H__
#include <stdint.h>

#define MAP_FOR_ALGOS(f, ...) \
	f(TREE, ##__VA_ARGS__) \
	f(RING, ##__VA_ARGS__) \
	f(COLLNET, ##__VA_ARGS__) \
  f(BUTTERFLY, ##__VA_ARGS__) \
  f(BUTTERFLY2, ##__VA_ARGS__) \
  f(BUTTERFLY_YZ, ##__VA_ARGS__)

#define NCCL_NUM_ALGORITHMS 6 // Tree/Ring/CollNet
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET 2
#define NCCL_ALGO_BUTTERFLY 3
#define NCCL_ALGO_BUTTERFLY2 4
#define NCCL_ALGO_BUTTERFLY_YZ 5
extern const char *ncclAlgoStr[NCCL_NUM_ALGORITHMS];

#define NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP 1 // Split tree (send/recv from different ranks) always flowing in the same direction
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Split tree (send/recv from different ranks) flowing in both directions
#define NCCL_TOPO_PATTERN_TREE 3            // Simple tree (send/recv from same rank) flowing in both directions
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_BUTTERFLY 5       // Butterfly

struct ncclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;
  int* devUserRanks;
};


#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

struct ncclButterfly {
  int edgeRank;
  int *peerRanks;
  int *devPeerRanks;
};

#define NCCL_MAX_BUTTERFLY_STEPS 10
struct ncclButterfly_yz
{
  int myRank;
  int peerCount;
  int lastoneCompressed;
  int peerRanks[NCCL_MAX_BUTTERFLY_STEPS];
};

struct ncclChannel {
  union {
    struct {
      struct ncclRing ring;
      struct ncclTree treeUp;
      struct ncclTree treeDn;
      struct ncclTree collTreeUp;
      struct ncclTree collTreeDn;
      struct ncclButterfly butterfly;
      struct ncclButterfly_yz butterfly_yz;

      int id;

      // Communication structures
      struct ncclPeer* peers;
      struct ncclPeer* devPeers;

      // Operation list for aggregation
      struct ncclColl* collectives;
      int collStart;
      int collCount;
      int collFifoHead; // Only used by GPU
      int collFifoTail; // Only used by CPU
    };
    int data[0x80];
  };
};
static_assert(sizeof(struct ncclChannel) == 0x80*sizeof(int), "ncclChannel must have a pow2 size");

#define MAXCHANNELS 32
struct ncclTopoRanks {
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeUpRecv[MAXCHANNELS];
  int treeUpSend[MAXCHANNELS];
  int treeDnRecv[MAXCHANNELS];
  int treeDnSend[MAXCHANNELS];
  // butterfly - lyz
  int butterflyRecv[MAXCHANNELS];
  int butterflySend[MAXCHANNELS];
};


extern const char *ncclAlgoStr[NCCL_NUM_ALGORITHMS];

#endif
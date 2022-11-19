#ifndef __MACROS_H__
#define __MACROS_H__
#include <stdint.h>

#define MAP_FOR_ALGOS(f, ...) \
	f(TREE, ##__VA_ARGS__) \
	f(RING, ##__VA_ARGS__) \
	f(COLLNET, ##__VA_ARGS__)

#define NCCL_NUM_ALGORITHMS 3 // Tree/Ring/CollNet
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET 2

#define NCCL_TOPO_PATTERN_BALANCED_TREE 1 // Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2	  // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3		  // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4		  // Ring

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

struct ncclChannel {
  union {
    struct {
      struct ncclRing ring;
      struct ncclTree tree;
      struct ncclTree collTree;

      int id;

      // Communication structures
      struct ncclPeer* peers;
      struct ncclPeer* devPeers;

      // Operation list for aggregation
      struct ncclWork* workFifo;
      int workCount;
      uint64_t workFifoTail; // Only used by CPU
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
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];
};

extern const char *ncclAlgoStr[NCCL_NUM_ALGORITHMS];

#endif
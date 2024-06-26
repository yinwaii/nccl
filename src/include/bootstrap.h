/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"
#include "nccl_net.h"
#include <memory>
#include <array>

namespace ncclpp {
using UniqueId = std::array<uint8_t, NCCL_UNIQUE_ID_BYTES>;
/// Base class for bootstraps.
class Bootstrap {
 public:
  Bootstrap(){};
  virtual ~Bootstrap() = default;
  virtual int getRank() = 0;
  virtual int getNranks() = 0;
  virtual void send(void* data, int size, int peer) = 0;
  virtual void recv(void* data, int size, int peer) = 0;
  virtual void allGather(void* allData, int size) = 0;
};
/// A native implementation of the bootstrap using TCP sockets.
class TcpBootstrap : public Bootstrap {
 public:
  static UniqueId createUniqueId();
  UniqueId getUniqueId() const;
  TcpBootstrap(int rank, int nRanks);
  ~TcpBootstrap();

  void initialize(UniqueId *uniqueId);
  int getRank() override;
  int getNranks() override;
  void send(void* data, int size, int peer) override;
  void recv(void* data, int size, int peer) override;
  void allGather(void* allData, int size) override;
	void close();

private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

}

ncclResult_t bootstrapNetInit();
ncclResult_t bootstrapCreateRoot(ncclUniqueId* commId, bool idFromEnv);
ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out);
ncclResult_t bootstrapInit(ncclUniqueId* id, int rank, int nranks, void** commState);
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);
ncclResult_t bootstrapSend(void* commState, int peer, void* data, int size);
ncclResult_t bootstrapRecv(void* commState, int peer, void* data, int size);
ncclResult_t bootstrapClose(void* commState);
ncclResult_t bootstrapAbort(void* commState);
#endif

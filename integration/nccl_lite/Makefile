CXX ?= g++
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
REAL_NCCL ?= libnccl.so.2
TORCHRUN ?= torchrun
BUILD := build
LITE := ../../experimental/lite

.PHONY: all check check-torch clean
all: $(BUILD)/libuccl_lite_nccl_compat.so

check: $(BUILD)/test_compat
	UCCL_LITE_REAL_NCCL=$(REAL_NCCL) UCCL_LITE_NCCL_ENABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 ./$(BUILD)/test_compat

check-torch: $(BUILD)/libuccl_lite_nccl_compat.so
	LD_PRELOAD=$(abspath $(BUILD)/libuccl_lite_nccl_compat.so) UCCL_LITE_REAL_NCCL=$(REAL_NCCL) UCCL_LITE_NCCL_ENABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 $(TORCHRUN) --standalone --nproc-per-node=8 test_torch.py

$(BUILD):
	mkdir -p $@

$(BUILD)/libuccl_lite_nccl_compat.so: nccl_compat.cc $(LITE)/src/collectives/multiprocess_collectives.cc $(LITE)/include/uccl_lite/multiprocess_collectives.h | $(BUILD)
	$(CXX) -O2 -std=c++17 -fPIC -shared -pthread -I$(LITE)/include -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include nccl_compat.cc $(LITE)/src/collectives/multiprocess_collectives.cc -L$(CUDA_HOME)/lib64 -lcudart -ldl -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

$(BUILD)/test_compat: test_compat.cc $(BUILD)/libuccl_lite_nccl_compat.so
	$(CXX) -O2 -std=c++17 -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include $< -L$(BUILD) -luccl_lite_nccl_compat -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(abspath $(BUILD)) -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

clean:
	rm -rf $(BUILD)

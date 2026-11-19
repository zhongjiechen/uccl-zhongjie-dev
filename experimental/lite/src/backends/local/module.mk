# Eight-L4 BF16 transport, sharing the strict ABI and CPU control protocol.
# This target has no native NCCL or MSCCL++ link dependency.
.PHONY: nccl-local8-strict uccl_lite_local8_flags_force
nccl-local8-strict: $(BUILD)/nccl_local8/strict/libnccl.so.2

LOCAL8_PACKET_BYTES ?= 1048576
LOCAL8_PACKET_FLAGS = -DUCCL_LITE_MP_PACKET_BYTES=$(LOCAL8_PACKET_BYTES) -DUCCL_LITE_MP_AG_BLOCKS=64 -DUCCL_LITE_MP_AG_SMALL_BLOCKS=1 -DUCCL_LITE_MP_AG_SINGLE_PRODUCERS=32 -DUCCL_LITE_MP_WAIT_LANES=8 -DUCCL_LITE_MP_RS_BLOCKS=64 -DUCCL_LITE_MP_RS_SINGLE_TILE_PRODUCERS=56
LOCAL8_HOST_FLAGS = -DUCCL_LITE_MP_PACKET_BYTES=$(LOCAL8_PACKET_BYTES) -DUCCL_LITE_MP_AG_PIPELINE_MIN_BYTES=4194304 -DUCCL_LITE_MP_PARALLEL_RS_SINGLE_TILE -DUCCL_LITE_MP_POLL_WORKERS -DUCCL_LITE_MP_GLOBAL_WORKER_ISOLATION
LOCAL8_OBJECTS = $(NCCL_ABI_OBJECTS) $(BUILD)/backends/local/backend.o $(BUILD)/backends/local/transport.o $(BUILD)/backends/local/kernels.o $(BUILD)/cpu_router/cpu_executor.o

uccl_lite_local8_flags_force:

$(BUILD)/nccl_local8/build.flags: uccl_lite_local8_flags_force
	mkdir -p $(@D)
	@printf '%s\n' '$(NVCC) $(NVCC_FLAGS) $(LOCAL8_PACKET_FLAGS)' '$(CXX) $(LOCAL8_HOST_FLAGS)' '$(CUDA_HOME) $(NCCL_HOME)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

$(BUILD)/backends/local/kernels.o: src/backends/local/kernels.cu src/backends/local/packet_layout.cuh $(BUILD)/nccl_local8/build.flags
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(LOCAL8_PACKET_FLAGS) -std=c++17 -Xcompiler=-fPIC -Iinclude -c $< -o $@

$(BUILD)/backends/local/backend.o: src/backends/local/backend.cc $(BUILD)/nccl_local8/build.flags
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -pthread -MMD -MP -Iinclude -I$(CUDA_HOME)/include -c $< -o $@

$(BUILD)/backends/local/transport.o: src/backends/local/transport.cc src/backends/local/packet_layout.cuh $(BUILD)/nccl_local8/build.flags
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -pthread -MMD -MP $(LOCAL8_HOST_FLAGS) -Iinclude -I$(CUDA_HOME)/include -c $< -o $@

$(BUILD)/nccl_local8/libuccl_lite_nccl_local8.so: $(LOCAL8_OBJECTS) integration/nccl_lite/nccl_exports.version
	$(CXX) -shared -pthread $(LOCAL8_OBJECTS) -Wl,--version-script=integration/nccl_lite/nccl_exports.version -Wl,-soname,libnccl.so.2 -Wl,-z,defs -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64 -o $@

$(BUILD)/nccl_local8/strict/libnccl.so.2: $(BUILD)/nccl_local8/libuccl_lite_nccl_local8.so
	mkdir -p $(@D)
	ln -sfn ../libuccl_lite_nccl_local8.so $@

-include $(wildcard $(BUILD)/backends/local/*.d $(BUILD)/nccl_local8/*.d)

.PHONY: nccl-local
nccl-local: nccl-local8-strict

include src/backends/multinode/policy.mk

# Track named policy changes even when an existing BUILD directory is reused.
# Experimental EXTRA_NVCC_FLAGS changes still require a fresh BUILD directory.
.PHONY: uccl_lite_nccl_policy_force
uccl_lite_nccl_policy_force:

$(BUILD)/nccl_lite/build.policy: uccl_lite_nccl_policy_force
	mkdir -p $(@D)
	@printf '%s\n' '$(UCCL_LITE_NCCL_POLICY)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

.PHONY: nccl-mscclpp nccl-mscclpp-strict
nccl-mscclpp: nccl-mscclpp-strict
nccl-mscclpp-strict: $(BUILD)/nccl_lite/strict/libnccl.so.2 $(BUILD)/nccl_lite/libuccl_lite_nccl_mscclpp.so

$(BUILD)/nccl_lite/libuccl_lite_nccl_strict.so: $(NCCL_ABI_OBJECTS) $(BUILD)/backends/multinode/backend.o $(BUILD)/backends/multinode/collectives.o integration/nccl_lite/nccl_exports.version
	$(CXX) -shared -pthread $(NCCL_ABI_OBJECTS) $(BUILD)/backends/multinode/backend.o $(BUILD)/backends/multinode/collectives.o -Wl,--version-script=integration/nccl_lite/nccl_exports.version -Wl,-soname,libnccl.so.2 -L$(MSCCLPP_IB_BUILD)/lib -lmscclpp -Wl,-rpath,$(MSCCLPP_IB_BUILD)/lib -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64 $(UCCL_LITE_NCCL_POLICY_LIBS) $(NCCL_STRICT_EXTRA_LIBS) -o $@

$(BUILD)/nccl_lite/strict/libnccl.so.2: $(BUILD)/nccl_lite/libuccl_lite_nccl_strict.so
	mkdir -p $(@D)
	ln -sfn ../libuccl_lite_nccl_strict.so $@
$(BUILD)/backends/multinode/collectives.o: src/backends/multinode/collectives.cu src/backends/multinode/device_collectives.cuh src/backends/multinode/algorithm_transport.h include/uccl_lite/backends/multinode.h
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) $(UCCL_LITE_NCCL_POLICY_FLAGS) -std=c++20 -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden $(IB_INC) -c $< -o $@

$(BUILD)/backends/multinode/collectives.o: $(BUILD)/nccl_lite/build.policy src/backends/multinode/policy.mk

$(BUILD)/backends/multinode/backend.o: src/backends/multinode/backend.cc $(BUILD)/runtime/build.flags
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -pthread -MMD -MP -Iinclude -I$(CUDA_HOME)/include -c $< -o $@

.PHONY: nccl-multinode
nccl-multinode: nccl-mscclpp-strict

-include $(wildcard $(BUILD)/backends/multinode/*.d)

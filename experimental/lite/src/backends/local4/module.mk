include src/backends/local4/policy.mk

.PHONY: uccl_lite_nccl_local_policy_force
uccl_lite_nccl_local_policy_force:

$(BUILD)/nccl_local:
	mkdir -p $@

$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_POLICY = $(UCCL_LITE_NCCL_LOCAL_POLICY)
$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_NVCC = $(NVCC)
$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_CXX = $(CXX)
$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_NVCC_FLAGS = $(NVCC_FLAGS) $(UCCL_LITE_NCCL_LOCAL_FLAGS) $(UCCL_LITE_NCCL_LOCAL_REPRO_FLAGS)
$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_INCLUDES = $(MSCCLPP_INC) -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
$(BUILD)/nccl_local/build.policy: export UCCL_LOCAL_BUILD_LIBS = $(MSCCLPP_BUILD)/lib $(CUDA_HOME)/lib64 $(UCCL_LITE_NCCL_LOCAL_LIBS) $(NCCL_LOCAL_EXTRA_LIBS)
$(BUILD)/nccl_local/build.policy: uccl_lite_nccl_local_policy_force integration/nccl_lite/write_build_stamp.py | $(BUILD)/nccl_local
	$(PYTHON) -B integration/nccl_lite/write_build_stamp.py $@

.PHONY: nccl-local-strict
nccl-local-strict: $(BUILD)/nccl_local/strict/libnccl.so.2
$(BUILD)/nccl_local/local_strict.o: src/backends/local4/collectives.cu src/backends/local4/collectives.h $(BUILD)/nccl_local/build.policy src/backends/local4/policy.mk
	mkdir -p $(@D) $(@D)/cuda-intermediates
	$(NVCC) $(NVCC_FLAGS) $(UCCL_LITE_NCCL_LOCAL_FLAGS) $(UCCL_LITE_NCCL_LOCAL_REPRO_FLAGS) -std=c++17 -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden $(MSCCLPP_INC) -c $< -o $@
$(BUILD)/nccl_local/libuccl_lite_nccl_local_strict.so: $(NCCL_ABI_OBJECTS) $(BUILD)/backends/local4/backend.o $(BUILD)/nccl_local/local_strict.o integration/nccl_lite/nccl_exports.version $(BUILD)/nccl_local/build.policy $(UCCL_LITE_NCCL_LOCAL_DEPS)
	$(CXX) -shared -pthread $(NCCL_ABI_OBJECTS) $(BUILD)/backends/local4/backend.o $(BUILD)/nccl_local/local_strict.o -Wl,--version-script=integration/nccl_lite/nccl_exports.version -Wl,-soname,libnccl.so.2 -L$(MSCCLPP_BUILD)/lib -lmscclpp -Wl,-rpath,$(MSCCLPP_BUILD)/lib -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64 $(UCCL_LITE_NCCL_LOCAL_LIBS) $(NCCL_LOCAL_EXTRA_LIBS) -o $@

$(BUILD)/backends/local4/backend.o: src/backends/local4/backend.cc $(BUILD)/runtime/build.flags
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -pthread -MMD -MP -Iinclude -I$(CUDA_HOME)/include -c $< -o $@

.PHONY: nccl-local4
nccl-local4: nccl-local-strict
-include $(wildcard $(BUILD)/backends/local4/*.d)

$(BUILD)/nccl_local/strict/libnccl.so.2: $(BUILD)/nccl_local/libuccl_lite_nccl_local_strict.so
	mkdir -p $(@D)
	ln -sfn ../libuccl_lite_nccl_local_strict.so $@
-include $(BUILD)/nccl_local/local_strict.d
$(BUILD)/nccl_local/strict_abi_check: tests/nccl_mscclpp/strict_abi.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DUCCL_LITE_ABI_LOCAL -DUCCL_LITE_ABI_EXTENDED -DUCCL_LITE_ABI_BOUNDARY512 -DUCCL_LITE_ABI_SERIAL_PACKET -I$(NCCL_HOME)/include $(MPI_INC) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib $(MPI_LIB) -lcudart

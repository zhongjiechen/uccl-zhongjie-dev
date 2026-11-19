# One NCCL frontend/profile implementation, linked unchanged into both targets.
NCCL_ABI_OBJECTS = $(BUILD)/runtime/nccl_abi.o $(BUILD)/runtime/profile.o

.PHONY: uccl_lite_runtime_flags_force
uccl_lite_runtime_flags_force:

$(BUILD)/runtime/build.flags: uccl_lite_runtime_flags_force
	mkdir -p $(@D)
	@printf '%s\n' '$(CXX) $(CUDA_HOME) $(NCCL_HOME)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp

$(BUILD)/runtime/nccl_abi.o: integration/nccl_lite/nccl_abi.cc $(BUILD)/runtime/build.flags
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -pthread -MMD -MP -Iinclude -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -c $< -o $@

$(BUILD)/runtime/profile.o: src/runtime/profile.cc $(BUILD)/runtime/build.flags
	mkdir -p $(@D)
	$(CXX) -O3 -std=c++17 -fPIC -MMD -MP -Iinclude -I$(CUDA_HOME)/include -c $< -o $@

-include $(wildcard $(BUILD)/runtime/*.d)

.PHONY: backend-profile-check
backend-profile-check: $(BUILD)/runtime/backend_profile_check
	$<

$(BUILD)/runtime/backend_profile_check: tests/runtime/backend_profile_check.cc src/runtime/profile.cc include/uccl_lite/runtime/profile.h include/uccl_lite/backend.h
	mkdir -p $(@D)
	$(CXX) -O2 -std=c++17 -Iinclude -I$(CUDA_HOME)/include tests/runtime/backend_profile_check.cc src/runtime/profile.cc -o $@

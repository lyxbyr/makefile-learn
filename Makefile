
src_directory := src
obj_directory := objs


cc_srcs := $(shell find $(src_directory) -name "*.cc")
#$(info $(cc_srcs)) 
cc_objs := $(patsubst %.cc, %.o, $(cc_srcs))
cc_objs := $(subst $(src_directory)/,$(obj_directory)/,$(cc_objs))
#$(info $(cc_objs))

.PYTHON : debug

clean:
	@rm -rf objs pro

debug:
	@echo $(cc_objs)

run : pro
	@./$<

pro : $(cc_objs)
	@g++ $^ -o $@

$(obj_directory)/%.o : $(src_directory)/%.cc
#	@echo $(dir $@)
	@mkdir -p $(dir $@)
	@g++ -g -c $< -o $@




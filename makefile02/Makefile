


# 定义cpp源代码路径，并转换为objs下的o文件
cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst %.cpp, %.o, $(cpp_srcs))
cpp_objs := $(subst src/, objs/, $(cpp_objs))

# 定义cu源代码路径，并转换为objs下的cuo文件
cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(patsubst %.cu, %.cuo, $(cu_srcs))
cu_objs := $(subst src/, objs/, $(cu_objs))


# 定义参数，可以随意修改
workspace := workspace
binary    := pro

# 这里定义头文件库文件和链接目标没有加-I -L, -l, 后面用foreach一次性增加
include_paths := /usr/local/cuda-10.2/include
library_paths  := /usr/local/cuda-10.2/lib64
link_librarys  := cudart

# 定义编译选项
cpp_compile_flags := -m64 -fPIC -g -O0 -std=c++11
cu_compile_flags  := -m64 -g -O0 -std=c++11


# 2种类型
# 1. 字符串
# 2.字符串数组
# 空格隔开就是数组
#
# 对头文件、库文件、目标统一增加-I -L -l
# foreach var, list cmd
#			var  = item
#			list = link_librarys
#     cmd  = -Wl, -rpath=$(item)		
#
# output = []
# for item in link_library:
# 		output.append(f"-Wl, -rpath={item}")
# rpaths = output

# -L 指定链接目标时查找的目录
# -l 指定链接的目标名称。符合libname.so, -lname 规则
# -I 指定编译时头文件查找目录
rpaths        := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 合并选项
cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        := $(rapths) $(library_paths) $(link_librarys)




#$@ 生成项
#$< 依赖项第一个
#$^ 依赖项所有
#$? s+

objs/%.o : src/%.cpp
		@mkdir -p $(dir $@)
		@echo Compile $<
		@g++ -c $< -o $@ $(cpp_compile_flags)

#定义cuda文件的编译方式
objs/%.cuo : src/%.cu
		@mkdir -p $(dir $@)
		@echo Compile $<
		@nvcc -c $< -o $@ $(cu_compile_flags)

#定义workspace/pro文件的编译
$(workspace)/$(binary) : $(cpp_objs) $(cu_objs)	
		@mkdir -p $(dir $@)
		@echo link $@
		@g++ $^ -o $@ $(link_flags)

#定义pro快捷编译指令，这里只发生编译，不执行
pro :$(workspace)/$(binary)	

#定义编译并执行的命令，并且执行目录切换到workspace下
run : pro
		@cd $(workspace) && ./$(binary)

debug :
		@echo $(cpp_objs)
		@echo $(cu_objs)

cleam :
		@rm -rf objs $(workspace)/$(binary)

# 指定伪标签，作为指令
.PYTHON : clean debug run pro				
SHELL=/bin/bash
.PHONY: build inline clean
inline: build
	(shopt -s nullglob; cd fast_g3; rm -f *.so; ln -s ../build/*.so ../build/*.dylib .)
build: build/build.ninja
	(cd build; meson compile)
build/build.ninja:
	rm -rf build
	mkdir build
	meson setup build --buildtype=release

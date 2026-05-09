先编译好NCCNN框架 ：git clone https://github.com/Tencent/ncnn.git
cd ncnn
source /usr/local/oecore-x86_64/environment-setup-riscv64-mchp-linux
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_BENCHMARK=OFF \
      -DNCNN_BUILD_TOOLS=OFF \
      -DNCNN_THREADS=ON ..
 make -j$(nproc)
编译执行文件：$CXX test_image_pic64.cpp -o test_image_pic64 -O3 -I/home/master/lnx1/pic64/deloy_/ncnn_pic64/src -I/home/master/lnx1/pic64/deloy_/ncnn_pic64/build/src /home/master/lnx1/pic64/deloy_/ncnn_pic64/build/src/libncnn.a -lpthread

路径看你们放的NCNN的位置更换

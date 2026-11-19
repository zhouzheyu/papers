# overview

## DiffKV

key insight:

* K的精度比V更高
* 不同token根据重要程度，精度不同
* 不同head精度不同

工程优化：

* 上述方法容易引起不规则内存占用，使用Unified Pages、Circular Free Page List、Bidirectional Page Table

## KVCodec

使用视频编解码器对KV cache进行压缩
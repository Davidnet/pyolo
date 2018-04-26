# PyYOLO (Kiwi Campus)
pyyolo (Kiwi campus) is a simple wrapper for YOLO. The original project is based at [PyYOLO by digitalbrain79](https://github.com/digitalbrain79/pyyolo)

# Warning
The makefile is modified so that is compiled against the architecture of the jetson and the compute capability of the jetson (6.2)
## Building Instructions
1. git clone --recursive https://github.com/thomaspark-pkj/pyyolo.git
2. (optional) Set GPU=1 and CUDNN=1 in Makefile to use GPU.
3. make
4. rm -rf build
5. python setup.py build (use setup_gpu.py for GPU)
6. sudo python setup.py install (use setup_gpu.py for GPU)

## Test
Run the examples.py script
```bash
python model_example.py
```
Requierements:
* Pillow
* cv2
* CUDA (compute capability = 6.2)
* CUDNN

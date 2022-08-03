<p align="center">
<img src="https://github.com/fivosts/clgen/blob/master/docs/logo-padded.png" width="800px" />
</p>

## Install

1. (Optional) If you think apt packages are missing from your system, install requirements.apt (works better if you have sudo access):
`bash requirements.apt`

2. Set up CMake workspace:
```
mkdir build; cd build
cmake ..
```
3. Install app:
```
make -j
```
4. Run the generated binary app:
```
cd ..
./clgen --help
```

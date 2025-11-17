# Build Instructions for P2P AllReduce

## System Requirements

### Hardware
- AMD GPU with ROCm support (e.g., MI250X, MI300X, or Radeon Instinct series)
- Multiple GPUs that support P2P (Peer-to-Peer) communication
- Recommended: GPUs connected via XGMI, Infinity Fabric, or high-speed PCIe

### Software
- **ROCm**: Version 5.0 or later
  - Download from: https://rocm.docs.amd.com/
  - Install HIP runtime and development tools
- **MPI**: Any MPI implementation
  - OpenMPI 4.0+ (recommended)
  - MPICH 3.3+
  - Or vendor-specific MPI (e.g., Cray MPI)
- **CMake**: Version 3.19 or later
- **C++ Compiler**: GCC 9+ or Clang 10+ with C++17 support

## Installation Steps

### 1. Install ROCm

#### Ubuntu/Debian
```bash
# Add ROCm repository
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
    
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/debian jammy main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install rocm-hip-sdk rocm-dev
```

#### RHEL/CentOS
```bash
# Add ROCm repository
sudo tee /etc/yum.repos.d/rocm.repo <<EOF
[rocm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/rhel8/
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

sudo yum install rocm-hip-sdk rocm-dev
```

### 2. Install MPI

#### Ubuntu/Debian
```bash
sudo apt install libopenmpi-dev openmpi-bin
```

#### RHEL/CentOS
```bash
sudo yum install openmpi openmpi-devel
# Add to PATH
module load mpi/openmpi-x86_64
```

### 3. Set Environment Variables

Add to your `~/.bashrc` or `~/.bash_profile`:

```bash
# ROCm paths
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# MPI paths (if needed)
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH
```

Then reload:
```bash
source ~/.bashrc
```

### 4. Verify Installation

```bash
# Check ROCm
rocm-smi
hipconfig

# Check MPI
mpirun --version

# Check GPU P2P capability
rocm-smi --showtopoinfo
```

### 5. Build the Project

```bash
cd p2p_allreduce
mkdir build && cd build

# Configure
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_PATH=/opt/rocm

# Build
make -j$(nproc)

# Optional: Install
sudo make install
```

## Build Options

You can customize the build with these CMake options:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \        # Release, Debug, or RelWithDebInfo
  -DROCM_PATH=/opt/rocm \              # Path to ROCm installation
  -DCMAKE_INSTALL_PREFIX=/usr/local   # Installation prefix
```

## Troubleshooting

### "HIP not found"
- Ensure ROCm is installed: `ls /opt/rocm`
- Set ROCM_PATH: `export ROCM_PATH=/opt/rocm`
- Add to CMake: `-DROCM_PATH=/opt/rocm`

### "MPI not found"
- Install MPI: `sudo apt install libopenmpi-dev`
- Or specify MPI location: `-DMPI_HOME=/path/to/mpi`

### "hipMalloc failed"
- Check GPU is visible: `rocm-smi`
- Set GPU: `export HIP_VISIBLE_DEVICES=0,1,2,3`
- Check permissions: User must be in `video` and `render` groups
  ```bash
  sudo usermod -a -G video,render $USER
  ```

### P2P Access Issues
- Verify P2P is enabled:
  ```bash
  # Check topology
  rocm-smi --showtopoinfo
  
  # Or use hipinfo
  /opt/rocm/bin/rocminfo | grep -A 5 "Link Type"
  ```
- Some systems require P2P to be enabled in BIOS
- PCIe switches may limit P2P capability

### Performance Issues
- Use GPUs with direct connections (XGMI/Infinity Fabric preferred over PCIe)
- Increase message size for better bandwidth utilization
- Check CPU pinning: `numactl --show`
- Use high-performance MPI transport (e.g., UCX with ROCm support)

## Running Tests

### Single Node (multiple GPUs)
```bash
# Run with 4 GPUs
mpirun -np 4 ./test_allreduce

# Run with specific GPUs
export HIP_VISIBLE_DEVICES=0,1,2,3
mpirun -np 4 ./test_allreduce
```

### Multi-Node

Create a hostfile (e.g., `hostfile.txt`):
```
node1 slots=4
node2 slots=4
```

Run:
```bash
mpirun -np 8 --hostfile hostfile.txt \
  -x HIP_VISIBLE_DEVICES=0,1,2,3 \
  ./test_allreduce
```

### With Slurm
```bash
srun -N 2 -n 8 --ntasks-per-node=4 \
  --gpus-per-node=4 \
  ./test_allreduce
```

## Docker Support (Optional)

You can also build and run in a Docker container:

```dockerfile
FROM rocm/dev-ubuntu-22.04:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY . /workspace/p2p_allreduce
WORKDIR /workspace/p2p_allreduce

# Build
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)
```

Build and run:
```bash
docker build -t p2p_allreduce .
docker run --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  p2p_allreduce ./build/test_allreduce
```

## Performance Tuning

### Environment Variables
```bash
# HIP settings
export HSA_FORCE_FINE_GRAIN_PCIE=1  # Enable fine-grained memory
export GPU_MAX_HW_QUEUES=4           # Increase command queues

# MPI settings (OpenMPI with UCX)
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export UCX_TLS=sm,self,rc_x,rocm_copy,rocm_ipc
```

### Algorithm Tuning
You can modify the threshold in `src/allreduce.cpp`:
```cpp
// Adjust this based on your hardware
static constexpr size_t SMALL_MSG_THRESHOLD = 32 * 1024;  // 32KB
```

## Verification

To verify correct installation and operation:

```bash
# 1. Check GPU topology
rocm-smi --showtopoinfo

# 2. Run simple test
mpirun -np 2 ./test_allreduce

# 3. Expected output
# Rank 0: Testing AllReduce SUM with 1024 elements
# Rank 1: Testing AllReduce SUM with 1024 elements
# Rank 0: AllReduce SUM test PASSED!
# Rank 1: AllReduce SUM test PASSED!
```

## Additional Resources

- ROCm Documentation: https://rocm.docs.amd.com/
- HIP Programming Guide: https://rocm.docs.amd.com/projects/HIP/
- MPI Tutorial: https://www.open-mpi.org/doc/
- Performance Analysis: https://rocm.docs.amd.com/projects/rocprofiler/

## Support

For issues specific to:
- **ROCm/HIP**: https://github.com/ROCm/ROCm/issues
- **MPI**: Check your MPI implementation's documentation
- **This project**: Open an issue in the repository

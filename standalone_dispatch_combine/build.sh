#!/bin/bash
# Build script for standalone dispatch-combine

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building Standalone Dispatch-Combine ===${NC}"

# Check for required tools
echo "Checking dependencies..."

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake not found. Please install CMake 3.16+${NC}"
    exit 1
fi

if ! command -v hipcc &> /dev/null; then
    echo -e "${RED}Error: hipcc not found. Please install ROCm/HIP${NC}"
    exit 1
fi

if ! command -v mpicc &> /dev/null; then
    echo -e "${RED}Error: mpicc not found. Please install MPI${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All dependencies found${NC}"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Removing existing build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=../install

# Build
echo "Building..."
make -j$(nproc)

echo -e "${GREEN}=== Build Complete ===${NC}"
echo ""
echo "To run the example:"
echo "  cd $BUILD_DIR"
echo "  mpirun -np 8 ./example"
echo ""
echo "To install:"
echo "  make install"

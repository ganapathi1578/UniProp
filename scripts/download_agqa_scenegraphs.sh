#!/bin/bash

# Exit on any error
set -e

echo "=========================================="
echo "Downloading AGQA Scene Graphs"
echo "=========================================="

# Define target directory relative to the script location
# Assumes script is run from project root or inside scripts/
if [ -d "scripts" ]; then
    PROJECT_ROOT="."
else
    PROJECT_ROOT=".."
fi

TARGET_DIR="${PROJECT_ROOT}/data/dataset/agqa_scene_graphs"
ZIP_FILE="${PROJECT_ROOT}/data/dataset/agqa_scene_graphs.zip"

# Create directories if they don't exist
mkdir -p "${TARGET_DIR}"

# Ensure gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing gdown..."
    pip install gdown
fi

# Download the file from Google Drive
echo "Downloading from Google Drive (this may take a while)..."
gdown "1CXU0tWpv-1kkkwkNzpU-BwQoAazPR1kR" -O "${ZIP_FILE}"

# Unzip the file
echo "Extracting the scene graphs..."
unzip -q -o "${ZIP_FILE}" -d "${TARGET_DIR}"

# Remove the zip file to save space
echo "Cleaning up..."
rm "${ZIP_FILE}"

# Flatten directory if the zip extracted into a subfolder with the same name
if [ -d "${TARGET_DIR}/AGQA_scene_graphs" ]; then
    echo "Flattening directory structure..."
    mv ${TARGET_DIR}/AGQA_scene_graphs/* ${TARGET_DIR}/
    rmdir "${TARGET_DIR}/AGQA_scene_graphs"
fi

echo "=========================================="
echo "Done! Scene graphs are ready at: ${TARGET_DIR}"
echo "=========================================="

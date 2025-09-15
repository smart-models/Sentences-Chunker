# GitHub Actions Docker Workflow Documentation

## Overview
This GitHub Actions workflow automatically builds and publishes Docker images for the Sentences-Chunker project to GitHub Container Registry (ghcr.io). It creates separate images for CPU and GPU variants with appropriate tagging.

## 📦 Published Images

The workflow publishes two variants of the Docker image:

- **CPU Image**: Optimized for CPU-only environments
- **GPU Image**: CUDA 12.1 enabled for GPU acceleration

## 🏷️ Image Tags

### For Main Branch
- `ghcr.io/{owner}/{repo}:latest-cpu` - Latest CPU build from main
- `ghcr.io/{owner}/{repo}:latest-gpu` - Latest GPU build from main

### For Version Tags (e.g., v1.2.3)
- `ghcr.io/{owner}/{repo}:v1.2.3-cpu` - Specific version for CPU
- `ghcr.io/{owner}/{repo}:v1.2.3-gpu` - Specific version for GPU

### For Pull Requests
- `ghcr.io/{owner}/{repo}:pr-{number}-cpu` - PR build for CPU
- `ghcr.io/{owner}/{repo}:pr-{number}-gpu` - PR build for GPU

## 🔄 Workflow Triggers

The workflow runs on:
1. **Push to main branch** - Creates `latest-*` tags
2. **Version tags** (v*.*.*) - Creates versioned releases
3. **Pull requests** - Creates PR-specific tags for testing
4. **Manual dispatch** - Can be triggered manually from GitHub UI

## 🚀 Usage

### Pull Images
```bash
# Latest CPU version
docker pull ghcr.io/{owner}/sentences-chunker:latest-cpu

# Latest GPU version
docker pull ghcr.io/{owner}/sentences-chunker:latest-gpu

# Specific version
docker pull ghcr.io/{owner}/sentences-chunker:v1.0.0-cpu
docker pull ghcr.io/{owner}/sentences-chunker:v1.0.0-gpu
```

### Create a Release
```bash
git tag v1.0.0
git push origin v1.0.0
```

## 🔧 Key Features

- **Matrix Build Strategy**: Parallel builds for CPU/GPU variants
- **Docker Buildx**: Advanced caching and multi-platform support
- **Security**: SBOM generation and build attestation
- **Optimization**: GitHub Actions cache for faster rebuilds
- **Verification**: Automatic image validation after push
- **Release Management**: Automatic GitHub Release creation for version tags

## 📝 Notes

- No additional secrets needed (uses GITHUB_TOKEN)
- Each variant has isolated cache scope
- Workflow can be manually triggered from GitHub UI
- PR builds allow testing before merge

For detailed documentation, see the comments in the workflow file.
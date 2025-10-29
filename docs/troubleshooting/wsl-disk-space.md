# WSL Disk Space Solutions

## Problem: Running Out of Disk Space

When building RAG indexes, you may encounter `OSError: [Errno 28] No space left on device` during model downloads. This guide provides solutions to free up space or move storage to another drive.

## Solution 1: Move Entire WSL Distribution to D: Drive (Recommended)

This moves your entire WSL installation (including all data) from `C:\Users\...\AppData\Local\Packages\...` to `D:\WSL\`.

### Steps:

1. **Export your WSL distribution** (open PowerShell as Administrator):
   ```powershell
   # List your distributions
   wsl --list --verbose
   
   # Export to D: drive (replace Ubuntu with your distro name)
   wsl --export Ubuntu D:\WSL\Ubuntu.tar
   ```

2. **Unregister the old distribution** (this removes it from C: drive):
   ```powershell
   wsl --unregister Ubuntu
   ```

3. **Import to new location on D: drive**:
   ```powershell
   wsl --import Ubuntu D:\WSL\Ubuntu D:\WSL\Ubuntu.tar
   ```

4. **Set default user** (if needed):
   ```powershell
   # Start WSL and edit config
   wsl -d Ubuntu
   sudo nano /etc/wsl.conf
   ```
   Add:
   ```ini
   [user]
   default=your_username
   ```
   Then restart:
   ```powershell
   wsl --terminate Ubuntu
   wsl -d Ubuntu
   ```

5. **Clean up** (optional - delete the .tar file after confirming everything works):
   ```powershell
   Remove-Item D:\WSL\Ubuntu.tar
   ```

## Solution 2: Move HuggingFace Cache to /mnt/d Within WSL

If you want to keep WSL on C: but move the model cache to D: drive:

### Steps:

1. **Mount D: drive in WSL** (if not already mounted):
   ```bash
   # D: drive should already be available at /mnt/d
   # Verify it exists:
   ls /mnt/d
   ```

2. **Create cache directory on D: drive**:
   ```bash
   mkdir -p /mnt/d/.cache/huggingface/hub
   ```

3. **Set HuggingFace cache environment variable** (persistent):
   
   Add to `~/.bashrc` or `~/.profile`:
   ```bash
   export HF_HOME=/mnt/d/.cache/huggingface
   export TRANSFORMERS_CACHE=/mnt/d/.cache/huggingface/hub
   export HF_HUB_CACHE=/mnt/d/.cache/huggingface/hub
   ```
   
   Then reload:
   ```bash
   source ~/.bashrc
   ```

4. **Move existing cache** (if you have one):
   ```bash
   # Find current cache location
   echo $HOME/.cache/huggingface
   
   # Move it if it exists
   if [ -d "$HOME/.cache/huggingface" ]; then
       mv $HOME/.cache/huggingface /mnt/d/.cache/
   fi
   ```

5. **Verify**:
   ```bash
   python -c "from huggingface_hub import cached_assets_path; print(cached_assets_path())"
   ```

## Solution 3: Move Project to /mnt/d

If your project is on C: drive, move it to D::

1. **Copy project to D: drive** (from Windows):
   ```powershell
   # From D:\Repo\cmw-rag, move to D:\Projects\cmw-rag
   Move-Item -Path D:\Repo\cmw-rag -Destination D:\Projects\cmw-rag
   ```

2. **Access from WSL**:
   ```bash
   cd /mnt/d/Projects/cmw-rag
   ```

3. **Update any absolute paths** in your configuration if needed.

## Solution 4: Clean Up Space

Before moving things around, try cleaning up:

### Within WSL:
```bash
# Clean package cache
sudo apt clean
sudo apt autoremove

# Clean HuggingFace cache (removes models not recently used)
python -c "
from huggingface_hub import scan_cache_dir
scan_cache_dir().delete_revisions([], min_size=1024**3*2)  # Remove files > 2GB
"

# Check disk usage
df -h
du -sh ~/.cache/*
```

### On Windows:
- Empty Recycle Bin
- Run Disk Cleanup
- Uninstall unused programs
- Clear browser cache

## Solution 5: Configure WSL to Use D: Drive by Default

You can configure WSL to store distributions on D: by default using `.wslconfig`:

1. **Create/edit `.wslconfig`** in `C:\Users\<YourUsername>\`:
   ```ini
   [wsl2]
   # Optional: Set memory limit
   memory=8GB
   
   # Optional: Set swap
   swap=4GB
   ```

2. **For new distributions**, import them directly to D::
   ```powershell
   wsl --import <DistroName> D:\WSL\<DistroName> <path-to-tar-file>
   ```

## Verification

After implementing any solution:

1. **Check available space**:
   ```bash
   df -h /mnt/d
   ```

2. **Test model download**:
   ```bash
   cd /mnt/d/Projects/cmw-rag  # or your project path
   source .venv-wsl/bin/activate
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('ai-forever/FRIDA')"
   ```

## Recommendation

- **If C: drive is nearly full**: Use Solution 1 (move entire WSL to D:)
- **If you just need more cache space**: Use Solution 2 (move HuggingFace cache to /mnt/d)
- **If project space is the issue**: Use Solution 3 (move project to /mnt/d)

Combining Solutions 1 and 2 gives you maximum flexibility and keeps everything organized on D: drive.


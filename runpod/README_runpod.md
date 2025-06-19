# 🚀 Running LLM Evaluation Framework on RunPod

This guide will help you set up and run the LLM Evaluation Framework on RunPod cloud computing platform, both manually and automatically.

## 📋 Prerequisites

Before starting, make sure you have the following files ready to transfer to your RunPod instance:

1. `Example_QA_data.xlsx` 📊
2. `env` (API keys file) 🔑
3. `requirements.txt` 📦

## 🤖 Automatic Pod Initialization (Recommended)

### Extra Dependencies for Automation

To use the automatic pod initialization, install these Python packages:

```bash
pip install runpod pyperclip python-dotenv
```

**Dependencies breakdown:**
- `runpod`: Official RunPod Python SDK for API interactions
- `pyperclip==1.9.0`: For clipboard operations (copying URLs/commands)
- `python-dotenv`: For loading environment variables from .env files

### Setup Environment Variables

Add your RunPod API key to your `env` file:

```bash
RUNPOD_API_KEY=your_runpod_api_key_here
```

> 💡 **Get your API key**: Visit [RunPod Settings](https://www.runpod.io/console/user/settings) → API Keys

### Using the Automation Script

The project includes the **[`runpod_initialize.ipynb`](runpod_initialize.ipynb)** Jupyter notebook for automatic pod creation and setup:

**Key automation features:**
- ✅ Automatic pod creation with specified GPU type
- ✅ SSH key setup and connection
- ✅ File transfer (project files, requirements, etc.)
- ✅ Virtual environment creation
- ✅ Dependency installation
- ✅ Jupyter kernel setup

### Configuration Options

When creating a pod automatically, you can customize:

```python
pod = runpod.create_pod(
    name="llm-eval-pod",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_type_id="NVIDIA A40",  # Options: "NVIDIA A40", "NVIDIA RTX A4500", "NVIDIA GeForce RTX 3080"
    cloud_type="COMMUNITY",    # Options: "ALL", "COMMUNITY", "SECURE"
    container_disk_in_gb=100,
    volume_in_gb=200,
    ports="8888/http,22/tcp",
    volume_mount_path="/workspace",
    start_jupyter=True
)
```

**GPU Type Selection:**
- Use `runpod.get_gpus()` to see all available GPU types
- Recommended: NVIDIA A40 for best performance/cost ratio

### Automatic Setup Process

The automation script will:

1. 🚀 **Create Pod**: Launch with specified configuration
2. ⏱️ **Wait**: 90 seconds for pod initialization
3. 🔗 **Connect**: Establish SSH connection
4. 📁 **Transfer**: Copy all project files
5. 🐍 **Environment**: Create virtual environment
6. 📦 **Install**: All dependencies including:
   - Core requirements from `requirements.txt`
   - `flash-attn==2.6.3`
   - Additional packages for specific use cases
7. 📓 **Jupyter**: Setup kernel for notebook access

**Estimated setup time:** ~16-20 minutes

## 🔐 SSH Setup (First Time Only)

### Generate SSH Key

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

### Add SSH Key to RunPod

1. Navigate to your SSH key file: `~/.ssh/id_ed25519.pub` (or `C:\Users\your-username\.ssh\id_ed25519.pub` on Windows)
2. Copy the public key content
3. Add it to your RunPod account at: https://www.runpod.io/console/user/settings

## 📤 Manual File Transfer to Pod

Use SCP to copy your files to the RunPod instance:

```bash
scp -P [PORT] -i ~/.ssh/id_ed25519 /path/to/your/requirements.txt root@[POD_IP]:/workspace/requirements.txt
```

> 💡 **Tip**: Replace `[PORT]` and `[POD_IP]` with the actual values from your RunPod dashboard.

## 🔌 Connect to Your Pod

Connect via SSH using the command provided in RunPod dashboard under "SSH over exposed TCP":

```bash
ssh root@[POD_IP] -p [PORT] -i ~/.ssh/id_ed25519
```

## 🐍 Environment Setup

### Create Virtual Environment

To persist your installations after pod restarts, create a virtual environment in the workspace:

```bash
python -m venv /workspace/myenv
source /workspace/myenv/bin/activate
cd /workspace
```

### Install Dependencies

```bash
# Install Jupyter kernel support
pip install --upgrade ipykernel
python -m ipykernel install --name myenv --user --display-name "Python (myenv)"

# Install project requirements
pip install -r requirements.txt && pip install flash-attn==2.6.3
```

### Advanced Dependencies for Specific Use Cases

**For RAG with Visual Support (ColPali):**
```bash
pip install cohere==5.15.0
pip install --upgrade byaldi
apt-get update
apt-get install -y poppler-utils
pip install -q pdf2image transformers==4.51.3 qwen-vl-utils
```

**For Smol Agents:**
```bash
pip install smolagents
pip install 'smolagents[e2b]'
pip install openpyxl==3.1.5
```

**For Code Execution Environment:**
```bash
# Create separate environment for code execution
python -m venv /workspace/test_LLM
source /workspace/test_LLM/bin/activate
pip install -r requirements_code_execution.txt
```

## 💻 VSCode Integration

### Jupyter Notebook Connection

1. Open VSCode locally
2. Create/open a Jupyter notebook
3. Click "Select Kernel" → "Existing Jupyter Server"
4. Paste your RunPod Jupyter URL (format: `https://[POD_ID]-8888.proxy.runpod.net/?token=[TOKEN]`)

## 📥 Download Results

To download your generated Excel files and graphs:

```bash
scp -P [PORT] -i ~/.ssh/id_ed25519 root@[POD_IP]:/workspace/*.{xlsx,png} /local/destination/path/
```

## 💾 Storage Management

### Check Disk Space

```bash
df -h
```

### ⚠️ Important Notes

- **Community Cloud Limitation**: Network volumes are only available in Secure Cloud, not Community Cloud
- **Resource Availability**: If you stop your pod, resources may not be available when you restart
- **Cost Optimization**: Consider terminating (not just stopping) your pod when done to save credits
- **Auto-cleanup**: The automation script handles environment setup, but manual cleanup may be needed

## 🔧 Pod Management via API

### List Your Pods
```python
import runpod
pods = runpod.get_pods()
print(pods)
```

### Stop/Terminate Pod
```python
# Stop pod (keeps data, costs storage)
runpod.stop_pod(pod_id)

# Terminate pod (destroys everything, saves costs)
runpod.terminate_pod(pod_id)
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

**Connection Issues:**
- Ensure your SSH key is properly added to RunPod settings
- Check if pod is fully initialized (wait 90+ seconds after creation)

**Package Installation:**
- Always activate your virtual environment before installing packages
- Use the correct requirements file for your use case

**Storage Issues:**
- Monitor disk usage regularly with `df -h`
- Clean up unnecessary files and virtual environments

**Automation Script Issues:**
- Verify API key is set correctly in environment
- Check SSH key path and permissions
- Ensure local files exist before transfer

**File Transfer Errors:**
- Use `-r` flag for recursive directory transfer
- Check file permissions and paths
- Some files may need individual transfer commands

## 🚀 Quick Start with Automation

1. **Install automation dependencies:**
   ```bash
   pip install runpod pyperclip python-dotenv
   ```

2. **Set up environment variables:**
   Add `RUNPOD_API_KEY=your_key` to your `env` file

3. **Run the automation notebook:**
   Open [`runpod_initialize.ipynb`](runpod_initialize.ipynb)

4. **Wait for completion:**
   ~16-20 minutes for full setup

5. **Start evaluating:**
   Connect via VSCode or SSH and run your evaluations!

## 📚 Useful Links

- [RunPod Python SDK](https://github.com/runpod/runpod-python) 🐍
- [RunPod SSH Documentation](https://docs.runpod.io/pods/configuration/use-ssh) 🔗
- [File Transfer Guide](https://docs.runpod.io/pods/storage/transfer-files) 📁
- [VSCode Connection Tutorial](https://blog.runpod.io/how-to-connect-vscode-to-runpod/) 💻
- [RunPod FAQ](https://docs.runpod.io/references/faq) ❓
- [RunPod API Reference](https://docs.runpod.io/references/api/manage-pods) 🔧

---

🎉 **You're all set!** Your LLM Evaluation Framework should now be ready to run on RunPod with either manual or automatic setup. Happy evaluating! 🚀
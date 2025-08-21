# BillSum Knowledge Distillation - Vast.ai Production Guide

## 🚀 Quick Start for Vast.ai

This guide will help you run the complete BillSum Knowledge Distillation experiment on Vast.ai with real PEFT training (no simulation).

### 📋 Prerequisites

1. **Vast.ai Account**: Sign up at [vast.ai](https://vast.ai)
2. **GPU Requirements**: 
   - Minimum: 16GB VRAM (RTX 4090, A4000, etc.)
   - Recommended: 24GB VRAM (RTX 6000 Ada, A6000, etc.)
3. **Hugging Face Token** (optional but recommended): Get from [hf.co/settings/tokens](https://huggingface.co/settings/tokens)

### 🖥️ Vast.ai Instance Setup

#### 1. Create Instance
```bash
# Search for suitable instances
vastai search offers 'cuda>=11.8 gpu_ram>=16 disk_space>=50'

# Rent instance (replace ID with actual offer ID)
vastai create instance <OFFER_ID> \
    --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel \
    --disk 50 \
    --ssh
```

#### 2. Instance Configuration
Choose instances with:
- **GPU**: RTX 4090, A4000, RTX 6000 Ada, A6000, or better
- **VRAM**: Minimum 16GB, preferably 24GB+
- **Storage**: At least 50GB
- **Image**: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel` or similar

### 🔧 Environment Setup

#### 1. Connect to Instance
```bash
# SSH into your instance
ssh -p <PORT> root@<IP_ADDRESS>
```

#### 2. Clone Repository
```bash
cd /workspace
git clone https://github.com/MothMalone/BillSum_Sel.git
cd BillSum_Sel
```

#### 3. Install Dependencies
```bash
# Update system
apt-get update && apt-get install -y git vim

# Install Python dependencies
pip install -r requirements.txt

# Install additional dependencies for production
pip install bitsandbytes accelerate
```

#### 4. Set Environment Variables (Optional)
```bash
# Set Hugging Face token if you have one
export HF_TOKEN="your_token_here"

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 🧪 Running the Experiment

#### Option 1: Full Production Experiment (Recommended)
```bash
# Run complete experiment with real PEFT training
python real_production_experiment.py
```

This will:
- ✅ Check GPU environment and memory
- ✅ Load BillSum dataset  
- ✅ Run 3 data selection methods (random, length-based, embedding-based)
- ✅ Perform real PEFT training with LoRA on each selected dataset
- ✅ Evaluate trained models with ROUGE metrics
- ✅ Generate comprehensive results comparison

#### Option 2: Quick Test (Memory-Optimized)
```bash
# Lightweight test version
python vast_production_experiment.py
```

### 📊 Expected Results

The experiment will compare three data selection methods using **BERTScore F1** and **ROUGE-Lsum** as the main evaluation metrics:

| Method | Description | Expected BERTScore F1 | Expected ROUGE-Lsum | Training Time |
|--------|-------------|----------------------|---------------------|---------------|
| Random | Baseline random sampling | ~0.82-0.85 | ~0.25-0.35 | ~5-10 min |
| Length-based | Optimal length filtering | ~0.84-0.87 | ~0.30-0.40 | ~5-10 min |
| Embedding-based | Semantic clustering | ~0.86-0.89 | ~0.35-0.45 | ~10-15 min |

**Main Metrics:**
- 🎯 **BERTScore F1**: Semantic similarity using BERT embeddings (primary metric)
- 🎯 **ROUGE-Lsum**: Longest common subsequence ROUGE for summarization (primary metric)

### 🔍 Monitoring Progress

The experiment provides detailed logging:
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check logs (in another terminal)
tail -f /path/to/logs
```

### 📁 Results

Results are saved to:
- `results/production/real_experiment_results.json` - Complete results
- `results/production/<method>/` - Individual model checkpoints
- `results/production/<method>/training_metadata.json` - Training details

### 🚨 Troubleshooting

#### Memory Issues
```bash
# Check available GPU memory
nvidia-smi

# If you get CUDA OOM errors:
# 1. Restart Python kernel
# 2. Reduce batch size in the script
# 3. Try the lightweight version instead
```

#### Dataset Loading Issues
```bash
# If dataset fails to load:
# 1. Check internet connection
# 2. Try without HF_TOKEN first
# 3. Verify dataset availability
```

#### Dependencies Issues
```bash
# If imports fail:
pip install --upgrade transformers datasets evaluate rouge-score bert-score
pip install bitsandbytes accelerate peft
```

### 💡 Performance Tips

1. **Use larger instances** for faster training (24GB+ VRAM)
2. **Set HF_TOKEN** for reliable dataset access
3. **Monitor GPU utilization** - should be 80-100% during training
4. **Use tmux/screen** to keep session alive if connection drops

### 🏁 Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 5-10 min | Environment prep, dependencies |
| Data Selection | 5-15 min | Embedding computation (longest) |
| Training | 15-30 min | 3 methods × 5-10 min each |
| Evaluation | 10-15 min | Model inference and metrics |
| **Total** | **35-70 min** | Complete experiment |

### 📈 Cost Estimation

Typical Vast.ai costs:
- **RTX 4090 (24GB)**: $0.20-0.40/hour
- **RTX A6000 (48GB)**: $0.30-0.60/hour  
- **Total experiment cost**: $0.20-0.70 USD

### 🎯 Success Criteria

The experiment is successful if:
- ✅ All 3 methods complete training without errors
- ✅ BERTScore F1 and ROUGE-Lsum scores show improvement over random baseline
- ✅ Embedding-based selection performs best on both main metrics
- ✅ Results are saved and properly formatted

### 📧 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the logs for specific error messages
3. Ensure your instance meets the minimum requirements
4. Try the lightweight version first if memory is limited

---

## 🔬 Technical Details

### Memory Optimization
- 4-bit quantization with BitsAndBytesConfig
- LoRA with rank 8 for parameter efficiency
- Gradient checkpointing to reduce memory
- Small batch sizes with gradient accumulation
- BF16 precision for memory and speed

### Model Configuration
- Base model: `microsoft/DialoGPT-medium` (memory efficient)
- LoRA rank: 8, alpha: 32
- Max sequence length: 1024 tokens
- Batch size: 1 with 8x gradient accumulation

### Data Selection Methods
1. **Random**: Baseline random sampling
2. **Length-based**: Filters for optimal length (200-800 words)
3. **Embedding-based**: Uses sentence transformers + k-means clustering

### Evaluation Metrics
- **BERTScore F1**: Semantic similarity using BERT embeddings (main metric)
- **ROUGE-Lsum**: Longest common subsequence ROUGE for summarization (main metric)
- ROUGE-1, ROUGE-2, ROUGE-L scores
- Real model inference (not simulation)
- Sample predictions for qualitative analysis

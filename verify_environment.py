#!/usr/bin/env python3
"""
BillSum Production Environment Verification Script
Checks if all dependencies and hardware requirements are met for Vast.ai
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_python_version():
    print_header("Python Environment")
    print(f"Python Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_gpu_availability():
    print_header("GPU Environment")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {memory_gb:.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                
                if memory_gb < 16:
                    print(f"  ⚠️  WARNING: Less than 16GB VRAM (has {memory_gb:.1f}GB)")
                else:
                    print(f"  ✅ Memory OK for production")
            
            return True
        else:
            print("❌ ERROR: No CUDA GPU detected")
            return False
            
    except ImportError:
        print("❌ ERROR: PyTorch not installed")
        return False

def check_required_packages():
    print_header("Package Dependencies")
    
    required_packages = [
        'transformers',
        'datasets', 
        'peft',
        'accelerate',
        'bitsandbytes',
        'sentence_transformers',
        'rouge_score',
        'bert_score',
        'sklearn',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'rouge_score':
                import rouge_score
                version = rouge_score.__version__
            elif package == 'bert_score':
                import bert_score
                version = bert_score.__version__
            elif package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✅ {package}: {version}")
            
        except ImportError:
            print(f"❌ {package}: NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages installed")
        return True

def check_memory_availability():
    print_header("System Memory")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"Used RAM: {(memory.total - memory.available) / (1024**3):.1f} GB")
        
        if memory.available < 8 * (1024**3):  # 8GB
            print("⚠️  WARNING: Less than 8GB available RAM")
        else:
            print("✅ RAM OK")
            
        return True
    except ImportError:
        print("❌ psutil not available, skipping memory check")
        return True

def check_disk_space():
    print_header("Disk Space")
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        
        print(f"Total Disk: {total / (1024**3):.1f} GB")
        print(f"Used Disk: {used / (1024**3):.1f} GB") 
        print(f"Free Disk: {free / (1024**3):.1f} GB")
        
        if free < 20 * (1024**3):  # 20GB
            print("⚠️  WARNING: Less than 20GB free disk space")
            print("  BillSum dataset + models require ~10-15GB")
        else:
            print("✅ Disk space OK")
            
        return True
    except Exception as e:
        print(f"❌ Error checking disk space: {e}")
        return False

def check_internet_connectivity():
    print_header("Internet Connectivity")
    try:
        import urllib.request
        
        # Test Hugging Face Hub connectivity
        test_urls = [
            ('Hugging Face Hub', 'https://huggingface.co'),
            ('PyTorch Hub', 'https://pytorch.org'),
        ]
        
        for name, url in test_urls:
            try:
                urllib.request.urlopen(url, timeout=10)
                print(f"✅ {name}: Connected")
            except Exception as e:
                print(f"❌ {name}: Failed ({e})")
                
        return True
    except Exception as e:
        print(f"❌ Error checking connectivity: {e}")
        return False

def check_huggingface_token():
    print_header("Hugging Face Authentication")
    
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')
    
    if token:
        print(f"✅ HF Token found: {token[:8]}...{token[-4:]}")
    else:
        print("⚠️  No HF token found (optional but recommended)")
        print("  Set HF_TOKEN environment variable for better access")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami(token=token)
        print(f"✅ Token valid for user: {user['name']}")
        return True
    except Exception as e:
        if token:
            print(f"❌ Token validation failed: {e}")
        else:
            print("ℹ️  Proceeding without authentication")
        return True

def run_quick_test():
    print_header("Quick Functionality Test")
    
    try:
        # Test data loading
        print("Testing dataset loading...")
        from datasets import load_dataset
        dataset = load_dataset("billsum", split="test[:5]")
        print(f"✅ Loaded {len(dataset)} samples from BillSum")
        
        # Test model loading
        print("Testing model loading...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        print("✅ Tokenizer loaded successfully")
        
        # Test GPU memory
        if torch.cuda.is_available():
            print("Testing GPU memory allocation...")
            device = torch.device('cuda')
            test_tensor = torch.randn(1000, 1000, device=device)
            print(f"✅ GPU memory test passed")
            del test_tensor
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    print("BillSum Production Environment Verification")
    print("==========================================")
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU Environment", check_gpu_availability), 
        ("Required Packages", check_required_packages),
        ("System Memory", check_memory_availability),
        ("Disk Space", check_disk_space),
        ("Internet Connectivity", check_internet_connectivity),
        ("HuggingFace Auth", check_huggingface_token),
        ("Quick Functionality", run_quick_test)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results[name] = False
    
    # Summary
    print_header("Verification Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Environment ready for production!")
        print("You can now run: python real_production_experiment.py")
    elif passed >= total - 2:
        print("\n⚠️  Environment mostly ready, but some issues detected")
        print("Review warnings above - experiment may still work")
    else:
        print("\n❌ Environment needs attention before running experiment")
        print("Please fix the failed checks above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

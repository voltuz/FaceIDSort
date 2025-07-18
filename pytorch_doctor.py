import torch
import sys

print("--- PyTorch GPU Diagnostic Script ---")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 35)

# Test 1: Is CUDA available at all according to PyTorch?
print("[Test 1] Checking torch.cuda.is_available()")
is_available = torch.cuda.is_available()
print(f"Result: {is_available}")

if not is_available:
    print("\n❌ CRITICAL: PyTorch reports that CUDA is NOT available.")
    print("This is the root cause. The following tests will likely fail.")
    print("This usually means one of the following:")
    print("  1. The installed PyTorch version is for CPU-only.")
    print("  2. The NVIDIA driver is not installed correctly or is not detected.")
    print("  3. There is a severe mismatch between the driver and the PyTorch CUDA version.")
else:
    print("\n✅ SUCCESS: PyTorch can see a CUDA-enabled device.")

print("-" * 35)

# Test 2: How many GPUs does PyTorch see?
print("[Test 2] Checking torch.cuda.device_count()")
try:
    device_count = torch.cuda.device_count()
    print(f"Result: Found {device_count} CUDA device(s).")
    if device_count == 0:
        print("❌ ERROR: CUDA is available, but no devices were found.")
    else:
        print("✅ SUCCESS: GPUs were enumerated.")
except Exception as e:
    print(f"❌ ERROR: An exception occurred while counting devices: {e}")

print("-" * 35)

# Test 3: Get the name of the primary GPU
print("[Test 3] Getting properties of device 0")
if is_available and device_count > 0:
    try:
        device_name = torch.cuda.get_device_name(0)
        print(f"Result: Device 0 is a '{device_name}'")
        print("✅ SUCCESS: Successfully queried GPU properties.")
    except Exception as e:
        print(f"❌ ERROR: An exception occurred while getting device properties: {e}")
else:
    print("Skipping test, as no CUDA devices are available.")
    
print("-" * 35)

# Test 4: The ULTIMATE TEST - Try to perform a simple operation on the GPU
print("[Test 4] Attempting a simple tensor operation on 'cuda:0'")
if is_available and device_count > 0:
    try:
        # Create a simple tensor on the CPU
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"  - Created a tensor on CPU: {cpu_tensor}")
        
        # Move the tensor to the GPU
        gpu_tensor = cpu_tensor.to('cuda:0')
        print(f"  - Successfully moved tensor to GPU: {gpu_tensor}")
        
        # Perform a simple operation
        result_tensor = gpu_tensor * 2
        print(f"  - Performed operation on GPU. Result: {result_tensor}")
        
        print("\n✅✅✅ ULTIMATE SUCCESS: All GPU operations are working correctly!")
        
    except Exception as e:
        print("\n❌❌❌ ULTIMATE FAILURE: A runtime error occurred during the GPU operation.")
        print("This is the specific error message we need:")
        import traceback
        traceback.print_exc()
else:
    print("Skipping test, as no CUDA devices are available.")

print("\n--- Diagnostic Complete ---")
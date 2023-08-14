from torch import cuda

def print_device_info():
    check_item = "Nvidia Cuda GPU available?"
    print(f"{check_item: <30}:{cuda.is_available()}")

    check_item = "GPU name"
    print(f"{check_item: <30}:{cuda.get_device_name()}")

    check_item = "Device properties"
    print(f"{check_item: <30}:{cuda.get_device_properties(cuda.current_device)}")

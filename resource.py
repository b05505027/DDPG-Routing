import psutil

def get_processes_memory():
    # Get list of all running processes
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
        try:
            # Append to the list of processes
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    # Sort the processes by memory usage
    processes.sort(key=lambda p: p['memory_percent'], reverse=True)

    return processes

if __name__ == "__main__":
    # Get processes sorted by memory usage
    processes = get_processes_memory()

    # Print the top processes
    print(f"{'PID':<10}{'Name':<25}{'Memory Usage (%)':>15}")
    for proc in processes[:10]:  # adjust number as needed
        print(f"{proc['pid']:<10}{proc['name']:<25}{proc['memory_percent']:>15.2f}")


from pymavlink import mavutil
import time

# Change this to your MAVProxy output, or direct COM port
CONNECTION_STRING = 'udp:127.0.0.1:14551'
# CONNECTION_STRING = 'COM5'
BAUD_RATE = 57600

print(f"Connecting to {CONNECTION_STRING}...")
try:
    master = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUD_RATE)
    print("Waiting for heartbeat...")
    
    # Wait for heartbeat
    msg = master.wait_heartbeat(timeout=10)
    if msg:
        print(f"Heartbeat from system (system {master.target_system} component {master.target_component})")
    else:
        print("Timeout waiting for heartbeat. Check connection, COM port, or MAVProxy.")
        exit(1)

    print("\nListening for messages for 10 seconds...")
    start_time = time.time()
    while time.time() - start_time < 10:
        # Wait for a message
        msg = master.recv_match(blocking=True, timeout=1.0)
        if not msg:
            continue
        
        msg_type = msg.get_type()
        
        # Only print specific messages so the terminal isn't flooded
        if msg_type in ['GLOBAL_POSITION_INT', 'ATTITUDE', 'HEARTBEAT']:
            print(f"[{msg_type}] {msg}")

except Exception as e:
    print(f"Error: {e}")

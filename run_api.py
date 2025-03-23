import uvicorn
import argparse
import socket
import os

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Create a socket connection to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't need to be reachable, just used to determine interface
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"  # Fallback to localhost if can't determine IP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sign Language Classification API")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host IP address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), 
                        help="Port number (default: 8000 or from PORT env variable)")
    parser.add_argument("--reload", action="store_true", 
                        help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Only try to get local IP in development environments (not needed in Docker)
    is_docker = os.environ.get('DOCKER_CONTAINER', False)
    local_ip = "container-ip" if is_docker else get_local_ip()
    
    print("\n===== Sign Language Classification API =====")
    print(f"Starting server on {args.host}:{args.port}")
    
    if not is_docker:
        print("\nAccess the API using one of the following URLs:")
        print(f" - Local access:     http://localhost:{args.port}")
        print(f" - On this machine:  http://{local_ip}:{args.port}")
        print(f" - API documentation: http://localhost:{args.port}/docs")
        print("\nNOTE: If you cannot access the API, check that:")
        print(" 1. Your firewall allows connections to this port")
        print(" 2. You're accessing via 'localhost' or your machine's actual IP (not 0.0.0.0)")
        print(" 3. If accessing from another device, ensure the server machine allows incoming connections")
    else:
        print("\nRunning in Docker container")
        print(f"API documentation available at: http://host-ip:{args.port}/docs")
    
    print("\nPress CTRL+C to stop the server")
    print("============================================\n")
    
    # Check if we're running in a package or directly
    try:
        # First try the module path that would be used in a normal installation
        uvicorn.run("src.Sign_Language_Classification.api.app:app", 
                    host=args.host, 
                    port=args.port, 
                    reload=args.reload)
    except ModuleNotFoundError:
        # If that fails, try direct module path which might work better in Docker
        uvicorn.run("Sign_Language_Classification.api.app:app", 
                    host=args.host, 
                    port=args.port, 
                    reload=args.reload)
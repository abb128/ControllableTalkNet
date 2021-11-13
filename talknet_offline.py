import socket
import sys

import os
from controllable_talknet import *


def handle_connection(connection, client_addr):
    data = connection.recv(4096)
    data = str(data, "utf-8")

    lines = data.split("\n")
    if(len(lines) < 4):
        print("Received an invalid request")
        return False
    

    (transcript, output_path, s_model, d_model, p_model, fast_mode) \
        = (lines[0], lines[1], lines[2], lines[3], lines[4], lines[5] == "True")

    (succ, err) = generate_audio(s_model, d_model, p_model, transcript, output_path, fast_mode)
    
    if(not succ):
        print("An error has occurred in audio generation:",err)
        return False
    
    return True


def run_server(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (host, port)
    sock.bind(server_address)

    sock.listen(1)
    while True:
        connection, client_addr = sock.accept()

        success = handle_connection(connection, client_addr)

        if(success):
            connection.sendall(b"Yes")
        else:
            connection.sendall(b"No")
        
        connection.close()


if __name__ == "__main__":
    if os.path.exists("/talknet/is_docker"):
        run_server("0.0.0.0", 8050)
    else:
        print("Unsupported platform")
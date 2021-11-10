import socket
import sys

import os
from controllable_talknet import *


def handle_connection(connection, client_addr):
    data = connection.recv(4096)
    data = str(data, "utf-8")

    lines = data.split("\n")
    if(len(lines) < 3):
        print("Received an invalid request")
        return False
    
    (transcript, output_path, custom_model) = (lines[0], lines[1], lines[2])

    ref_path = ""
    pitch_options = "dra"
    f0s = []
    f0s_wo_silence = []
    pitch_factor = 0.0

    if(len(data) == 4):
        ref_path = lines[3]
        pitch_options = ""
        if(not os.path.exists(ref_path)):
            print("Received a ref path that doesn't exist")
            return False
    
    (succ, err) = generate_audio(custom_model, transcript, pitch_options, pitch_factor,
        ref_path, output_path, f0s, f0s_wo_silence)
    
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
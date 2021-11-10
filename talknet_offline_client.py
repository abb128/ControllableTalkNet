import socket
import sys
import pathlib

import os

def send_request(transcript, out_path, custom_model, ref_path):
    request_str = str(transcript) + "\n" + str(out_path) + "\n" + str(custom_model)

    if(not(ref_path is None)):
        request_str = request_str + "\n" + str(ref_path)
    

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr = ("127.0.0.1", 8050)

    client.connect(addr)

    client.send(bytes(request_str, "utf-8"))

    resp = client.recv(1024)

    client.close()

    return resp == b"Yes"


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Request TalkNet server to synthesize speech")
    parser.add_argument("transcript", type=str, help="Text transcript")
    parser.add_argument("out_path", type=pathlib.Path, help="Output path, ending in .wav")
    parser.add_argument("custom_model", type=str, help="Google drive ID of the model to use")
    parser.add_argument("--ref", dest="ref_path", type=pathlib.Path, help="Reference audio path", required=False)
    
    args = parser.parse_args()
    success = send_request(args.transcript, args.out_path, args.custom_model, args.ref_path)

    if(success):
        sys.exit(0)
    else:
        sys.exit(1)
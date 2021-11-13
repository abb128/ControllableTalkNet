import socket
import sys
import pathlib

import os

def send_request(transcript, out_path, s_model, d_model, p_model, fast_mode):
    if((d_model is None) or (len(d_model) < 3)):
        d_model = s_model
    
    if((p_model is None) or (len(p_model) < 3)):
        p_model = d_model
    
    request_str = str(transcript) + "\n" \
                + str(out_path)   + "\n" \
                + str(s_model)    + "\n" \
                + str(d_model)    + "\n" \
                + str(p_model)    + "\n" \
                + str(fast_mode)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr = ("127.0.0.1", 8050)

    client.connect(addr)

    client.send(bytes(request_str, "utf-8"))

    resp = client.recv(4096)

    client.close()

    return resp == b"Yes"


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Request TalkNet server to synthesize speech")
    parser.add_argument("transcript", type=str, help="Text transcript")
    parser.add_argument("out_path", type=pathlib.Path, help="Output path, ending in .wav")
    parser.add_argument("custom_model", type=str, help="Google drive ID of the model to use")
    parser.add_argument("--d-model", dest="d_model", type=str, help="Optional Google drive ID for duration model", required=False)
    parser.add_argument("--p-model", dest="p_model", type=str, help="Optional Google drive ID for pitch model", required=False)
    parser.add_argument("--fast", dest="fast_mode", action="store_const", const=True, default=False, help="Run in fast mode (skips denoising, output may be noisier)")
    
    args = parser.parse_args()
    success = send_request(args.transcript, args.out_path, args.custom_model, args.d_model, args.p_model, args.fast_mode)

    if(success):
        sys.exit(0)
    else:
        sys.exit(1)
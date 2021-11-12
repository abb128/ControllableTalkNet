import sys
import os
import base64
import torch
import numpy as np
import tensorflow as tf
import crepe
import scipy
from scipy.io import wavfile
import psola
import io
import nemo
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.tts.models import TalkNetSpectModel
from nemo.collections.tts.models import TalkNetPitchModel
from nemo.collections.tts.models import TalkNetDursModel
from talknet_singer import TalkNetSingerModel
import json
from tqdm import tqdm
import gdown
import zipfile
import resampy
import traceback
import ffmpeg
import time
import uuid
import re

sys.path.append("hifi-gan")
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser


DEVICE = "cuda:0"
CPU_PITCH = False
RUN_PATH = os.path.dirname(os.path.realpath(__file__))

torch.set_grad_enabled(False)
if CPU_PITCH:
    tf.config.set_visible_devices([], "GPU")
DICT_PATH = os.path.join(RUN_PATH, "horsewords.clean")


playback_hide = None
playback_style = None




def load_hifigan(model_name, conf_name):
    # Load HiFi-GAN
    conf = os.path.join("hifi-gan", conf_name + ".json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device(DEVICE))
    state_dict_g = torch.load(model_name, map_location=torch.device(DEVICE))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser


def generate_json(input, outpath):
    output = ""
    sample_rate = 22050
    lpath = input.split("|")[0].strip()
    size = os.stat(lpath).st_size
    x = {
        "audio_filepath": lpath,
        "duration": size / (sample_rate * 2),
        "text": input.split("|")[1].strip(),
    }
    output += json.dumps(x) + "\n"
    with open(outpath, "w", encoding="utf8") as w:
        w.write(output)


asr_model = (
    EncDecCTCModel.from_pretrained(model_name="asr_talknet_aligner").cpu().eval()
)


def forward_extractor(tokens, log_probs, blank):
    """Computes states f and p."""
    n, m = len(tokens), log_probs.shape[0]
    # `f[s, t]` -- max sum of log probs for `s` first codes
    # with `t` first timesteps with ending in `tokens[s]`.
    f = np.empty((n + 1, m + 1), dtype=float)
    f.fill(-(10 ** 9))
    p = np.empty((n + 1, m + 1), dtype=int)
    f[0, 0] = 0.0  # Start
    for s in range(1, n + 1):
        c = tokens[s - 1]
        for t in range((s + 1) // 2, m + 1):
            f[s, t] = log_probs[t - 1, c]
            # Option #1: prev char is equal to current one.
            if s == 1 or c == blank or c == tokens[s - 3]:
                options = f[s : (s - 2 if s > 1 else None) : -1, t - 1]
            else:  # Is not equal to current one.
                options = f[s : (s - 3 if s > 2 else None) : -1, t - 1]
            f[s, t] += np.max(options)
            p[s, t] = np.argmax(options)
    return f, p


def backward_extractor(f, p):
    """Computes durs from f and p."""
    n, m = f.shape
    n -= 1
    m -= 1
    durs = np.zeros(n, dtype=int)
    if f[-1, -1] >= f[-2, -1]:
        s, t = n, m
    else:
        s, t = n - 1, m
    while s > 0:
        durs[s - 1] += 1
        s -= p[s, t]
        t -= 1
    assert durs.shape[0] == n
    assert np.sum(durs) == m
    assert np.all(durs[1::2] > 0)
    return durs


def preprocess_tokens(tokens, blank):
    new_tokens = [blank]
    for c in tokens:
        new_tokens.extend([c, blank])
    tokens = new_tokens
    return tokens


parser = (
    nemo.collections.asr.data.audio_to_text.AudioToCharWithDursF0Dataset.make_vocab(
        notation="phonemes",
        punct=True,
        spaces=True,
        stresses=False,
        add_blank_at="last",
    )
)

arpadict = None


def load_dictionary(dict_path):
    arpadict = dict()
    with open(dict_path, "r", encoding="utf8") as f:
        for l in f.readlines():
            word = l.split("  ")
            assert len(word) == 2
            arpadict[word[0].strip().upper()] = word[1].strip()
    return arpadict


def replace_words(input, dictionary):
    regex = re.findall(r"[\w'-]+|[^\w'-]", input)
    assert input == "".join(regex)
    for i in range(len(regex)):
        word = regex[i].upper()
        if word in dictionary.keys():
            regex[i] = "{" + dictionary[word] + "}"
    return "".join(regex)


def arpa_parse(input, model):
    global arpadict
    if arpadict is None:
        arpadict = load_dictionary(DICT_PATH)
    z = []
    space = parser.labels.index(" ")
    input = replace_words(input, arpadict)
    input = input.replace("\n", " \n")
    input = input.replace(".", ". ")
    while "{" in input:
        if "}" not in input:
            input.replace("{", "")
        else:
            pre = input[: input.find("{")]
            if pre.strip() != "":
                x = model.parse(text=pre.strip())
                seq_ids = x.squeeze(0).cpu().detach().numpy()
                z.extend(seq_ids)
            z.append(space)

            arpaword = input[input.find("{") + 1 : input.find("}")]
            arpaword = (
                arpaword.replace("0", "")
                .replace("1", "")
                .replace("2", "")
                .strip()
                .split(" ")
            )

            seq_ids = []
            for x in arpaword:
                if x == "":
                    continue
                if x.replace("_", " ") not in parser.labels:
                    continue
                seq_ids.append(parser.labels.index(x.replace("_", " ")))
            seq_ids.append(space)
            z.extend(seq_ids)
            input = input[input.find("}") + 1 :]
    if input != "":
        x = model.parse(text=input.strip())
        seq_ids = x.squeeze(0).cpu().detach().numpy()
        z.extend(seq_ids)
    if z[-1] == space:
        z = z[:-1]
    if z[0] == space:
        z = z[1:]
    return [
        z[i] for i in range(len(z)) if (i == 0) or (z[i] != z[i - 1]) or (z[i] != space)
    ]


def to_arpa(input):
    arpa = ""
    z = []
    space = parser.labels.index(" ")
    while space in input:
        z.append(input[: input.index(space)])
        input = input[input.index(space) + 1 :]
    z.append(input)
    for y in z:
        if len(y) == 0:
            continue

        arpaword = " {"
        for s in y:
            if parser.labels[s] == " ":
                arpaword += "_ "
            else:
                arpaword += parser.labels[s] + " "
        arpaword += "} "
        if not arpaword.replace("{", "").replace("}", "").replace(" ", "").isalnum():
            arpaword = arpaword.replace("{", "").replace(" }", "")
        arpa += arpaword
    return arpa.replace("  ", " ").replace(" }", "}").strip()


def get_duration(wav_name, transcript, tokens):
    if not os.path.exists(os.path.join(RUN_PATH, "temp")):
        os.mkdir(os.path.join(RUN_PATH, "temp"))
    if "_" not in transcript:
        generate_json(
            os.path.join(RUN_PATH, "temp", wav_name + "_conv.wav")
            + "|"
            + transcript.strip(),
            os.path.join(RUN_PATH, "temp", wav_name + ".json"),
        )
    else:
        generate_json(
            os.path.join(RUN_PATH, "temp", wav_name + "_conv.wav") + "|" + "dummy",
            os.path.join(RUN_PATH, "temp", wav_name + ".json"),
        )

    data_config = {
        "manifest_filepath": os.path.join(RUN_PATH, "temp", wav_name + ".json"),
        "sample_rate": 22050,
        "batch_size": 1,
    }

    dataset = nemo.collections.asr.data.audio_to_text._AudioTextDataset(
        manifest_filepath=data_config["manifest_filepath"],
        sample_rate=data_config["sample_rate"],
        parser=parser,
    )

    dl = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    blank_id = asr_model.decoder.num_classes_with_blank - 1

    for sample_idx, test_sample in tqdm(enumerate(dl), total=len(dl)):
        log_probs, _, greedy_predictions = asr_model(
            input_signal=test_sample[0], input_signal_length=test_sample[1]
        )

        log_probs = log_probs[0].cpu().detach().numpy()
        target_tokens = preprocess_tokens(tokens, blank_id)

        f, p = forward_extractor(target_tokens, log_probs, blank_id)
        durs = backward_extractor(f, p)

        del test_sample
        return durs
    return None


def crepe_f0(wav_path, hop_length=256):
    # sr, audio = wavfile.read(io.BytesIO(wav_data))
    sr, audio = wavfile.read(wav_path)
    audio_x = np.arange(0, len(audio)) / 22050.0
    f0time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

    x = np.arange(0, len(audio), hop_length) / 22050.0
    freq_interp = np.interp(x, f0time, frequency)
    conf_interp = np.interp(x, f0time, confidence)
    audio_interp = np.interp(x, audio_x, np.absolute(audio)) / 32768.0
    weights = [0.5, 0.25, 0.25]
    audio_smooth = np.convolve(audio_interp, np.array(weights)[::-1], "same")

    conf_threshold = 0.25
    audio_threshold = 0.0005
    for i in range(len(freq_interp)):
        if conf_interp[i] < conf_threshold:
            freq_interp[i] = 0.0
        if audio_smooth[i] < audio_threshold:
            freq_interp[i] = 0.0

    # Hack to make f0 and mel lengths equal
    if len(audio) % hop_length == 0:
        freq_interp = np.pad(freq_interp, pad_width=[0, 1])
    return (
        torch.from_numpy(freq_interp.astype(np.float32)),
        torch.from_numpy(frequency.astype(np.float32)),
    )


def f0_to_audio(f0s):
    volume = 0.2
    sr = 22050
    freq = 440.0
    base_audio = (
        np.sin(2 * np.pi * np.arange(256.0 * len(f0s)) * freq / sr) * volume
    ).astype(np.float32)
    shifted_audio = psola.vocode(base_audio, sr, target_pitch=f0s)
    for i in range(len(f0s)):
        if f0s[i] == 0.0:
            shifted_audio[i * 256 : (i + 1) * 256] = 0.0
    print(type(shifted_audio[0]))
    buffer = io.BytesIO()
    wavfile.write(buffer, sr, shifted_audio.astype(np.float32))
    b64 = base64.b64encode(buffer.getvalue())
    sound = "data:audio/x-wav;base64," + b64.decode("ascii")
    return sound



def select_file(dropdown_value):
    if dropdown_value is not None:
        if not os.path.exists(os.path.join(RUN_PATH, "temp")):
            os.mkdir(os.path.join(RUN_PATH, "temp"))
        ffmpeg.input(os.path.join(RUN_PATH, dropdown_value)).output(
            os.path.join(RUN_PATH, "temp", dropdown_value + "_conv.wav"),
            ar="22050",
            ac="1",
            acodec="pcm_s16le",
            map_metadata="-1",
            fflags="+bitexact",
        ).overwrite_output().run(quiet=True)
        fo_with_silence, f0_wo_silence = crepe_f0(
            os.path.join(RUN_PATH, "temp", dropdown_value + "_conv.wav")
        )
        return [
            "Analyzed " + dropdown_value,
            fo_with_silence,
            f0_wo_silence,
            dropdown_value,
        ]
    else:
        return ["No audio analyzed", None, None]



def debug_pitch(n_clicks, pitch_clicks, current_f0s):
    if not n_clicks or current_f0s is None or n_clicks <= pitch_clicks:
        if n_clicks is not None:
            pitch_clicks = n_clicks
        else:
            pitch_clicks = 0
        return [
            None,
            playback_hide,
            pitch_clicks,
        ]
    pitch_clicks = n_clicks
    return [f0_to_audio(current_f0s), playback_style, pitch_clicks]


hifigan_sr = None


def download_model(model, custom_model):
    try:
        global hifigan_sr, h2, denoiser_sr
        d = "https://drive.google.com/uc?id="
        if model == "Custom":
            drive_id = custom_model
        else:
            drive_id = model
        if drive_id == "" or drive_id is None:
            return ("Missing Drive ID", None, None)
        if not os.path.exists(os.path.join(RUN_PATH, "models")):
            os.mkdir(os.path.join(RUN_PATH, "models"))
        if not os.path.exists(os.path.join(RUN_PATH, "models", drive_id)):
            os.mkdir(os.path.join(RUN_PATH, "models", drive_id))
            zip_path = os.path.join(RUN_PATH, "models", drive_id, "model.zip")
            gdown.download(
                d + drive_id,
                zip_path,
                quiet=False,
            )
            if not os.path.exists(zip_path):
                os.rmdir(os.path.join(RUN_PATH, "models", drive_id))
                return ("Model download failed", None, None)
            if os.stat(zip_path).st_size < 16:
                os.remove(zip_path)
                os.rmdir(os.path.join(RUN_PATH, "models", drive_id))
                return ("Model zip is empty", None, None)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(RUN_PATH, "models", drive_id))
            os.remove(zip_path)

        # Download super-resolution HiFi-GAN
        sr_path = "hifi-gan/hifisr"
        if not os.path.exists(sr_path):
            gdown.download(
                d + "14fOprFAIlCQkVRxsfInhEPG0n-xN4QOa", sr_path, quiet=False
            )
        if not os.path.exists(sr_path):
            raise Exception("HiFI-GAN model failed to download!")
        if hifigan_sr is None:
            hifigan_sr, h2, denoiser_sr = load_hifigan(sr_path, "config_32k")

        return (
            None,
            os.path.join(RUN_PATH, "models", drive_id, "TalkNetSpect.nemo"),
            os.path.join(RUN_PATH, "models", drive_id, "hifiganmodel"),
        )
    except Exception as e:
        return (str(e), None, None)



model_cache = []
def get_cached(spect, d, p):
    global model_cache
    for a in model_cache:
        if(a[0] == spect) and (a[1] == d) and (a[2] == p):
            return a[3]
    
    return None

def cache(spect, d, p, val):
    global model_cache
    while(len(model_cache) >= 4):
        model_cache.pop()
    
    model_cache.append([spect, d, p, val])


hifigan_cache = []
def get_hifigan_cached(hifigan_path):
    global hifigan_cache
    for a in hifigan_cache:
        if(a[0] == hifigan_path):
            return a[1]
    
    while(len(hifigan_cache) >= 4):
        hifigan_cache.pop()
    
    hifigan, h, denoiser = load_hifigan(hifigan_path, "config_v1")

    hifigan_cache.push([hifigan_path, (hifigan, h, denoiser)])

    return (hifigan, h, denoiser)


def generate_audio(
    custom_model_spect,
    custom_model_dur,
    custom_model_pitch,
    transcript,
    out_wav
):
    global model_cache, hifigan_cache

    if transcript is None or transcript.strip() == "":
        return (False, "No transcript entered")
    
    load_error, talknet_path, hifigan_path = download_model("Custom", custom_model_spect)
    if load_error is not None:
        return (False, load_error)
    
    load_error, talknet_path_dur, _ = download_model("Custom", custom_model_dur)
    if load_error is not None:
        return (False, load_error)

    load_error, talknet_path_pitch, _ = download_model("Custom", custom_model_pitch)
    if load_error is not None:
        return (False, load_error)

    try:
        with torch.no_grad():
            tnmodel = get_cached(custom_model_spect, custom_model_dur, custom_model_pitch)
            hifigan, h, denoiser = get_hifigan_cached(hifigan_path)

            if(tnmodel is None):
                singer_path = os.path.join(
                    os.path.dirname(talknet_path), "TalkNetSinger.nemo"
                )
                if os.path.exists(singer_path):
                    tnmodel = TalkNetSingerModel.restore_from(singer_path)
                else:
                    tnmodel = TalkNetSpectModel.restore_from(talknet_path)
                durs_path = os.path.join(
                    os.path.dirname(talknet_path_dur), "TalkNetDurs.nemo"
                )
                pitch_path = os.path.join(
                    os.path.dirname(talknet_path_pitch), "TalkNetPitch.nemo"
                )

                if os.path.exists(durs_path):
                    tndurs = TalkNetDursModel.restore_from(durs_path)
                    tnmodel.add_module("_durs_model", tndurs)
                else:
                    return (False, "Model doesn't support duration prediction")
                
                if os.path.exists(pitch_path)
                    tnpitch = TalkNetPitchModel.restore_from(pitch_path)
                    tnmodel.add_module("_pitch_model", tnpitch)
                else:
                    return (False, "Model doesn't support pitch prediction")
                
                tnmodel.eval()


                cache(custom_model_spect, custom_model_dur, custom_model_pitch, tnmodel)
            
            token_list = arpa_parse(transcript, tnmodel)
            tokens = torch.IntTensor(token_list).view(1, -1).to(DEVICE)
            arpa = to_arpa(token_list)

            if tndurs is None or tnpitch is None:
                return (False, "Model doesn't support pitch prediction")
            
            spect = tnmodel.generate_spectrogram(tokens=tokens)

            y_g_hat = hifigan(spect.float())
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE

            # does this sound better or worse when denoised?
            audio_denoised = denoiser(audio.view(1, -1), strength=30)[:, 0]
            audio_np = (
                audio_denoised.detach().cpu().numpy().reshape(-1).astype(np.int16)
            )

            # Resample to 32k
            wave = resampy.resample(
                audio_np,
                h.sampling_rate,
                h2.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            # HiFi-GAN super-resolution
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device(DEVICE))
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                h2.n_fft,
                h2.num_mels,
                h2.sampling_rate,
                h2.hop_size,
                h2.win_size,
                h2.fmin,
                h2.fmax,
            )
            y_g_hat2 = hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze()
            audio2 = audio2 * MAX_WAV_VALUE
            audio2_denoised = denoiser(audio2.view(1, -1), strength=24)[:, 0]

            # High-pass filter, mixing and denormalizing
            audio2_denoised = audio2_denoised.detach().cpu().numpy().reshape(-1)
            b = scipy.signal.firwin(
                101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
            )
            y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
            y *= 4.0  # superres strength
            y_out = y.astype(np.int16)
            y_padded = np.zeros(wave_out.shape)
            y_padded[: y_out.shape[0]] = y_out
            sr_mix = wave_out + y_padded

            buffer = open(out_wav, "wb")
            wavfile.write(buffer, 32000, sr_mix.astype(np.int16))
            buffer.close()

            return (True, arpa)
    except Exception:
        return (False, str(traceback.format_exc()))
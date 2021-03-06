import os
import ffmpeg
import numpy as np
import torch
from io import BytesIO
import cv2
import scipy.io.wavfile as wav
import tempfile
import torch.nn as nn


def assert_arr_continuous(arr):

    return sorted(arr) == list(range(min(arr), max(arr) + 1))


def find_closest_match(labels, len_box):
    for i in range(1, len(labels)):
        if len(labels[-i]) == len_box:
            # print(i)
            return labels[-i]

    raise ValueError


def initialization(weights, type="xavier", init=None):
    if type == "normal":
        if init is None:
            torch.nn.init.normal_(weights)
        else:
            torch.nn.init.normal_(weights, mean=init[0], std=init[1])
    elif type == "xavier":
        if init is None:
            torch.nn.init.xavier_normal_(weights)
        else:
            torch.nn.init.xavier_normal_(weights, gain=init)
    elif type == "kaiming":
        torch.nn.init.kaiming_normal_(weights)
    elif type == "orthogonal":
        if init is None:
            torch.nn.init.orthogonal_(weights)
        else:
            torch.nn.init.orthogonal_(weights, gain=init)
    else:
        raise NotImplementedError("Unknown initialization method")


def initialize_weights(
    net, type="xavier", init=None, init_bias=False, batchnorm_shift=None
):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            initialization(m.weight, type=type, init=init)
            if init_bias and hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm") != -1 and batchnorm_shift is not None:
            torch.nn.init.normal_(m.weight, 1.0, batchnorm_shift)
            torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
            for layer_params in m._all_weights:
                for param in layer_params:
                    if "weight" in param:
                        initialization(m._parameters[param])

    net.apply(init_func)


def swp_extension(file, ext):
    return os.path.splitext(file)[0] + ext


def save_audio(path, audio, audio_rate=16000):
    if torch.is_tensor(audio):
        aud = audio.squeeze().detach().cpu().numpy()
    else:
        aud = audio.copy()  # Make a copy so that we don't alter the object

    aud = ((2 ** 15) * aud).astype(np.int16)
    wav.write(path, audio_rate, aud)


def save_video(
    path,
    video,
    fps=25,
    scale=1,
    audio=None,
    audio_rate=16000,
    overlay_pts=None,
    ffmpeg_experimental=False,
):
    success = True
    out_size = (scale * video.shape[2], scale * video.shape[1])
    video_path = get_temp_path(ext=".mp4")
    if torch.is_tensor(video):
        vid = video.squeeze().detach().cpu().numpy()
    else:
        vid = video.copy()  # Make a copy so that we don't alter the object

    if np.min(vid) < 0:
        vid = 127 * vid + 127
    elif np.max(vid) <= 1:
        vid = 255 * vid

    is_color = True
    if vid.ndim == 3:
        is_color = False

    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), out_size, isColor=is_color
    )
    for i, frame in enumerate(vid):
        if is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if scale != 1:
            frame = cv2.resize(frame, out_size)

        write_frame = frame.astype("uint8")

        if overlay_pts is not None:
            for pt in overlay_pts[i]:
                cv2.circle(
                    write_frame, (int(scale * pt[0]), int(scale * pt[1])), 2, (0, 0, 0), -1
                )

        writer.write(write_frame)
    writer.release()

    inputs = [ffmpeg.input(video_path)["v"]]

    if audio is not None:  # Save the audio file
        audio_path = swp_extension(video_path, ".wav")
        save_audio(audio_path, audio, audio_rate)
        inputs += [ffmpeg.input(audio_path)["a"]]

    try:
        if ffmpeg_experimental:
            out = ffmpeg.output(
                *inputs, path, strict="-2", loglevel="panic", vcodec="h264"
            ).overwrite_output()
        else:
            out = ffmpeg.output(
                *inputs, path, loglevel="panic", vcodec="h264"
            ).overwrite_output()
        out.run(quiet=True)
    except ValueError:
        success = False

    if audio is not None and os.path.isfile(audio_path):
        os.remove(audio_path)
    if os.path.isfile(video_path):
        os.remove(video_path)

    return success


def video_to_stream(video, audio=None, fps=25, audio_rate=16000):
    temp_file = get_temp_path(ext=".mp4")
    save_video(temp_file, video, audio=audio, fps=fps, audio_rate=audio_rate)
    stream = BytesIO(open(temp_file, "rb").read())

    if os.path.isfile(temp_file):
        os.remove(temp_file)

    return stream


def get_temp_path(ext=""):
    file_path = next(tempfile._get_candidate_names()) + ext
    if os.path.exists("/tmp"):  # If tmp exists then prepend to the path
        file_path = "/tmp/" + file_path

    return file_path


def load_state_dict_flexible(model, state_dict):
    model_state_dict = model.state_dict()
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(
                    f"Skip loading parameter: {k}, "
                    f"required shape: {model_state_dict[k].shape}, "
                    f"loaded shape: {state_dict[k].shape}"
                )
                state_dict[k] = model_state_dict[k]
        else:
            print(f"Dropping parameter {k}")
    model.load_state_dict(state_dict, strict=False)

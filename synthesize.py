import os
from model.models import ForwardTransformer
from data.audio import Audio
from vocoding.predictors import HiFiGANPredictor
from argparse import ArgumentParser


def synthesize(text):
    if not text:
        return None
    dir_path = os.path.dirname(os.path.realpath(__file__))

    model_path = os.path.join(dir_path, "tts_model")
    vocoder_path = os.path.join(dir_path, "vocoding", "hifigan", "custom")

    wav_no_vc_path = os.path.join(dir_path, "output", "wav_no_vc.wav")
    wave_vc_path = os.path.join(dir_path, "output", "wav_vc.wav")

    model = ForwardTransformer.load_model(model_path)
    vocoder = HiFiGANPredictor.from_folder(vocoder_path)
    audio = Audio.from_config(model.config)

    out = model.predict(text, speed_regulator=1.)

    wav_vc = vocoder([out['mel'].numpy().T])[0]

    audio.save_wav(wav_vc, wav_path=wave_vc_path)
    return wave_vc_path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--text', '-t', dest='text', default=None, type=str)
    args = parser.parse_args()
    if args.text is None:
        print(" ################ ")
        print("ُError: no text entered")
        print("################")
        exit(0)
    synthesize(args.text)

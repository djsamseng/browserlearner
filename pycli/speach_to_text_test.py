import os

import pocketsphinx
import pyaudio

CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 44100

KWS_LIST_PATH = "/home/samuel/dev/browserlearner/pycli/keyphrase.list"
DIC_PATH = "/home/samuel/dev/browserlearner/pycli/cmudict-en-us.dict"

def main():
    for phrase in pocketsphinx.LiveSpeech(dic=DIC_PATH):
        print(phrase, phrase.score())
        if phrase.score() > -1600:
            print("Success")


def keyphrase():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        # sudo killall pulseaudio
        input_device_index=0,
        frames_per_buffer=CHUNK,
    )
    stream.start_stream()

    model_dir = pocketsphinx.get_model_path()
    config = pocketsphinx.Decoder.default_config()
    config.set_string("-hmm", os.path.join(model_dir, "en-us"))
    #config.set_string('-dict', os.path.join(model_dir, 'en-us/cmudict-en-us.dict'))
    config.set_string("-kws", "/home/samuel/dev/browserlearner/pycli/keyphrase.list")

    decoder = pocketsphinx.Decoder(config)
    decoder.start_utt()
    while True:
        buf = stream.read(CHUNK)
        if buf:
            decoder.process_raw(buf, False, False)
        else:
            print("No audio")
            break
        if decoder.hyp() != None:
            print ([(seg.word, seg.prob, seg.start_frame, seg.end_frame) for seg in decoder.seg()])
            print ("Detected keyphrase, restarting search")
            decoder.end_utt()
            decoder.start_utt()

if __name__ == "__main__":
    #keyphrase()
    main()
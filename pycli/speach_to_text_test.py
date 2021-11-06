
import pocketsphinx

def main():
    for phrase in pocketsphinx.LiveSpeech():
        print(phrase)

if __name__ == "__main__":
    main()
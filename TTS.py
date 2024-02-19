from gtts import gTTS

LABEL_PATH = "Model/labels.txt"
folder = open(LABEL_PATH, 'r')
label = folder.read().splitlines()
print(label)
FOLDER = "Speech"
for i in range(len(label)):
    text = label[i]
    print(f"\n Converting {label[i]}...")
    tts = gTTS(text, slow=False, lang='en-in')
    tts.save(f'{FOLDER}/{text}.mp3')
    print("\n Success!")
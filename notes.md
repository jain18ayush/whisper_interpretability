**afplay [audio file]** is a command line tool to play audio files on Mac OS X. It is similar to the play command in Linux.

**ffmpeg -i input.m4a output.wav** is a command line tool to convert audio files from one format to another. It is a powerful tool that can handle a wide range of audio and video formats.

last file that we stopped at: data/es_dev_0/common_voice_es_19706723.mp3

```bash
for file in speech/cv-corpus-18.0-delta-2024-06-14/fr/clips/*.mp3; do
  ffmpeg -i "$file" "speech/cv-corpus-18.0-delta-2024-06-14/fr/clips/$(basename "${file%.*}").wav"
done
```

# Ideas
- Can we mess with the decoder instead of the encoder?
- Need to figure out how to use the existing architecture to get data.

# TODO
- [ ] Document the process so far

# Process
- installed sox from brew [brew install sox]

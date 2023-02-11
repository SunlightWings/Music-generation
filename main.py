import librosa
import librosa.display 
import numpy as np
import matplotlib.pyplot as plt
# to handle mp3 files, we install pydub library, on windows add ffmpeg, else librosa prefers .wav format
audio_data, sr = librosa.load( 'C:\\Users\\prabin\\OneDrive\\Desktop\\minor project\\chopin.wav' )
# print('hello')

# mfccs = librosa.feature.mfcc(y = audio_data, sr=sr)

# ###to convert into array
# mfcc_array = np.array(mfccs)
# # print(mfcc_array)

# #to display the audio features
# librosa.display.specshow(mfccs, sr=sr, x_axis='time')
# plt.colorbar()
# plt.savefig('mfcc_plot.png')
# plt.show()


# Convert the frequency axis from samples to Hz
stft = librosa.stft(audio_data)
frequencies = librosa.fft_frequencies(sr=100, n_fft=len(stft))



#print(frequencies)

for i in range(1,len(frequencies)):

     print(librosa.hz_to_note(frequencies[i]))

import matplotlib.pyplot as plt

import librosa

# Load the audio file

y, sr = librosa.load("./c_major.ogg")



# Calculate the frame duration in samples

frame_duration = 30  # in ms

frame_samples = int((frame_duration / 1000) * sr)



# frameplt =[]

# Iterate over the frames



for i in range(10, len(y), frame_samples):

    frame = y[i:i + frame_samples]

    # frameplt.append(frame)




    # Compute the short-time Fourier transform (STFT) of the frames

    stft = librosa.stft(frame)



    # Convert the STFT to frequency and magnitude spectrograms

    frequency, magnitude = librosa.magphase(stft)

    # Convert the frequency axis from samples to Hz

    frequencies = librosa.fft_frequencies( sr=sr, n_fft=len(stft))

# midi_file = 'example.mid'
# midi_data = librosa.midi.read_midi(midi_file)

# # Extract notes from the MIDI file
# notes = librosa.midi_to_notes(midi_data)
# print(notes)


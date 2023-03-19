import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import wave

def convert_to_wav(input_file):
    if os.path.isfile(input_file):
        new_file = input_file.replace('.mov', '.wav')
        command = f'ffmpeg -i {input_file} -vn -acodec pcm_s16le -ar 44100 -ac 2 {new_file} >/dev/null 2>&1'
        os.system(command)
        os.remove(input_file)
        
        return new_file
    
def wav_to_spectrograzzm(input_file):
    y, sr = librosa.load(input_file)

    # Generate a spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to decibels
    spec_db = librosa.power_to_db(spec, ref=np.max)

    return spec_db, sr

def detect_lung(input_file):
    spectrogram, sr = wav_to_spectrograzzm(input_file)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel', sr=sr, cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig('spec.png', bbox_inches='tight')
    print('saved')


    num_frames, num_frequencies = spectrogram.shape

    threshold = -30

    above_threshold = np.where(spectrogram > threshold)

    start_wheezing = above_threshold[0][0]
    end_wheezing = above_threshold[0][-1]

    start_breathing = end_wheezing + 1
    while start_breathing < num_frames and spectrogram[start_breathing, :].max() > threshold:
        start_breathing += 1

    next_wheezing = start_breathing
    while next_wheezing < num_frames and spectrogram[next_wheezing, :].max() <= threshold:
        next_wheezing += 1

    gap_duration = next_wheezing - start_breathing


    with wave.open('a.wav', 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        duration = num_frames / frame_rate

    duration_in_ms = (gap_duration / 44100*duration) * 1000


    spectrogram = [item for sublist in spectrogram for item in sublist]

    max_val = np.max(spectrogram)
    
    if max_val > 0:
        return True
    else:
        return False

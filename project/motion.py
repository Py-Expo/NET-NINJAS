from ctypes import _NamedFuncPointer
from logging import _nameToLevel
from flask import Flask, render_template, send_file, request, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import librosa
from moviepy.editor import VideoFileClip
import pygame
import os

app = Flask(_NamedFuncPointer)

def play_alert_sound():
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound('alert_sound.wav')  # Change 'alert_sound.wav' to the path of your sound file
    alert_sound.play()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/motion_amplification', methods=['POST'])
def motion_amplification():
    video_path = r'D:\project1\static\2.mp4'  # Adjust the path to your video file
    if not os.path.exists(video_path):
        return jsonify({'result': 'Error: Video file not found.'})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'result': 'Error: Unable to open video file.'})

    try:
        video = VideoFileClip(video_path)
        audio_path = r'D:\project1\static\audio_extracted.wav'  # Adjust the path to save the audio file
        video.audio.write_audiofile(audio_path)
        print("Audio extracted and saved to:", audio_path)

        y, sr = librosa.load(audio_path)
        video = VideoFileClip(video_path)
        audio = video.audio
        sr = audio.fps
        duration = librosa.get_duration(y=y, sr=sr)
        t = np.linspace(0, duration, len(y))

        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(t, y)
        plt.title('Time Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Magnitude')

        n_fft = int(sr*0.025)
        fft_result = np.fft.fft(y, n=n_fft)
        magnitude = np.abs(fft_result)
        freq = np.fft.fftfreq(n_fft, 1 / sr)
        plt.subplot(2, 1, 2)
        plt.plot(freq[:n_fft // 2], magnitude[:n_fft // 2])
        plt.title('FFT Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, sr / 2)

        threshold_frequency = 20000
        if any(freq > threshold_frequency):
            play_alert_sound()

        plt.tight_layout()
        spectrum_path = 'static/spectrum.png'
        plt.savefig(spectrum_path)
        plt.close()

        return jsonify({'result': 'success', 'spectrum_path': spectrum_path})

    finally:
        cap.release()
        video.close()


if _nameToLevel == '_main_':
    app.run(debug=True)
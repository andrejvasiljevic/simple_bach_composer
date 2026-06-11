"""
Module: ssm_func
Description: Utility functions for converting array grids to MIDI, synthesizing 
audio via FluidSynth, and managing the end-to-end rendering pipeline.
"""
import os
import random
from typing import List

import numpy as np
import pretty_midi
from scipy.io import wavfile
from music21 import stream, note, instrument

# Audio & MIDI Constants
DEFAULT_SAMPLE_RATE = 44100
MAX_MIDI_VELOCITY = 127
# BASE_VELOCITY = 90


def piano_roll_to_midi(song: np.ndarray, out_file: str = "output.mid", step: float = 0.25) -> str:
    """
    Converts a structured NumPy array back into a standard multi-track MIDI file.
    
    Args:
        song (np.ndarray): 2D array representing the song (time_steps, 4 parts).
        out_file (str): Desired output path for the MIDI file.
        step (float): Duration of a single time step in quarter-lengths.
        
    Returns:
        str: The file path to the saved MIDI file.
    """
    print(f"Converting piano roll to MIDI: {out_file}...")
    s = stream.Stream()
 
    # Default Classical String Quartet Layout
    instruments = [
        instrument.Violin(),          # Voice 0: Melody / Soprano
        instrument.Flute(),           # Voice 1: Wind Instrument counter-melody (or Oboe/Clarinet)
        instrument.Piano(),           # Voice 2: Piano Right Hand (Harmony / Accompaniment)
        instrument.Piano()            # Voice 3: Piano Left Hand (Bass line / Roots)
    ]

    velocity_profiles = [95, 80, 60, 75]

    for i, inst in enumerate(instruments):
        BASE_VELOCITY = velocity_profiles[i]

        part = stream.Part()
        part.insert(0, inst)
        t = 0
        has_notes = False

        while t < len(song):
            pitch = song[t, i]

            if pitch != 0:
                has_notes = True
                start = t
                
                # Determine note sustain duration
                while t < len(song) and song[t, i] == pitch:
                    t += 1
                
                dur = (t - start) * step
                n = note.Note(int(pitch))
                n.quarterLength = dur
                
                # Humanization: Random velocity swing
                human_velocity = BASE_VELOCITY + random.randint(-15, 15)
                
                # Boost the Bass voice slightly to anchor the harmony
                if i == 3: 
                    human_velocity = min(MAX_MIDI_VELOCITY, human_velocity + 10)
                    
                n.volume.velocity = human_velocity
                part.insert(start * step, n) 
            else:
                t += 1

        if has_notes:
            s.insert(0, part)

    s.write("midi", fp=out_file)
    return out_file


def convert_midi_to_wav(midi_path: str, output_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bool:
    """
    Synthesizes a MIDI file into a WAV audio file using the default system soundfont.
    Includes a safety limiter to prevent clipping and noise floor amplification.
    """
    print(f"Synthesizing audio: '{midi_path}' -> '{output_path}'...")

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        audio_data = midi_data.fluidsynth(fs=sample_rate) 

        # Safe Normalization Limiter
        max_peak = np.max(np.abs(audio_data))
        
        if max_peak < 0.001:
            print("Warning: Synthesizer output is almost silent. Verify MIDI notes.")
        elif max_peak > 0:
            audio_data = audio_data / max_peak 
            
        audio_data = np.int16(audio_data * 32767)

        wavfile.write(output_path, sample_rate, audio_data)
        print(f"Success! Playable file ready at: {output_path}")
        return True

    except Exception as e:
        print(f"Synthesis error: {e}")
        return False


def pipeline_array_to_audio(song_array: np.ndarray, final_wav_name: str = "final_output.wav"):
    """ Master function running the end-to-end Array -> MIDI -> WAV pipeline. """
    print("--- Starting Audio Generation Pipeline ---")
    
    temp_midi_file = "temp_render.mid"
    
    # 1. Array -> MIDI
    piano_roll_to_midi(song=song_array, out_file=temp_midi_file)
    
    # 2. MIDI -> Audio
    success = convert_midi_to_wav(midi_path=temp_midi_file, output_path=final_wav_name)
    
    # 3. Cleanup
    if success and os.path.exists(temp_midi_file):
        os.remove(temp_midi_file)
        
    print("--- Pipeline Complete ---")
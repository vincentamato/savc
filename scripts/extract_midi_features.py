import os
import sys
import csv
from tqdm import tqdm
from music21 import converter, instrument, key, tempo, meter, chord, note
from multiprocessing import Pool

def extract_features_from_midi(midi_path):
    # Parse the MIDI file
    try:
        score = converter.parse(midi_path)
    except Exception:
        return None

    # Key detection
    try:
        ks = score.analyze('key')
        key_tonic = str(ks.tonic)
        key_mode = ks.mode
    except:
        key_tonic = "None"
        key_mode = "None"

    # Tempo
    tempos = score.metronomeMarkBoundaries()
    tempo_values = [t[2].number for t in tempos if t[2] and t[2].number]
    avg_tempo = sum(tempo_values)/len(tempo_values) if tempo_values else 0.0

    # Notes and pitches
    pitch_values = []
    num_notes = 0
    for el in score.recurse():
        if isinstance(el, note.Note):
            num_notes += 1
            pitch_values.append(el.pitch.midi)
        elif isinstance(el, chord.Chord):
            num_notes += len(el.notes)
            pitch_values.extend([n.pitch.midi for n in el.notes])
    avg_pitch = sum(pitch_values)/len(pitch_values) if pitch_values else 0.0

    # Instrumentation
    instruments = score.getInstruments(returnDefault=False)
    instrument_names = [str(i.instrumentName) for i in instruments if i.instrumentName]
    instrument_count = len(instrument_names)

    # Time signature
    ts = score.recurse().getElementsByClass(meter.TimeSignature)
    time_sig = ts[0].ratioString if len(ts) > 0 else "None"

    features = {
        "path": midi_path,
        "key_tonic": key_tonic,
        "key_mode": key_mode,
        "avg_tempo": avg_tempo,
        "num_notes": num_notes,
        "avg_pitch": avg_pitch,
        "instrument_count": instrument_count,
        "time_signature": time_sig
    }

    return features

def process_midi_directory(midi_dir, output_csv="midi_features.csv", num_workers=32, save_batch_size=100):
    # Collect all .mid files
    midi_files = []
    for root, dirs, files in os.walk(midi_dir):
        for fname in files:
            if fname.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, fname))

    if len(midi_files) == 0:
        print(f"No MIDI files found in {midi_dir}")
        sys.exit(1)

    fieldnames = ["path", "key_tonic", "key_mode", "avg_tempo", "num_notes", "avg_pitch", "instrument_count", "time_signature"]

    # Write the header once at the start
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    results = []

    with Pool(processes=num_workers) as p:
        # imap returns an iterator, which we wrap with tqdm for a progress bar
        for i, res in enumerate(tqdm(p.imap(extract_features_from_midi, midi_files),
                                     total=len(midi_files),
                                     desc="Processing MIDIs",
                                     unit="file")):
            if res is not None:
                results.append(res)

            # Save in batches of 100
            if save_batch_size and len(results) >= save_batch_size:
                with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    for r in results:
                        writer.writerow(r)
                # Clear results after writing
                results = []

    # Write any remaining results to CSV
    if len(results) > 0:
        with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for r in results:
                writer.writerow(r)

    print(f"Feature extraction complete. Results saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_midi_features.py <midi_directory> [output_csv]")
        sys.exit(1)

    midi_dir = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "midi_features.csv"

    # Set num_workers to 32 for maximum parallelism
    # save_batch_size=100 to save every 100 processed MIDIs
    process_midi_directory(midi_dir, out_csv, num_workers=32, save_batch_size=100)

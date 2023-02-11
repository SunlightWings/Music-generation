import pretty_midi

# Create a PrettyMIDI object
midi = pretty_midi.PrettyMIDI()

# Create an instrument instance
piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))

# Define the notes of the C Major scale
notes = [pretty_midi.Note(velocity=100, pitch=i, start=0, end=0.5) for i in [60, 62, 64, 65, 67, 69, 71, 72]]

# Add the notes to the instrument
piano.notes = notes

# Add the instrument to the PrettyMIDI object
midi.instruments.append(piano)

# Write the PrettyMIDI object to a MIDI file
midi.write('c_major_scale.mid')

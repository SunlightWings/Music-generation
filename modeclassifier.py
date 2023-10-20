from music21 import *
import scipy
from collections import Counter


# enumeration of all the notes in a given mode, from 0 to 11 for 12 pitches in an octave

Chromatic = [0,1,2,3,4,5,6,7,8,9,10,11]
Ionian = [0,2,4,5,7,9,11]
Dorian = [0,2,3,5,7,9,10]
Phrygian = [0,1,3,5,7,8,10]
Lydian = [0,2,4,6,7,9,11]
Mixolydian = [0,2,4,5,7,9,10]
Aeolian = [0,2,3,5,7,8,10]
Locrian = [0,1,3,5,6,8,10]

#variations
Harmonic_minor = [0,2,3,5,7,8,11]
Melodic_minor = [0,2,3,5,7,9,11]

#Eastern classical modes
Bhairav = [0,1,4,5,7,8,11]
Poorvi = [0,1,4,6,7,8,11]
Marva = [0,1,4,6,7,9,11]
Todi = [0,2,3,6,7,8,11]



modeList = [Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian, Harmonic_minor, Melodic_minor, Bhairav,Poorvi, Marva, Todi]


# modetuples has modenames and modes
# modename will be used later to name the given mode
# mode will be used later to compute hamming distance

modeTuples = [ ("Ionian" , Ionian),
("Dorian" , Dorian),
("Phrygian", Phrygian),
("Lydian" , Lydian), 
("Mixolydian" , Mixolydian),
("Aeolian" , Aeolian),
("Locrian" , Locrian),
("Harmonic_minor", Harmonic_minor),
("Melodic_minor_ascend", Melodic_minor),
("Bhairav",Bhairav),
("Poorvi", Poorvi),
("Marva", Marva),
("Todi", Todi),
]


# notes referenced to C, initialized to zero
# notes will be used later to transpose other keys to the reference key
# the numbers will be used when rootnote is returned (at rootnote class later)
midinumbers = {
    'c' : 0,
    'C' : 0,
    'c#' : 1,
    'C#' : 1,
    'd-' : 1,
    'D-' : 1,
    'd' : 2,
    'D' :2,
    'd#' : 3,
    'D#' : 3,
    'e-' : 3,
    'E-' : 3,
    'e' : 4,
    'E' : 4,
    'f' : 5,
    'F' : 5,
    'f#' : 6,
    'F#' : 6,
    'g-' :6,
    'G-': 6,
    'g' : 7,
    'G' : 7,
    'g#' : 8,
    'G#' :8,
    'a-' : 8,
    'A-' :8,
    'a' : 9,
    'A' : 9,
    'a#' : 10,
    'A#' : 10,
    'b-' : 10,
    'B-' : 10,
    'b' : 11,
    'B' : 11,

}



# has arguments filepath and the condition for removing drums:true or false
def open_midi(midi_path, remove_drums):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    # list tracks
    print ((mf.tracks))
    print(mf)
    if (remove_drums):
        for i in range(len(mf.tracks)):
           #remove drum tracks
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]          
    
    # converted to stream
    return midi.translate.midiFileToStream(mf)



# returns name of rootnote and enumerated value
# enum will be returned as midinumbers written in the above cell
class Rootnote:
    def __init__(self, noteName):
        self.noteName = noteName

   

    def __str__(self):
        return f"{self.noteName}"
    
    def asnum(self):
        return midinumbers[self.noteName]
    



    # takes base_midi (which has midi filename) as argument
def extract_notes(midi_part):
    parent_element = []
    ret = []
    for nt in midi_part.flat.notes:        
        if isinstance(nt, note.Note):
            ret.append(max(0.0, nt.pitch.ps))
            parent_element.append(nt)
        elif isinstance(nt, chord.Chord):
            for pitch in nt.pitches:
                ret.append(max(0.0, pitch.ps))
                parent_element.append(nt)
    
    # returns appended pitch as ret
    # returns pitches of chords as parent_element
    return ret, parent_element





# takes filepath as an argument
def getMode(midi_filename):
    base_midi = open_midi(midi_filename, True)

    # gives key and scale as output. key is required, scale not required
    music_analysis = base_midi.analyze('key')

    # key stored as rootnote while scale is discarded
    rootnote = Rootnote(format(music_analysis).split(' ')[0])
    print("the root note is ", rootnote)
    # calls extract_notes function has appended pitch returned as ret in a; b has pitches of chords but its not used
    a, b = extract_notes(base_midi)
    
    # each elements in 'a' (has pitches) is transposed/ treated as base note, starting from 0.
    transposed = [x - rootnote.asnum() for x in a]
    # each octave is transposed to -1 octave having midi numbers from 0 to 11.
    refOctave = [x % 12 for x in transposed]

    # frequency and their count is kept, organized/sorted as below:
    freq = Counter(refOctave)
    freq = dict (freq) 
    sorted_notes= sorted(freq.items(), key=lambda x:x[1], reverse=True)

    print(sorted_notes)
    
    # mostusednotes list has notes (counts in descending order), only notes with 7 highest counts taken as below: 
    mostusednotes = []
    for i,j in sorted_notes:
        mostusednotes.append(i)

    mostusednotes = mostusednotes[:7]
    # mostusednotes in ascending order (notes themselves in that order)
    mostusednotes.sort()
    print("most used notes : ", mostusednotes)
    lowest = 999
    # hamming distance calculation between mostusednotes and 'mode' from modetuple written above
    # if distance is lowest, that mode is selected and returned
    for modeName, mode in modeTuples:
        distance = scipy.spatial.distance.hamming(mostusednotes, mode)
        if distance < lowest:
            lowest = distance 
            ourmode = modeName

    return rootnote, ourmode






# filepath is passed to above getMode function.
ourmode = getMode("C:/Users/prabin/OneDrive/Desktop/Minor final/After-Mid-Term/BleedingMe.mid")




# the required mode obtained. The index [1] means only string (modename) is printed 
print(ourmode[1])
import random, glob
import numpy as np
import pandas as pd 
from music21 import *
import music21
import pickle


music_files =[a for a in glob.glob("*/dataset/*/*/*")]
print("A random song", random.sample(music_files, 1))

a = pd.read_csv('songs_with_root_and_mode.csv')
l =[]
for i in range(len(a)):
    if(a.iloc[i, 4]=='Ionian'):
        l.append((a.iloc[i, 2], a.iloc[i, 3]))


def get_score(music_files):
    chords = []
    root_notes = []
    l = len(music_files)
    i=0
    for filename, root_note in music_files:
        i=i+1
        try:
            chords.append(converter.parse(filename))
            root_notes.append(root_note)
            print(f'Happening of {i}, remaining {l-i}', "filename = ", filename)
            
        except:
            print(f'Happening of {i}, remaining {l-i}', "filename = ", filename)
            print("file failed!!!!!")
            continue
        
    
    return chords, root_notes



score_of_all_musics, root_notes_of_all_musics = get_score(l)

print(len(score_of_all_musics), len(root_notes_of_all_musics))



def transpose_2(rootnote, noteWithOctave):
    mapping_dict = {}
    all_note_list0 = ["C","C#","D","D#","E","F","F#", "G","G#", "A","A#", "B" ]
    all_note_list1 = ["C","D-","D","E-","E","F","G-", "G","A-", "A","B-", "B" ]
    
    rootnote = rootnote.upper()
    if rootnote in all_note_list0:
        temp = 1
        root_index = all_note_list0.index(rootnote)
        transpose_distance = root_index-1
    elif rootnote in all_note_list1:
        temp=0
        root_index = all_note_list1.index(rootnote)
        transpose_distance = root_index-1
    
    assigning_index = 1
    for each_octave in range(0 , 9): 
        for each_note in range (0, len(all_note_list0)):
            #print(each_note)
            
            temp_dict0 = { all_note_list0[each_note] + str(each_octave) : assigning_index }
            temp_dict1 = { all_note_list1[each_note] + str(each_octave) : assigning_index }
            mapping_dict.update(temp_dict0)
            mapping_dict.update(temp_dict1)
            assigning_index += 1

    mapping_dict.update({"<SOC>"  : assigning_index})
    mapping_dict.update({"<EOC>"  : assigning_index + 1})
    note_index = mapping_dict.get(noteWithOctave)
    #print(f'{rootnote}\t{noteWithOctave}')
    transposed_index = note_index-transpose_distance
    if(transposed_index<1):
        return 'C0'
    else:
        result = [key for key, value in mapping_dict.items() if value == transposed_index]
        return result[0]


mapping_dict = {}
all_note_list0 = ["C","C#","D","D#","E","F","F#", "G","G#", "A","A#", "B" ]
all_note_list1 = ["C","D-","D","E-","E","F","G-", "G","A-", "A","B-", "B" ]
assigning_index = 1
for each_octave in range(0 , 9):
    for each_note in range (0, len(all_note_list0)):
        temp_dict0 = { all_note_list0[each_note] + str(each_octave) : assigning_index}
        temp_dict1 = { all_note_list1[each_note] + str(each_octave) : assigning_index}
        mapping_dict.update(temp_dict0)
        mapping_dict.update(temp_dict1)
        assigning_index += 1
mapping_dict.update({"<SOC>": assigning_index})
mapping_dict.update({"<EOC>": assigning_index + 1})
mapping_dict.update({'0': assigning_index + 2})
mapping_dict.update({'0.1': assigning_index + 3})
mapping_dict.update({'0.3': assigning_index + 4})
mapping_dict.update({'1.25': assigning_index + 5})
mapping_dict.update({'2': assigning_index + 6})


def round_chord_durations(number):
    if(number>1.25):
        return 2
    if(number>.30):
        return 1.25
    if(number>.10):
        return 0.3
    return 0.1


def round_tempo(number):
    if(number>=20 and number<40):
        return 30
    if(number>=40 and number<60):
        return 50
    if(number>=60 and number<76):
        return 70
    if(number>=76 and number<108):
        return 92
    if(number>=108 and number<120):
        return 104
    if(number>=120 and number<168):
        return 144
    if(number>=168 and number<200):
        return 184
    if(number>=200):
        return 210

def play_midi_file(midi_file_name):
    mf = midi.MidiFile()

    mf.open(midi_file_name) # path='abc.midi'
    mf.read()
    mf.close()
    s = midi.translate.midiFileToStream(mf)
    s.show('midi')




def get_chord_and_duration_data_in_C(individual_score, root_note):
    no_of_tracks = len(individual_score.parts)
    flags=[0,0,0,0]
    chord_duration_data = []
    note_and_chord_sequence =[]
    duration_sequence = []
    note_and_chord_duration =[]   ## not for now:
    all_tempo= []
    
    #to see the number of tracks:
    #print(len(individual_score.parts))
    
    for element in individual_score.flat:
        #print("element = ", element, type(element))
        
        
        if isinstance(element, chord.Chord):
            flags[0] = flags[0] + 1
            if (flags[0] == 1):
                note_and_chord_sequence.append('<SOC>') ## Start of Chord
                for pitch in element.pitches:
                    transposed_pitch = transpose_2(root_note, pitch.nameWithOctave)
                    #print("transposed_pitch ", transposed_pitch, type(transposed_pitch))
                    note_and_chord_sequence.append(transposed_pitch)
                note_and_chord_sequence.append('<EOC>') ## End of Chord

                chord_duration =str(round_chord_durations(element.duration.quarterLength))
                #print (type(element.duration.quarterLength))
                note_and_chord_sequence.append(chord_duration)
                #print(((pitch.nameWithOctave) for pitch in element.pitches), chord_duration)
                #duration_sequence.append(chord_duration)
                for x in range (0, len(element.pitches) + 1):
                    duration_sequence.append('0')
            elif (flags[0] == no_of_tracks):
                flags[0] =0
            
            
            
            
        elif isinstance(element, note.Note):
            flags[1] = flags[1] + 1
            if (flags[1] == 1):
                for pitch in element.pitches:
                    transposed_pitch = transpose_2(root_note, pitch.nameWithOctave)
                    note_and_chord_sequence.append(transposed_pitch)
                    #print("note")
                note_duration = str(round_chord_durations(element.duration.quarterLength))
                note_and_chord_sequence.append(note_duration)
                #print(((pitch.nameWithOctave) for pitch in element.pitches), note_duration)
                #duration_sequence.append(note_duration)
            elif (flags[1] == no_of_tracks):
                flags[1] =0
        
        
        elif isinstance(element, note.Rest):
            rest_note_name = element.name
            #print(rest_note_name)
            
            
        elif isinstance(element, tempo.MetronomeMark):
            flags[2] = flags[2] + 1
            if (flags[2] == 1):
                tempo_bpm = element.getQuarterBPM()
                #note_and_chord_sequence.append(str(round_tempo(tempo_bpm)))
                #all_tempo.append(tempo_bpm)
                #print(tempo_bpm)
            elif (flags[2] == no_of_tracks):
                flags[2] =0
       
        elif isinstance(element,meter.TimeSignature):
            flags[3] = flags[3]+1
            if(flags[3]==1):
                current_time_signature = element.ratioString
                #note_and_chord_sequence.append('TS')
                #note_and_chord_sequence.append(current_time_signature)
            elif(flags[3] == no_of_tracks):
                flags[3]= 0
            
    
    #print("tempo ko lagi = " ,np.quantile(all_tempo, .25), np.quantile(all_tempo, .50), np.quantile(all_tempo, .75) )
    #print("tempo ko lagi max min = " ,np.max(all_tempo) , np.min(all_tempo)  )
    #print(note_and_chord_sequence)
    return note_and_chord_sequence


print("\nGetting chords and notes\n")

chords_and_duration_data_all_music_in_C= []
for i in range(len(score_of_all_musics)):
    note_and_chord_sequence = get_chord_and_duration_data_in_C(score_of_all_musics[i], root_notes_of_all_musics[i])
    chords_and_duration_data_all_music_in_C.append((note_and_chord_sequence))
    

main_dataset = chords_and_duration_data_all_music_in_C



no_of_timesteps = 60
x = []
y = []

## CD stands for chord and duration.

for each_music_with_CD in main_dataset:
    for each_CD in range(0, len(each_music_with_CD) - no_of_timesteps,  1):
        
        ## preparing input and output sequences:
        input_ = each_music_with_CD[each_CD:each_CD + no_of_timesteps]
        output = each_music_with_CD[each_CD + no_of_timesteps]
        #print(input_)
        
       
        
        x.append(input_)
        y.append(output)
        
x=np.array(x)
y=np.array(y)


#assigning unique integer to every chords_and_duration



#preparing input sequences::

x_seq=[]
for each_row in x:
    temp=[]
    for each_piece in each_row:
        #assigning unique integer to every note
        temp.append(mapping_dict[each_piece])
    x_seq.append(temp)
    
x_seq = np.array(x_seq)
print(x_seq.shape)


# preparing th output sequences as well::


y_seq=np.array([mapping_dict[i] for i in y])
print(y_seq.shape)

print("\nSplitting x and y \n")
# preserving 80% of the data for training and the rest 20% for the evaluation:

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_seq,y_seq,test_size=0.2,random_state=0)



from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

print("\nTensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

print(tf.config.list_physical_devices('GPU'))

import tensorboard

from keras import backend as K

from keras.layers import *
from keras.models import *
from keras.callbacks import *

from tensorflow import keras

K.clear_session()
model = Sequential()
#embedding layer
model.add(Input(shape= (None,)))
model.add(Embedding(len(mapping_dict), 100,trainable=True)) 

model.add(LSTM(512, return_sequences =True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences =True))
model.add(Dropout(0.3))


model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))

#model.add(Dense(64))
#model.add(Activation('relu'))

model.add(Dense(len(mapping_dict), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop',metrics=['acc'])



print(model.summary())



keras.utils.plot_model(model, "my_first_model_without_tempo.png", show_shapes=True)


#defining call back to save the best model during training>
mc=ModelCheckpoint('new_models/Ionian_MODEL.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_acc', patience = 4)




import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




history = model.fit(np.array(x_train),np.array(y_train),batch_size=128,epochs=50, 
                   validation_data=(np.array(x_val),np.array(y_val)),verbose=1, callbacks=[mc, early_stop])



#loading best model (Previously trained modle)
from keras.models import load_model
model = load_model('new_models/Ionian_MODEL.h5')


import dill
args = {
    "x_train":x_train,
    "x_val":x_val,
    "y_train":y_train,
    "y_val":y_val,
}

with open('new_models/Ionian.dill', 'wb') as f:
    dill.dump(args, f)
    print("\nDump succcedded to new_models/Ionian.dill\n\n")      



import numpy as np
import random
ind = np.random.randint(0,len(x_val)-1)
random_music = x_val[ind]
print(random_music)




no_of_timesteps = 60
predictions=[]
for i in range(200):

    random_music = random_music.reshape(1,no_of_timesteps)
    #print("random music = ", random_music)
    

    prob  = model.predict(random_music)[0]
    y_pred= np.argmax(prob,axis=0)
    predictions.append(y_pred)

    random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
    random_music = random_music[1:]
    
print(predictions)

unique_x_int_to_CD = dict((num, note) for note, num in mapping_dict.items())


predicted_CD = [unique_x_int_to_CD[i] for i in predictions]

print(predicted_CD)


print(predicted_CD[0][0].isdigit())
def round_chord_durations(number):
    if(number>1.25):
        return 2
    if(number>.30):
        return 1.25
    if(number>.10):
        return 0.3
    return 0.1


import time
def pred_out_to_midi(pred_output, initial_ts, initial_tempo, x_val_index):
    
    #generate new score                  
    midi_stream = stream.Stream()
    rounded_durations = ['0.1', '0.3', '1.25', '2'] 
    
    ts_numerator, ts_denominator = initial_ts.split('/')
    new_ts = meter.TimeSignature(f'{ts_numerator}/{ts_denominator}')
    midi_stream.insert(0, new_ts)
    midi_stream.insert(0, tempo.MetronomeMark(number = initial_tempo))
    
    count=0
    for i in range(0, len(pred_output)):
        #print(pred_output[i])
        j=0
        
        if pred_output[i] == '<SOC>':
            i = i+1
            new_chord = []
            while((len(pred_output) > i ) and pred_output[i] != '<EOC>' and j<3):
                #print(pred_output[i])
                if (not pred_output[i][0].isdigit()): 
                    new_chord.append(pred_output[i])
                    i= i+1
                    j=j+1
            #out of while loop i.e end of one chord:
            # Parse and add a chord to the stream
            
            
            #to see if there exists the duration:
            if ((len(pred_output) > i+1 ) and '.' in pred_output[i +1]):
                d = duration.Duration(float(pred_output[i+1]))
                i= i+1
            else:
                d= duration.Duration(float(random.choice(rounded_durations)))
                
            try:
                c = chord.Chord(new_chord)
                c.duration = d
                midi_stream.append(c)
            except:
                print(f'o-o{new_chord}')
        
        
        elif ((len(pred_output) > i ) and pred_output[i] ==  '<EOC>'):
            continue
        
       
            
            
        
        elif((len(pred_output)>i ) and '.' not in pred_output[i]):
            # Parse and add a note to the stream
             #to see if there exists the duration:
            try:
                n = note.Note(pred_output[i])
                if( (len(pred_output) > i + 1)):

                    if ('.' in pred_output[i +1]):
                        d = duration.Duration(float(pred_output[i+1]))
                        i= i+1
                    else:
                        d= duration.Duration(float(random.choice(rounded_durations)))
                else:
                        d= duration.Duration(float(random.choice(rounded_durations)))
            except:
                continue
            
            n.duration = d
            midi_stream.append(n)
            
        
        
        else : #else it is digit
            print(f"c{count}\t {pred_output[i]}")
            count = count+1
            #if(int(pred_output[i]) >= 20):
             #   t = tempo.MetronomeMark(number=int(pred_output[i]))
              #  midi_stream.append(t)
           
            
            # midi_score.append(midi_score.flat.duration.quarterLength(1.5) )
        
    # Save the stream to a MIDI file
    timestr = time.strftime("%Y%m%d-%H%M%S")
    new_file = 'Ionian' + str(x_val_index)+ timestr + '.mid'
    return midi_stream.write('midi', fp=new_file) 




p =pred_out_to_midi(predicted_CD, '3/8', 90, ind)


play_midi_file(p)




'''
unique_x_CD_to_int
unique_x_CD
unique_x_int_to_CD

unique_y_CD

unique_y_CD_to_int

'''
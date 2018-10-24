import pretty_midi
import numpy as np
# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt
# %matplotlib inline
# For putting audio in the notebook
import IPython.display
import copy

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))



## Load the midi file and see its piano roll
## The white lines reperesent the downbeats, which are the first beat in bars.
DATASET_BASE_DIR = '../Few-Shot-Music-Generation/raw-data/freemidi/freemidi_data/'
filepath = DATASET_BASE_DIR + '2 Unlimited/dowhatsgoodforme.mid'
# Load the midi file
pm_src = pretty_midi.PrettyMIDI(filepath)

db = pm_src.get_downbeats()
b = pm_src.get_beats()
nb_db = len(db)
pm_slice_list = []


print('midi file length: {}'.format(pm_src.get_end_time()))
print('number of downbeats (bars): {}'.format(nb_db))

# Plot piano roll
plt.figure(figsize=(12, 4))
plot_piano_roll(pm_src, 24, 84)
ymin, ymax = plt.ylim()
# Plot downbeats as white lines, beats as grey lines
mir_eval.display.events(b, base=ymin, height=ymax, color='#AAAAAA')
mir_eval.display.events(db, base=ymin, height=ymax, color='#FFFFFF', lw=2)
# Only display 20 seconds for clarity
plt.xlim(25, 45)
plt.show()


## Slice the midi file
## The midi file has been stored in the `prettyMIDI` object as a whole. In this step, we slice it into a list of `prettyMIDI` objects, each of which contains a 2-bar segment.

for i in range(2,nb_db,2):
    pm_tmp = copy.deepcopy(pm_src)
    # print(pm_src.get_end_time())
    pm_tmp.adjust_times([db[i-2], db[i]], [0, db[i] - db[i-2]])
    print('Slice #{}, from {} to {}, length:{}.'.format(int(i/2),db[i-2],db[i], pm_tmp.get_end_time()))
    pm_slice_list.append(pm_tmp)


print(len(pm_slice_list))

## Randomly select a slice to listen to
fs = 16000
rand_slice_idx = np.random.randint(np.floor(nb_db/2))
print('Slice #{} sounds like...'.format(rand_slice_idx))
IPython.display.Audio(pm_slice_list[rand_slice_idx].synthesize(fs=16000), rate=16000)

# Sounds like sine waves...
# %%

import music21
from music21 import *
from music21 import converter, instrument, note, chord, stream
import pretty_midi
import numpy as np
import sys
import os
import midi
import glob
import math
import matplotlib.pyplot as plt
import sys
import scipy.stats
import statistics
from scipy import stats
from scipy.stats import sem, t
# %%
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return (m, m-h, m+h)
# %%
def calculate_mean_and_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    standard_error = sem(data)
    t_value = t.ppf((1 + confidence) / 2, n - 1)
    margin_of_error = t_value * standard_error
    return mean, margin_of_error
#%%
def plot_bar_chart_with_confidence_intervals(data1, data2, confidence_interval1, confidence_interval2, labels, x_label):
    x = np.arange(len(labels))  # x-axis positions

    fig, ax = plt.subplots()
    width = 0.35  # width of the bars

    # Plotting the bars
    rects1 = ax.bar(x - width/2, data1, width, label='Original')
    rects2 = ax.bar(x + width/2, data2, width, label='Generated')

    # Plotting the confidence intervals
    ax.errorbar(x - width/2, data1, yerr=confidence_interval1, fmt='none', capsize=4, color='black')
    ax.errorbar(x + width/2, data2, yerr=confidence_interval2, fmt='none', capsize=4, color='black')

    # Add labels, title, and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel('Values')
    # ax.set_title('Pitch Class distributions and confidence intervals for Original and Generated tracks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig("mean_conf_int_all.png")
    plt.show()

# %%
def plot_confidence_interval(x, values, z=1.96, color='#2187bb', mean_color='#f44336', horizontal_line_width=0.25):
    mean = np.mean(values)
    stdev = np.std(values)
    confidence_interval = z * stdev / math.sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color=mean_color)

    return mean, confidence_interval

# %%
def open_midi(midi_path, remove_drums):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = music21.midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [
                ev for ev in mf.tracks[i].events if ev.channel != 10]

    return music21.midi.translate.midiFileToStream(mf)
# %%


def total_used_note(track_num=1):
    pattern = midi.read_midifile(
        'C:\Facultate\IDG_Master\_Sem2\pokemon_assets_generation\pokemon_tracks\pkmn_battle_ost\Pokemon BlackWhite - Battle Team Plasma.mid')

    used_notes = 0
    for i in range(0, len(pattern[track_num])):
        if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
            used_notes += 1
    return used_notes


def avg_pitch_shift(track_num=0):
    pattern = midi.read_midifile(
        'C:\Facultate\IDG_Master\_Sem2\pokemon_assets_generation\pokemon_tracks\pkmn_battle_ost\Pokemon BlackWhite - Battle Team Plasma.mid')
    pattern.make_ticks_abs()
    resolution = pattern.resolution
    total_used_notee = total_used_note(track_num=track_num)
    d_note = np.zeros((max(total_used_notee - 1, 0)))
    # if total_used_note == 0:
    # return 0
    # d_note = np.zeros((total_used_note - 1))
    current_note = 0
    counter = 0
    for i in range(0, len(pattern[track_num])):
        if type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
            if counter != 0:
                d_note[counter - 1] = current_note - \
                    pattern[track_num][i].data[0]
                current_note = pattern[track_num][i].data[0]
                counter += 1
            else:
                current_note = pattern[track_num][i].data[0]
                counter += 1
    pitch_shift = np.mean(abs(d_note))
    return pitch_shift

# %%

def note_length_transition_matrix(path, track_num=0, normalize=0, pause_event=False):
    pattern = midi.read_midifile(path)
    if pause_event is False:
        transition_matrix = np.zeros((12, 12))
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        idx = None
        # basic unit: bar_length/96
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                time_sig = pattern[track_num][i].data
                bar_length = time_sig[track_num] * \
                    resolution * 4 / 2**(time_sig[1])
            elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                if 'time_sig' not in locals():  # set default bar length as 4 beat
                    bar_length = 4 * resolution
                    time_sig = [4, 2, 24, 8]
                unit = bar_length / 96.
                hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit *
                             72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                current_tick = pattern[track_num][i].tick
                current_note = pattern[track_num][i].data[0]
                # find note off
                for j in range(i, len(pattern[track_num])):
                    if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                        if pattern[track_num][j].data[0] == current_note:
                            note_length = pattern[track_num][j].tick - \
                                current_tick
                            distance = np.abs(
                                np.array(hist_list) - note_length)

                            last_idx = idx
                            idx = distance.argmin()
                            if last_idx is not None:
                                transition_matrix[last_idx][idx] += 1
                            break
    else:
        transition_matrix = np.zeros((24, 24))
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        idx = None
        # basic unit: bar_length/96
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                time_sig = pattern[track_num][i].data
                bar_length = time_sig[track_num] * \
                    resolution * 4 / 2**(time_sig[1])
            elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                check_previous_off = True
                if 'time_sig' not in locals():  # set default bar length as 4 beat
                    bar_length = 4 * resolution
                    time_sig = [4, 2, 24, 8]
                unit = bar_length / 96.
                tol = 3. * unit
                hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit *
                             72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                current_tick = pattern[track_num][i].tick
                current_note = pattern[track_num][i].data[0]
                # find next note off
                for j in range(i, len(pattern[track_num])):
                    # find next note off
                    if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                        if pattern[track_num][j].data[0] == current_note:

                            note_length = pattern[track_num][j].tick - \
                                current_tick
                            distance = np.abs(
                                np.array(hist_list) - note_length)
                            last_idx = idx
                            idx = distance.argmin()
                            if last_idx is not None:
                                transition_matrix[last_idx][idx] += 1
                            break
                        else:
                            if pattern[track_num][j].tick == current_tick:
                                check_previous_off = False

                # find previous note off/on
                if check_previous_off is True:
                    for j in range(i - 1, 0, -1):
                        if type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] != 0:
                            break

                        elif type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):

                            note_length = current_tick - \
                                pattern[track_num][j].tick
                            distance = np.abs(
                                np.array(hist_list) - note_length)

                            last_idx = idx
                            idx = distance.argmin()
                            if last_idx is not None:
                                if distance[idx] < tol:
                                    idx = last_idx
                                    transition_matrix[last_idx][idx + 12] += 1
                            break

    if normalize == 0:
        return transition_matrix

    elif normalize == 1:

        sums = np.sum(transition_matrix, axis=1)
        sums[sums == 0] = 1
        return transition_matrix / sums.reshape(-1, 1)

    elif normalize == 2:

        return transition_matrix / sum(sum(transition_matrix))

    else:
        print("invalid normalization mode, return unnormalized matrix")
        return transition_matrix

# %%
"""
# Add here paths for each folder you want to run the evaluation for

# battle_tracks_path = 'C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/pkmn_battle_ost/*.mid'
# ow_tracks_path = 'C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/pkmn_ow_ost/*.mid'
# mixed_tracks_path = 'C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/pkmn_ost/*.mid'

# battle_output_path = 'C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/battle_output/*.mid'
# ow_output_path = 'C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/ow_output/*.mid'
# mixed_output_path = 'C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/mixed_output/*.mid'

# Paths for biomes
"""
# INPUT FOLDERS
cave_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/cave/*.mid' ## 
city_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/city/*.mid' ## 
forest_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/forest/*.mid' ##
mountain_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/mountain/*.mid' ##
route_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/route/*.mid' ##
sea_ocean_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/sea_ocean/*.mid' ## 
tower_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/tower/*.mid' ## 

# OUTPUT FOLDERS
cave_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/cave_output/*.mid' ##
city_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/city_output/*.mid' ##
forest_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/forest_output/*.mid' ##
mountain_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/mountain_output/*.mid' ##
route_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/route_output/*.mid' ##
sea_ocean_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/sea_ocean_output/*.mid' ##
tower_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/tower_output/*.mid' ##

# %%

####################### TOTAL NOTE LENGTH TRANSITION MATRIX #######################

def totalNLTM(path, track_num=0, normalize = 0):
    bigNLTM = np.zeros((12, 12))
    for i, file in enumerate(glob.glob(path)):
        print('\r', 'Parsing file ', i, " ", file, end='\r')
        bigNLTM = bigNLTM + \
            note_length_transition_matrix(file, track_num=track_num, normalize=normalize)
    return bigNLTM


def showNLTM(mat, title):
    plt.figure(figsize=(8, 8))
    plt.matshow(mat, fignum=1, cmap=plt.cm.bwr, vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.title(title)
    plt.xticks(range(12),
               ['full', 'half', 'quarter', '8th', '16th', 'dot half', 'dot quarter', 'dot 8th',
                   'dot 16th', 'half note triplet', 'quarter note triplet', '8th note triplet'],
               rotation=45)
    plt.yticks(range(12),
               ['full', 'half', 'quarter', '8th', '16th', 'dot half', 'dot quarter', 'dot 8th', 'dot 16th', 'half note triplet', 'quarter note triplet', '8th note triplet'],)
    plt.savefig(title)
    plt.show()

####################### Old, used for battle, ow, mixed #######################

# battleOriginalNLTM = totalNLTM(battle_tracks_path, 1, 0)
# battleOriginalNLTM = battleOriginalNLTM / sum(sum(battleOriginalNLTM))
# battleGeneratedNLTM = totalNLTM(battle_output_path, 0, 0)
# battleGeneratedNLTM = battleGeneratedNLTM / sum(sum(battleGeneratedNLTM))
# battleTotal = battleOriginalNLTM - battleGeneratedNLTM
# showNLTM(battleTotal,"Difference note length transition matrix between original battle and generated battle")

# overworldOriginalNLTM = totalNLTM(ow_tracks_path, 1, 0)
# overworldOriginalNLTM = overworldOriginalNLTM / sum(sum(overworldOriginalNLTM))
# overworldGeneratedNLTM = totalNLTM(ow_output_path, 0, 0)
# overworldGeneratedNLTM = overworldGeneratedNLTM / sum(sum(overworldGeneratedNLTM))
# battleTotal = battleOriginalNLTM - overworldGeneratedNLTM
# showNLTM(battleTotal,"Difference note length transition matrix between original ow and generated ow")

# mixedOriginalNLTM = totalNLTM(mixed_tracks_path, 1, 0)
# mixedOriginalNLTM = mixedOriginalNLTM / sum(sum(mixedOriginalNLTM))
# mixedGeneratedNLTM = totalNLTM(mixed_output_path, 0, 0)
# mixedGeneratedNLTM = mixedGeneratedNLTM / sum(sum(mixedGeneratedNLTM))
# battleTotal = mixedOriginalNLTM - mixedGeneratedNLTM
# showNLTM(battleTotal,"Difference note length transition matrix between original mixed and generated mixed")

# totalNLTM(ow_tracks_path, "Overworld original", 1, 1)
# totalNLTM(mixed_tracks_path, "Mixed original", 1, 2)
# totalNLTM(battle_output_path, "Battle Generated", normalize=0)
# totalNLTM(ow_output_path, "Overworld generated", normalize=1)
# totalNLTM(mixed_output_path, "Mixed output", normalize=2)

#################### Old, used for battle, ow, mixed ####################

########################################################################
#################### New, used for biome generation ####################
########################################################################

#%%
#################### CAVE ####################

caveOriginalNLTM = totalNLTM(cave_folder_path, 1, 0)
caveOriginalNLTM = caveOriginalNLTM / sum(sum(caveOriginalNLTM))
caveGeneratedNLTM = totalNLTM(cave_output_folder_path, 0, 0)
caveGeneratedNLTM = caveGeneratedNLTM / sum(sum(caveGeneratedNLTM))
caveTotal = caveOriginalNLTM - caveGeneratedNLTM
print("\n\n")
print(caveTotal.mean())
showNLTM(caveTotal,"Difference note length transition matrix between original cave and generated cave")
#%%
#################### CITY ####################

cityOriginalNLTM = totalNLTM(city_folder_path, 1, 0)
cityOriginalNLTM = cityOriginalNLTM / sum(sum(cityOriginalNLTM))
cityGeneratedNLTM = totalNLTM(city_output_folder_path, 0, 0)
cityGeneratedNLTM = cityGeneratedNLTM / sum(sum(cityGeneratedNLTM))
cityTotal = cityOriginalNLTM - cityGeneratedNLTM
print("\n\n")
print(cityTotal.mean())
showNLTM(cityTotal,"Difference note length transition matrix between original city and generated city")
#%%
#################### FOREST ####################

forestOriginalNLTM = totalNLTM(forest_folder_path, 1, 0)
forestOriginalNLTM = forestOriginalNLTM / sum(sum(forestOriginalNLTM))
forestGeneratedNLTM = totalNLTM(forest_output_folder_path, 0, 0)
forestGeneratedNLTM = forestGeneratedNLTM / sum(sum(forestGeneratedNLTM))
forestTotal = forestOriginalNLTM - forestGeneratedNLTM
print("\n\n")
print(forestTotal.mean())
showNLTM(forestTotal,"Difference note length transition matrix between original forest and generated forest")
#%%
#################### MOUNTAIN ####################

mountainOriginalNLTM = totalNLTM(mountain_folder_path, 1, 0)
mountainOriginalNLTM = mountainOriginalNLTM / sum(sum(mountainOriginalNLTM))
mountainGeneratedNLTM = totalNLTM(mountain_output_folder_path, 0, 0)
mountainGeneratedNLTM = mountainGeneratedNLTM / sum(sum(mountainGeneratedNLTM))
mountainTotal = mountainOriginalNLTM - mountainGeneratedNLTM
print("\n\n")
print(mountainTotal.mean())
showNLTM(mountainTotal,"Difference note length transition matrix between original mountain and generated mountain")
#%%
#################### ROUTE ####################

routeOriginalNLTM = totalNLTM(route_folder_path, 1, 0)
routeOriginalNLTM = routeOriginalNLTM / sum(sum(routeOriginalNLTM))
routeGeneratedNLTM = totalNLTM(route_output_folder_path, 0, 0)
routeGeneratedNLTM = routeGeneratedNLTM / sum(sum(routeGeneratedNLTM))
routeTotal = routeOriginalNLTM - routeGeneratedNLTM
print("\n\n")
print(routeTotal.mean())
showNLTM(routeTotal,"Difference note length transition matrix between original route and generated route")
#%%
#################### SEA_OCEAN ####################

sea_oceanOriginalNLTM = totalNLTM(sea_ocean_folder_path, 1, 0)
sea_oceanOriginalNLTM = sea_oceanOriginalNLTM / sum(sum(sea_oceanOriginalNLTM))
sea_oceanGeneratedNLTM = totalNLTM(sea_ocean_output_folder_path, 0, 0)
sea_oceanGeneratedNLTM = sea_oceanGeneratedNLTM / sum(sum(sea_oceanGeneratedNLTM))
sea_oceanTotal = sea_oceanOriginalNLTM - sea_oceanGeneratedNLTM
print("\n\n")
print(sea_oceanTotal.mean())
showNLTM(sea_oceanTotal,"Difference note length transition matrix between original sea_ocean and generated sea_ocean")
#%%
#################### TOWER ####################

towerOriginalNLTM = totalNLTM(tower_folder_path, 1, 0)
towerOriginalNLTM = towerOriginalNLTM / sum(sum(towerOriginalNLTM))
towerGeneratedNLTM = totalNLTM(tower_output_folder_path, 0, 0)
towerGeneratedNLTM = towerGeneratedNLTM / sum(sum(towerGeneratedNLTM))
towerTotal = towerOriginalNLTM - towerGeneratedNLTM
print("\n\n")
print(towerTotal.mean())
showNLTM(towerTotal,"Difference note length transition matrix between original tower and generated tower")

#%%
#################### AVERAGE NLTM ####################
averageTotal = (caveTotal + cityTotal + forestTotal + mountainTotal + routeTotal + sea_oceanTotal + towerTotal)/7
showNLTM(averageTotal, "Average note length transition matrix")
# %%

####################### TOTAL PITCH CLASS TRANSITION MATRIX #######################


def pitch_class_transition_matrix(path):
    pokemon_midi = pretty_midi.PrettyMIDI(path)
    transition_matrix = pokemon_midi.get_pitch_class_transition_matrix(normalize=False)
    if np.isnan(transition_matrix[0].any()):
        return np.zeros((12, 12))
    return transition_matrix


def totalPCTM(path):
    bigPCTM = np.zeros((12, 12))
    for i, file in enumerate(glob.glob(path)):
        print('\r', 'Parsing file ', i, " ", file, end='\x1b[1K\r')
        current_file_PCTM = pitch_class_transition_matrix(file)
        bigPCTM = bigPCTM + current_file_PCTM
    return bigPCTM
 

def showPCTM(transition_matrix, title):
    plt.figure(figsize=(8, 8))
    plt.matshow(transition_matrix, fignum=1, cmap=plt.cm.bwr, vmin=-0.02, vmax=0.02)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(12),
               ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B'],)
    plt.yticks(range(12),
               ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B'],)
    plt.savefig(title)
    plt.show()

    
#################### Old, used for battle, ow, mixed ####################

# battleOriginalNLTM = totalPCTM(battle_tracks_path)
# battleOriginalNLTM = battleOriginalNLTM / sum(sum(battleOriginalNLTM))
# battleGeneratedNLTM = totalPCTM(battle_output_path)
# battleGeneratedNLTM = battleGeneratedNLTM / sum(sum(battleGeneratedNLTM))
# battleTotal = battleOriginalNLTM - battleGeneratedNLTM
# showPCTM(battleTotal,"Difference pitch class transition matrix between original battle and generated battle\n\n\n")

# overworldOriginalNLTM = totalPCTM(ow_tracks_path)
# overworldOriginalNLTM = overworldOriginalNLTM / sum(sum(overworldOriginalNLTM))
# overworldGeneratedNLTM = totalPCTM(ow_output_path)
# overworldGeneratedNLTM = overworldGeneratedNLTM / sum(sum(overworldGeneratedNLTM))
# battleTotal = battleOriginalNLTM - overworldGeneratedNLTM
# showPCTM(battleTotal,"Difference pitch class transition matrix between original ow and generated ow\n\n\n")

# mixedOriginalNLTM = totalPCTM(mixed_tracks_path)
# mixedOriginalNLTM = mixedOriginalNLTM / sum(sum(mixedOriginalNLTM))
# mixedGeneratedNLTM = totalPCTM(mixed_output_path)
# mixedGeneratedNLTM = mixedGeneratedNLTM / sum(sum(mixedGeneratedNLTM))
# battleTotal = mixedOriginalNLTM - mixedGeneratedNLTM
# showPCTM(battleTotal,"Difference pitch class transition matrix between original mixed and generated mixed\n\n\n")

# totalPCTM(battle_tracks_path, "Battle Original")
# totalPCTM(ow_tracks_path, "Overworld original")
# totalPCTM(mixed_tracks_path, "Mixed original")
# totalPCTM(battle_output_path, "Battle Generated")
# totalPCTM(ow_output_path, "Overworld generated")
# totalPCTM(mixed_output_path, "Mixed output")


#################### Old, used for battle, ow, mixed ####################


########################################################################
#################### New, used for biome generation ####################
########################################################################
#%%
#################### CAVE ####################

caveOriginalPCTM = totalPCTM(cave_folder_path)
caveOriginalPCTM = caveOriginalPCTM / sum(sum(caveOriginalPCTM))
caveGeneratedPCTM = totalPCTM(cave_output_folder_path)
caveGeneratedPCTM = caveGeneratedPCTM / sum(sum(caveGeneratedPCTM))
caveTotal = caveOriginalPCTM - caveGeneratedPCTM
print("\n\n")
print(caveTotal.mean())
showPCTM(caveTotal,"Difference pitch class transition matrix between original cave and generated cave")
#%%
#################### CITY ####################

cityOriginalPCTM = totalPCTM(city_folder_path)
cityOriginalPCTM = cityOriginalPCTM / sum(sum(cityOriginalPCTM))
cityGeneratedPCTM = totalPCTM(city_output_folder_path)
cityGeneratedPCTM = cityGeneratedPCTM / sum(sum(cityGeneratedPCTM))
cityTotal = cityOriginalPCTM - cityGeneratedPCTM
print("\n\n")
print(cityTotal.mean())
showPCTM(cityTotal,"Difference pitch class transition matrix between original city and generated city")
#%%
#################### FOREST ####################

forestOriginalPCTM = totalPCTM(forest_folder_path)
forestOriginalPCTM = forestOriginalPCTM / sum(sum(forestOriginalPCTM))
forestGeneratedPCTM = totalPCTM(forest_output_folder_path)
forestGeneratedPCTM = forestGeneratedPCTM / sum(sum(forestGeneratedPCTM))
forestTotal = forestOriginalPCTM - forestGeneratedPCTM
print("\n\n")
print(forestTotal.mean())
showPCTM(forestTotal,"Difference pitch class transition matrix between original forest and generated forest")
#%%
#################### MOUNTAIN ####################

mountainOriginalPCTM = totalPCTM(mountain_folder_path)
mountainOriginalPCTM = mountainOriginalPCTM / sum(sum(mountainOriginalPCTM))
mountainGeneratedPCTM = totalPCTM(mountain_output_folder_path)
mountainGeneratedPCTM = mountainGeneratedPCTM / sum(sum(mountainGeneratedPCTM))
mountainTotal = mountainOriginalPCTM - mountainGeneratedPCTM
print("\n\n")
print(mountainTotal.mean())
showPCTM(mountainTotal,"Difference pitch class transition matrix between original mountain and generated mountain")
#%%
#################### ROUTE ####################

routeOriginalPCTM = totalPCTM(route_folder_path)
routeOriginalPCTM = routeOriginalPCTM / sum(sum(routeOriginalPCTM))
routeGeneratedPCTM = totalPCTM(route_output_folder_path)
routeGeneratedPCTM = routeGeneratedPCTM / sum(sum(routeGeneratedPCTM))
routeTotal = routeOriginalPCTM - routeGeneratedPCTM
print("\n\n")
print(routeTotal.mean())
showPCTM(routeTotal,"Difference pitch class transition matrix between original route and generated route")
#%%
#################### SEA_OCEAN ####################

sea_oceanOriginalPCTM = totalPCTM(sea_ocean_folder_path)
sea_oceanOriginalPCTM = sea_oceanOriginalPCTM / sum(sum(sea_oceanOriginalPCTM))
sea_oceanGeneratedPCTM = totalPCTM(sea_ocean_output_folder_path)
sea_oceanGeneratedPCTM = sea_oceanGeneratedPCTM / sum(sum(sea_oceanGeneratedPCTM))
sea_oceanTotal = sea_oceanOriginalPCTM - sea_oceanGeneratedPCTM
print("\n\n")
print(sea_oceanTotal.mean())
showPCTM(sea_oceanTotal,"Difference pitch class transition matrix between original sea_ocean and generated sea_ocean")
#%%
#################### TOWER ####################

towerOriginalPCTM = totalPCTM(tower_folder_path)
towerOriginalPCTM = towerOriginalPCTM / sum(sum(towerOriginalPCTM))
towerGeneratedPCTM = totalPCTM(tower_output_folder_path)
towerGeneratedPCTM = towerGeneratedPCTM / sum(sum(towerGeneratedPCTM))
towerTotal = towerOriginalPCTM - towerGeneratedPCTM
print("\n\n")
print(towerTotal.mean())
showPCTM(towerTotal,"Difference pitch class transition matrix between original tower and generated tower")

# %%

####################### AVERAGE NUMBER OF DIFFERENT PITCHES #######################

## The number of different pitches within a sample.
def total_used_pitch(path):
    midi_file = pretty_midi.PrettyMIDI(path)
    piano_roll = midi_file.instruments[0].get_piano_roll(fs=100)
    sum_notes = np.sum(piano_roll, axis=1)
    used_pitch = np.sum(sum_notes > 0)
    return used_pitch


def average_total_used_pitch(path):
    sum = 0
    totalFiles = 0
    for i, file in enumerate(glob.glob(path)):
        print('\r', 'Parsing file ', i, " ", file, end='')
        totalUsedPitch = total_used_pitch(file)
        sum += totalUsedPitch
        totalFiles = i + 1
    return sum/totalFiles

#################### Old, used for battle, ow, mixed ####################

# averageTotalUsedPitch = average_total_used_pitch(battle_tracks_path)
# print("\n\n Battle original Average number of different pitches: ")
# print(averageTotalUsedPitch)
# averageTotalUsedPitch = average_total_used_pitch(ow_tracks_path)
# print("\n\n Overworld original Average number of different pitches: ")
# print(averageTotalUsedPitch)
# averageTotalUsedPitch = average_total_used_pitch(mixed_tracks_path)
# print("\n\n Mixed original Average number of different pitches: ")
# print(averageTotalUsedPitch)
# averageTotalUsedPitch = average_total_used_pitch(battle_output_path)
# print("\n\n Battle generated Average number of different pitches: ")
# print(averageTotalUsedPitch)
# averageTotalUsedPitch = average_total_used_pitch(ow_output_path)
# print("\n\n Overworld generated Average number of different pitches: ")
# print(averageTotalUsedPitch)
# averageTotalUsedPitch = average_total_used_pitch(mixed_output_path)
# print("\n\n Mixed generated Average number of different pitches: ")
# print(averageTotalUsedPitch)


########################################################################
#################### New, used for biome generation ####################
########################################################################
#%%
#################### CAVE ####################
averageTotalUsedPitch = average_total_used_pitch(cave_folder_path)
print("\n\n Cave original Average number of different pitches: ")
print(averageTotalUsedPitch)
averageTotalUsedPitch = average_total_used_pitch(cave_output_folder_path)
print("\n\n Cave generated Average number of different pitches: ")
print(averageTotalUsedPitch)
#%%
#################### CITY ####################
averageTotalUsedPitch = average_total_used_pitch(city_folder_path)
print("\n\n CITY original Average number of different pitches: ")
print(averageTotalUsedPitch)
averageTotalUsedPitch = average_total_used_pitch(city_output_folder_path)
print("\n\n CITY generated Average number of different pitches: ")
print(averageTotalUsedPitch)
#%%
#################### FOREST ####################
averageTotalUsedPitch = average_total_used_pitch(forest_folder_path)
print("\n\n FOREST original Average number of different pitches: ")
print(averageTotalUsedPitch)
averageTotalUsedPitch = average_total_used_pitch(forest_output_folder_path)
print("\n\n FOREST generated Average number of different pitches: ")
print(averageTotalUsedPitch)
#%%
#################### MOUNTAIN ####################
averageTotalUsedPitch = average_total_used_pitch(mountain_folder_path)
print("\n\n MOUNTAIN original Average number of different pitches: ")
print(averageTotalUsedPitch)
averageTotalUsedPitch = average_total_used_pitch(mountain_output_folder_path)
print("\n\n MOUNTAIN generated Average number of different pitches: ")
print(averageTotalUsedPitch)
#%%
#################### ROUTE ####################
averageTotalUsedPitch = average_total_used_pitch(route_folder_path)
print("\n\n ROUTE original Average number of different pitches: ")
print(averageTotalUsedPitch)
averageTotalUsedPitch = average_total_used_pitch(route_output_folder_path)
print("\n\n ROUTE generated Average number of different pitches: ")
print(averageTotalUsedPitch)
#%%
#################### SEA_OCEAN ####################
averageTotalUsedPitch = average_total_used_pitch(sea_ocean_folder_path)
print("\n\n SEA_OCEAN original Average number of different pitches: ")
print(averageTotalUsedPitch)
averageTotalUsedPitch = average_total_used_pitch(sea_ocean_output_folder_path)
print("\n\n SEA_OCEAN generated Average number of different pitches: ")
print(averageTotalUsedPitch)
#%%
#################### TOWER ####################
averageTotalUsedPitch = average_total_used_pitch(tower_folder_path)
print("\n\n TOWER original Average number of different pitches: ")
print(averageTotalUsedPitch)
averageTotalUsedPitch = average_total_used_pitch(tower_output_folder_path)
print("\n\n TOWER generated Average number of different pitches: ")
print(averageTotalUsedPitch)
# %%

####################### AVERAGE PITCH RANGE #######################


def pitch_range(path):
    midi_file = pretty_midi.PrettyMIDI(path)
    piano_roll = midi_file.instruments[0].get_piano_roll(fs=100)
    pitch_index = np.where(np.sum(piano_roll, axis=1) > 0)
    p_range = np.max(pitch_index) - np.min(pitch_index)
    return p_range


def average_pitch_range(path):
    sum = 0
    totalFiles = 0
    pitch_ranges = []
    for i, file in enumerate(glob.glob(path)):
        # print('\r', 'Parsing file ', i, " ", file, end='')
        pitchRange = pitch_range(file)
        pitch_ranges.append(pitchRange)
        sum += pitchRange
        totalFiles = i + 1
    mean, conf_int = calculate_mean_and_confidence_interval(pitch_ranges)
    return mean, conf_int


#################### Old, used for battle, ow, mixed ####################

# averagePitchRange = average_pitch_range(battle_tracks_path)
# print("\n\nBattle original Average pitch range: ")
# print(averagePitchRange)
# averagePitchRange = average_pitch_range(ow_tracks_path)
# print("\n\nOverworld originalAverage pitch range: ")
# print(averagePitchRange)
# averagePitchRange = average_pitch_range(mixed_tracks_path)
# print("\n\nMixed original Average pitch range: ")
# print(averagePitchRange)
# averagePitchRange = average_pitch_range(battle_output_path)
# print("\n\nBattle generated Average pitch range: ")
# print(averagePitchRange)
# averagePitchRange = average_pitch_range(ow_output_path)
# print("\n\nOverworld generated Average pitch range: ")
# print(averagePitchRange)
# averagePitchRange = average_pitch_range(mixed_output_path)
# print("\n\nMixed generated Average pitch range: ")
# print(averagePitchRange)

########################################################################
#################### New, used for biome generation ####################
########################################################################
#%%
#################### CAVE ####################
averagePitchRange, conf_int = average_pitch_range(cave_folder_path)
print("\n\n CAVE original Average pitch range: ")
print(averagePitchRange)
averagePitchRange, conf_int = average_pitch_range(cave_output_folder_path)
print("\n\n CAVE generated Average pitch range: ")
print(averagePitchRange)
#%%
#################### CITY ####################
averagePitchRange, conf_int = average_pitch_range(city_folder_path)
print("\n\n CITY original Average pitch range: ")
print(averagePitchRange)
averagePitchRange, conf_int = average_pitch_range(city_output_folder_path)
print("\n\n CITY generated Average pitch range: ")
print(averagePitchRange)
#%%
#################### FOREST ####################
averagePitchRange, conf_int = average_pitch_range(forest_folder_path)
print("\n\n FOREST original Average pitch range: ")
print(averagePitchRange)
averagePitchRange, conf_int = average_pitch_range(forest_output_folder_path)
print("\n\n FOREST generated Average pitch range: ")
print(averagePitchRange)
#%%
#################### MOUNTAIN ####################
averagePitchRange, conf_int = average_pitch_range(mountain_folder_path)
print("\n\n MOUNTAIN original Average pitch range: ")
print(averagePitchRange)
averagePitchRange, conf_int = average_pitch_range(mountain_output_folder_path)
print("\n\n MOUNTAIN generated Average pitch range: ")
print(averagePitchRange)
#%%
#################### ROUTE ####################
averagePitchRange, conf_int = average_pitch_range(route_folder_path)
print("\n\n ROUTE original Average pitch range: ")
print(averagePitchRange)
averagePitchRange, conf_int = average_pitch_range(route_output_folder_path)
print("\n\n ROUTE generated Average pitch range: ")
print(averagePitchRange)
#%%
#################### SEA_OCEAN ####################
averagePitchRange, conf_int = average_pitch_range(sea_ocean_folder_path)
print("\n\n SEA_OCEAN original Average pitch range: ")
print(averagePitchRange)
averagePitchRange, conf_int = average_pitch_range(sea_ocean_output_folder_path)
print("\n\n SEA_OCEAN generated Average pitch range: ")
print(averagePitchRange)
#%%
#################### TOWER ####################
averagePitchRange, conf_int = average_pitch_range(tower_folder_path)
print("\n\n TOWER original Average pitch range: ")
print(averagePitchRange)
averagePitchRange, conf_int = average_pitch_range(tower_output_folder_path)
print("\n\n TOWER generated Average pitch range: ")
print(averagePitchRange)
# %% TOTAL
original_apr_means = []
original_apr_CI = []

generated_apr_means = []
generated_apr_CI = []

apr_original_cave, conf_int_original_cave = average_pitch_range(cave_folder_path)
original_apr_means.append(apr_original_cave)
original_apr_CI.append(conf_int_original_cave)
apr_original_city, conf_int_original_city = average_pitch_range(city_folder_path)
original_apr_means.append(apr_original_city)
original_apr_CI.append(conf_int_original_city)
apr_original_forest, conf_int_original_forest = average_pitch_range(forest_folder_path)
original_apr_means.append(apr_original_forest)
original_apr_CI.append(conf_int_original_forest)
apr_original_mountain, conf_int_original_mountain = average_pitch_range(mountain_folder_path)
original_apr_means.append(apr_original_mountain)
original_apr_CI.append(conf_int_original_mountain)
apr_original_route, conf_int_original_route = average_pitch_range(route_folder_path)
original_apr_means.append(apr_original_route)
original_apr_CI.append(conf_int_original_route)
apr_original_sea, conf_int_original_sea = average_pitch_range(sea_ocean_folder_path)
original_apr_means.append(apr_original_sea)
original_apr_CI.append(conf_int_original_sea)
apr_original_tower, conf_int_original_tower = average_pitch_range(tower_folder_path)
original_apr_means.append(apr_original_tower)
original_apr_CI.append(conf_int_original_tower)

apr_generated_cave, conf_int_generated_cave = average_pitch_range(cave_output_folder_path)
generated_apr_means.append(apr_generated_cave)
generated_apr_CI.append(conf_int_generated_cave)
apr_generated_city, conf_int_generated_city = average_pitch_range(city_output_folder_path)
generated_apr_means.append(apr_generated_city)
generated_apr_CI.append(conf_int_generated_city)
apr_generated_forest, conf_int_generated_forest = average_pitch_range(forest_output_folder_path)
generated_apr_means.append(apr_generated_forest)
generated_apr_CI.append(conf_int_generated_forest)
apr_generated_mountain, conf_int_generated_mountain = average_pitch_range(mountain_output_folder_path)
generated_apr_means.append(apr_generated_mountain)
generated_apr_CI.append(conf_int_generated_mountain)
apr_generated_route, conf_int_generated_route = average_pitch_range(route_output_folder_path)
generated_apr_means.append(apr_generated_route)
generated_apr_CI.append(conf_int_generated_route)
apr_generated_sea, conf_int_generated_sea = average_pitch_range(sea_ocean_output_folder_path)
generated_apr_means.append(apr_generated_sea)
generated_apr_CI.append(conf_int_generated_sea)
apr_generated_tower, conf_int_generated_tower = average_pitch_range(tower_output_folder_path)
generated_apr_means.append(apr_generated_tower)
generated_apr_CI.append(conf_int_generated_tower)

biome_labels = ["Cave", "City", "Forest", "Mountain", "Route", "Sea", "Tower"]
plot_bar_chart_with_confidence_intervals(original_apr_means, generated_apr_means, original_apr_CI, generated_apr_CI, biome_labels, "Biomes")


# %%
####################### AVERAGE PITCH CLASS HISTOGRAM #######################


def pitch_class_histogram(path):
    midi_file = pretty_midi.PrettyMIDI(path)
    histogram = midi_file.get_pitch_class_histogram(
        use_duration=True, normalize=True)
    return histogram


def plotPitchClassHistogram(values, title):
    fig, ax = plt.subplots()
    plt.title(title + "Pitch class histogram")
    pitch = ['C', 'C#', 'D', 'E', 'Eb', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    ax.bar(pitch, values)
    plt.show()


def totalPitchClass(path):
    histogram = np.zeros(12)
    for i, file in enumerate(glob.glob(path)):
        values = pitch_class_histogram(file)
        histogram = np.add(histogram, values)
    return histogram

def plotTwoPitchHist(path1, path2, title):
    original = totalPitchClass(path1)
    original = original / np.sum(original)
    generated = totalPitchClass(path2)
    generated = generated / np.sum(generated)
    pitch = ['C', 'C#', 'D', 'E', 'Eb', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    x_axis = np.arange(len(pitch))
    plt.bar(x_axis - 0.2, original, width=0.4, label='Original')
    plt.bar(x_axis + 0.2, generated, width=0.4, label='Generated')    
    plt.title(title + " Pitch class histogram")
    plt.xticks(x_axis, pitch)
    plt.legend(loc='upper right')
    plt.savefig(title + " Pitch class histogram")
    plt.show()

def plotThreePitchHist(path1, path2, path3, title):
    originalOW = totalPitchClass(path1)
    # originalOW = originalOW / np.sum(originalOW)
    originalBattle = totalPitchClass(path2)
    # originalBattle = originalBattle / np.sum(originalBattle)
    generated = totalPitchClass(path3)
    generated = generated / np.sum(generated)
    pitch = ['C', 'C#', 'D', 'E', 'Eb', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    x_axis = np.arange(len(pitch))
    plt.bar(x_axis - 0.2, originalOW, width=0.2, label='Original OW')
    plt.bar(x_axis, originalBattle, width=0.2, label='Original Battle') 
    plt.bar(x_axis + 0.2, generated, width=0.2, label='Generated')    
    plt.title(title + " Pitch class histogram")
    plt.xticks(x_axis, pitch)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(title)

def buildSTDPlotPCH(xpos, CTEs, error, title):
    """
        For measuring and plotting mean and standard deviation
    """
    # Build the plot
    pitch = ['C', 'C#', 'D', 'E', 'Eb', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    
    fig, ax = plt.subplots()
    ax.bar(xpos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    # ax.set_ylabel()
    ax.set_xticks(xpos)
    ax.set_xticklabels(pitch)
    ax.set_title(title)
    ax.yaxis.grid(True)
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(title)
    plt.show()

def plotMeanAndStandardDeviationPCH(title, path, number_of_files, track_num=0):
    # Calculate pitch histogram for each file, save in a matrix
    pch_matrix = np.zeros((number_of_files, 12))
    for i, file in enumerate(glob.glob(path)): # pitch values for each file
        values = pitch_class_histogram(file)
        pch_matrix[i] += values
    # print(pch_matrix)

    # Calculate mean for each pitch from the matrix (each column)
    pch_means_array = pch_matrix.mean(0)
    # print(pch_means_array)    

    # Calculate standard deviation for each pitch from matrix (each column)
    pch_std_array = pch_matrix.std(0)
    # print(pch_std_array)
    
    x_pos = np.arange(len(pch_matrix[0]))
    CTEs = pch_means_array
    error = pch_std_array
    plot_title = title + " pitch class mean and standard deviation"
    buildSTDPlotPCH(x_pos, CTEs, error, plot_title)
    plt.show()
    plt.savefig(title)


def get_pitch_class_histograms_multiple_files(title, path, number_of_files, track_num=0):
    """
        Calculate pitch histogram for each file, returns a matrix
    """
    pch_matrix = np.zeros((number_of_files, 12))
    for i, file in enumerate(glob.glob(path)): # pitch values for each file
        values = pitch_class_histogram(file)
        pch_matrix[i] += values
    return pch_matrix

#################### Old, used for battle, ow, mixed ####################

# plotTwoPitchHist(battle_tracks_path, battle_output_path, "Battle")
# plotTwoPitchHist(ow_tracks_path, ow_output_path, "Overworld")
# plotThreePitchHist(ow_tracks_path, battle_tracks_path, mixed_output_path, "Mixed")

########################################################################
#################### New, used for biome generation ####################
########################################################################
#%%
#################### CAVE ####################
plotTwoPitchHist(cave_folder_path, cave_output_folder_path, "CAVE")
#%%
#################### CITY ####################
plotTwoPitchHist(city_folder_path, city_output_folder_path, "CITY")
#%%
#################### FOREST ####################
plotTwoPitchHist(forest_folder_path, forest_output_folder_path, "FOREST")
#%%
#################### MOUNTAIN ####################
plotTwoPitchHist(mountain_folder_path, mountain_output_folder_path, "MOUNTAIN")
#%%
#################### ROUTE ####################
plotTwoPitchHist(route_folder_path, route_output_folder_path, "ROUTE")
#%%
#################### SEA_OCEAN ####################
plotTwoPitchHist(sea_ocean_folder_path, sea_ocean_output_folder_path, "SEA_OCEAN")
#%%
#################### TOWER ####################
plotTwoPitchHist(tower_folder_path, tower_output_folder_path, "TOWER")
# %%
#%%
#####################################################################
#################### Mean and Standard Deviation ####################
#####################################################################
#%%
#################### CAVE ####################
plotMeanAndStandardDeviationPCH("CAVE original", cave_folder_path, 19, 1)
#%%
plotMeanAndStandardDeviationPCH("CAVE generated", cave_output_folder_path, 20, 0)
#%%
#################### CITY ####################
plotMeanAndStandardDeviationPCH("CITY original", city_folder_path, 89, 1)
#%%
plotMeanAndStandardDeviationPCH("CITY generated", city_output_folder_path, 20, 0)
#%%
#################### FOREST ####################
plotMeanAndStandardDeviationPCH("FOREST original", forest_folder_path, 22, 1)
#%%
plotMeanAndStandardDeviationPCH("FOREST generated", forest_output_folder_path, 20, 0)
#%%
#################### MOUNTAIN ####################
plotMeanAndStandardDeviationPCH("MOUNTAIN original", mountain_folder_path, 30, 1)
#%%
plotMeanAndStandardDeviationPCH("MOUNTAIN generated", mountain_output_folder_path, 20, 0)
#%%
#################### ROUTE ####################
plotMeanAndStandardDeviationPCH("ROUTE original", route_folder_path, 90, 1)
#%%
plotMeanAndStandardDeviationPCH("ROUTE generated", route_output_folder_path, 20, 0)
#%%
#################### SEA_OCEAN ####################
plotMeanAndStandardDeviationPCH("SEA_OCEAN original", sea_ocean_folder_path, 12, 1)
#%%
plotMeanAndStandardDeviationPCH("SEA_OCEAN generated", sea_ocean_output_folder_path, 20, 0)
#%%
#################### TOWER ####################
plotMeanAndStandardDeviationPCH("TOWER original", tower_folder_path, 16, 1)
#%%
plotMeanAndStandardDeviationPCH("TOWER generated", tower_output_folder_path, 20, 0)
#%%
all_generated_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/_all/*.mid' ##
all_original_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/_all/*.mid' ##

# Plot Confidence Interval
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], ['C','', 'C#','', 'D','', 'E','', 'Eb','', 'F','', 'F#','', 'G','', 'G#','', 'A','', 'Bb','', 'B',''])
plt.title("Original vs Generated pitch class histogram confidence interval")
generated_pch_matrix = get_pitch_class_histograms_multiple_files("All generated", all_generated_path, 140, 0)
# generated_pch_matrix = generated_pch_matrix / (sum(sum(generated_pch_matrix)))
original_pch_matrix = get_pitch_class_histograms_multiple_files("All original", all_original_path, 226, 1)
# original_pch_matrix = original_pch_matrix / sum(sum(original_pch_matrix))
print(generated_pch_matrix)
print(original_pch_matrix)
# each column of pch_matrix is a note, starting from 0=C, 1=C# ...
print(plot_confidence_interval(1, original_pch_matrix[:,0], mean_color='#228B22'))
print(plot_confidence_interval(2, generated_pch_matrix[:,1]))
print(plot_confidence_interval(3, original_pch_matrix[:,1], mean_color='#228B22'))
print(plot_confidence_interval(4, generated_pch_matrix[:,1]))
print(plot_confidence_interval(5, original_pch_matrix[:,2], mean_color='#228B22'))
print(plot_confidence_interval(6, generated_pch_matrix[:,2]))
print(plot_confidence_interval(7, original_pch_matrix[:,3], mean_color='#228B22'))
print(plot_confidence_interval(8, generated_pch_matrix[:,3]))
print(plot_confidence_interval(9, original_pch_matrix[:,4], mean_color='#228B22'))
print(plot_confidence_interval(10, generated_pch_matrix[:,4]))
print(plot_confidence_interval(11, original_pch_matrix[:,5], mean_color='#228B22'))
print(plot_confidence_interval(12, generated_pch_matrix[:,5]))
print(plot_confidence_interval(13, original_pch_matrix[:,6], mean_color='#228B22'))
print(plot_confidence_interval(14, generated_pch_matrix[:,6]))
print(plot_confidence_interval(15, original_pch_matrix[:,7], mean_color='#228B22'))
print(plot_confidence_interval(16, generated_pch_matrix[:,7]))
print(plot_confidence_interval(17, original_pch_matrix[:,8], mean_color='#228B22'))
print(plot_confidence_interval(18, generated_pch_matrix[:,8]))
print(plot_confidence_interval(19, original_pch_matrix[:,9], mean_color='#228B22'))
print(plot_confidence_interval(20, generated_pch_matrix[:,9]))
print(plot_confidence_interval(21, original_pch_matrix[:,10], mean_color='#228B22'))
print(plot_confidence_interval(22, generated_pch_matrix[:,10]))
print(plot_confidence_interval(23, original_pch_matrix[:,11], mean_color='#228B22'))
print(plot_confidence_interval(24, generated_pch_matrix[:,11]))
plt.show()
# %% Mean and Confidence Interval 
pitch_labels = ['C', 'C#', 'D', 'E', 'Eb', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
all_generated_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/_all/*.mid' ##
all_original_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/_all/*.mid' ##

generated_pch_matrix = get_pitch_class_histograms_multiple_files("All generated", all_generated_path, 140, 0)
original_pch_matrix = get_pitch_class_histograms_multiple_files("All original", all_original_path, 226, 1)

generated_mean = []
generated_CI = []
original_mean = []
original_CI = []

for column in zip(*generated_pch_matrix):
    mean, conf_int = calculate_mean_and_confidence_interval(column)
    generated_mean.append(mean)
    generated_CI.append(conf_int)

for column in zip(*original_pch_matrix):
    mean, conf_int = calculate_mean_and_confidence_interval(column)
    original_mean.append(mean)
    original_CI.append(conf_int)


plot_bar_chart_with_confidence_intervals(original_mean, generated_mean, original_CI, generated_CI, pitch_labels)

# %%

####################### NOTE LENGTH HISTOGRAM #######################

def note_length_hist(path, track_num=1, normalize=False, pause_event=False):
    """
        note_length_hist (Note length histogram):
        To extract the note length histogram, we first define a set of allowable beat length classes:
        [full, half, quarter, 8th, 16th, dot half, dot quarter, dot 8th, dot 16th, half note triplet, quarter note triplet, 8th note triplet].
        The pause_event option, when activated, will double the vector size to represent the same lengths for rests.
        The classification of each event is performed by dividing the basic unit into the length of (barlength)/96, and each note length is quantized to the closest length category.

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).
        'normalize' : If true, normalize by vector sum.
        'pause_event' : when activated, will double the vector size to represent the same lengths for rests.

        Returns:
        'note_length_hist': The output vector has a length of either 12 (or 24 when pause_event is True).
        """
    pattern = midi.read_midifile(path)
    if pause_event is False:
        note_length_hist = np.zeros((12))
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        # basic unit: bar_length/96
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                time_sig = pattern[track_num][i].data
                bar_length = time_sig[track_num] * \
                    resolution * 4 / 2**(time_sig[1])
            elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                if 'time_sig' not in locals():  # set default bar length as 4 beat
                    bar_length = 4 * resolution
                    time_sig = [4, 2, 24, 8]
                unit = bar_length / 96.
                hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit *
                             72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                current_tick = pattern[track_num][i].tick
                current_note = pattern[track_num][i].data[0]
                # find next note off
                for j in range(i, len(pattern[track_num])):
                    if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                        if pattern[track_num][j].data[0] == current_note:

                            note_length = pattern[track_num][j].tick - \
                                current_tick
                            distance = np.abs(
                                np.array(hist_list) - note_length)
                            idx = distance.argmin()
                            note_length_hist[idx] += 1
                            break
    else:
        note_length_hist = np.zeros((24))
        pattern.make_ticks_abs()
        resolution = pattern.resolution
        # basic unit: bar_length/96
        for i in range(0, len(pattern[track_num])):
            if type(pattern[track_num][i]) == midi.events.TimeSignatureEvent:
                time_sig = pattern[track_num][i].data
                bar_length = time_sig[track_num] * \
                    resolution * 4 / 2**(time_sig[1])
            elif type(pattern[track_num][i]) == midi.events.NoteOnEvent and pattern[track_num][i].data[1] != 0:
                check_previous_off = True
                if 'time_sig' not in locals():  # set default bar length as 4 beat
                    bar_length = 4 * resolution
                    time_sig = [4, 2, 24, 8]
                unit = bar_length / 96.
                tol = 3. * unit
                hist_list = [unit * 96, unit * 48, unit * 24, unit * 12, unit * 6, unit *
                             72, unit * 36, unit * 18, unit * 9, unit * 32, unit * 16, unit * 8]
                current_tick = pattern[track_num][i].tick
                current_note = pattern[track_num][i].data[0]
                # find next note off
                for j in range(i, len(pattern[track_num])):
                    # find next note off
                    if type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):
                        if pattern[track_num][j].data[0] == current_note:

                            note_length = pattern[track_num][j].tick - \
                                current_tick
                            distance = np.abs(
                                np.array(hist_list) - note_length)
                            idx = distance.argmin()
                            note_length_hist[idx] += 1
                            break
                        else:
                            if pattern[track_num][j].tick == current_tick:
                                check_previous_off = False

                # find previous note off/on
                if check_previous_off is True:
                    for j in range(i - 1, 0, -1):
                        if type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] != 0:
                            break

                        elif type(pattern[track_num][j]) == midi.events.NoteOffEvent or (type(pattern[track_num][j]) == midi.events.NoteOnEvent and pattern[track_num][j].data[1] == 0):

                            note_length = current_tick - \
                                pattern[track_num][j].tick
                            distance = np.abs(
                                np.array(hist_list) - note_length)
                            idx = distance.argmin()
                            if distance[idx] < tol:
                                note_length_hist[idx + 12] += 1
                            break

    if normalize is False:
        return note_length_hist

    elif normalize is True:

        return note_length_hist / np.sum(note_length_hist)


def plotNoteLengthHistogram(values, title):
    fig, ax = plt.subplots()
    plt.title(title + " Note length histogram")
    plt.xticks(rotation=45)
    notes = ['full', 'half', 'quarter', '8th', '16th', 'dot half', 'dot quarter', 'dot 8th',
             'dot 16th', 'half note triplet80', 'quarter note triplet', '8th note triplet']
    ax.bar(notes, values)
    plt.savefig(title)
    plt.show()


def totalNoteLength(path, track_num=0):
    
    histogram = np.zeros(12)
    for i, file in enumerate(glob.glob(path)):
        values = note_length_hist(file, track_num)
        histogram = np.add(histogram, values)
    return histogram

def plotTwoNotesHist(path1, path2, title, track_num1=1, track_num2=0):
    original = totalNoteLength(path1, track_num1)
    original = original / np.sum(original)
    generated = totalNoteLength(path2, track_num2)
    generated = generated / np.sum(generated)
    notes = ['full', 'half', 'quarter', '8th', '16th', 'dot half', 'dot quarter', 'dot 8th',
             'dot 16th', 'half note triplet80', 'quarter note triplet', '8th note triplet']
    x_axis = np.arange(len(notes))
    plt.bar(x_axis - 0.2, original, width=0.4, label='Original')
    plt.bar(x_axis + 0.2, generated, width=0.4, label='Generated')    
    plt.title(title + " Note length histogram")
    plt.xticks(x_axis, notes, rotation=45)
    plt.legend(loc='upper right')
    plt.savefig(title + " Note length histogram")
    plt.show()

# For measuring and plotting mean and standard deviation

def buildSTDPlotNLH(xpos, CTEs, error, title):
    # Build the plot
    notes = ['full', 'half', 'quarter', '8th', '16th', 'dot half', 'dot quarter', 'dot 8th',
         'dot 16th', 'half note triplet80', 'quarter note triplet', '8th note triplet']
    
    fig, ax = plt.subplots()
    ax.bar(xpos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    # ax.set_ylabel()
    ax.set_xticks(xpos)
    ax.set_xticklabels(notes)
    ax.set_title(title)
    ax.yaxis.grid(True)
    plt.xticks(rotation=45)
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(title)
    plt.show()

def plotMeanAndStandardDeviationNLH(title, path, number_of_files, track_num=0):
    # Calculate note histogram for each file, save in a matrix
    nlh_matrix = np.zeros((number_of_files, 12))
    for i, file in enumerate(glob.glob(path)): # note lengths values for each file
        values = note_length_hist(file, track_num)
        nlh_matrix[i] += values
    # print(nlh_matrix)

    # Calculate mean for each note length from the matrix (each column)
    nlh_means_array = nlh_matrix.mean(0)
    # print(nlh_means_array)    

    # Calculate standard deviation for each note length from matrix (each column)
    nlh_std_array = nlh_matrix.std(0)
    # print(nlh_std_array)
    
    x_pos = np.arange(len(nlh_matrix[0]))
    CTEs = nlh_means_array
    error = nlh_std_array
    plot_title = title + " note length mean and standard deviation"
    buildSTDPlotNLH(x_pos, CTEs, error, plot_title)
    plt.show()
    plt.savefig(title)

def get_note_length_histograms_multiple_files(title, path, number_of_files, track_num=0):
    """
        Calculate note histogram for each file, save in a matrix
    """
    nlh_matrix = np.zeros((number_of_files, 12))
    for i, file in enumerate(glob.glob(path)): # note lengths values for each file
        values = note_length_hist(file, track_num)
        nlh_matrix[i] += values
    return nlh_matrix

#################### Old, used for battle, ow, mixed ####################

# plotTwoNotesHist(battle_tracks_path, battle_output_path, "Battle")
# plotTwoNotesHist(ow_tracks_path, ow_output_path, "Overworld")
# plotTwoNotesHist(mixed_tracks_path, mixed_output_path, "Mixed")


########################################################################
#################### New, used for biome generation ####################
########################################################################
#%%
#################### CAVE ####################
plotTwoNotesHist(cave_folder_path, cave_output_folder_path, "CAVE")
#%%
#################### CITY ####################
plotTwoNotesHist(city_folder_path, city_output_folder_path, "CITY")
#%%
#################### FOREST ####################
plotTwoNotesHist(forest_folder_path, forest_output_folder_path, "FOREST")
#%%
#################### MOUNTAIN ####################
plotTwoNotesHist(mountain_folder_path, mountain_output_folder_path, "MOUNTAIN")
#%%
#################### ROUTE ####################
plotTwoNotesHist(route_folder_path, route_output_folder_path, "ROUTE")
#%%
#################### SEA_OCEAN ####################
plotTwoNotesHist(sea_ocean_folder_path, sea_ocean_output_folder_path, "SEA_OCEAN")
#%%
#################### TOWER ####################
plotTwoNotesHist(tower_folder_path, tower_output_folder_path, "TOWER")

#%%
#####################################################################
#################### Mean and Standard Deviation ####################
#####################################################################
#%%
#################### CAVE ####################
plotMeanAndStandardDeviationNLH("CAVE original", cave_folder_path, 19, 1)
#%%
plotMeanAndStandardDeviationNLH("CAVE generated", cave_output_folder_path, 20, 0)
#%%
#################### CITY ####################
plotMeanAndStandardDeviationNLH("CITY original", city_folder_path, 89, 1)
#%%
plotMeanAndStandardDeviationNLH("CITY generated", city_output_folder_path, 20, 0)
#%%
#################### FOREST ####################
plotMeanAndStandardDeviationNLH("FOREST original", forest_folder_path, 22, 1)
#%%
plotMeanAndStandardDeviationNLH("FOREST generated", forest_output_folder_path, 20, 0)
#%%
#################### MOUNTAIN ####################
plotMeanAndStandardDeviationNLH("MOUNTAIN original", mountain_folder_path, 30, 1)
#%%
plotMeanAndStandardDeviationNLH("MOUNTAIN generated", mountain_output_folder_path, 20, 0)
#%%
#################### ROUTE ####################
plotMeanAndStandardDeviationNLH("ROUTE original", route_folder_path, 90, 1)
#%%
plotMeanAndStandardDeviationNLH("ROUTE generated", route_output_folder_path, 20, 0)
#%%
#################### SEA_OCEAN ####################
plotMeanAndStandardDeviationNLH("SEA_OCEAN original", sea_ocean_folder_path, 12, 1)
#%%
plotMeanAndStandardDeviationNLH("SEA_OCEAN generated", sea_ocean_output_folder_path, 20, 0)
#%%
#################### TOWER ####################
plotMeanAndStandardDeviationNLH("TOWER original", tower_folder_path, 16, 1)
#%%
plotMeanAndStandardDeviationNLH("TOWER generated", tower_output_folder_path, 20, 0)
#%%
all_generated_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/_all/*.mid' ##
all_original_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/_all/*.mid' ##

# Plot Confidence Interval
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], 
['full', '', 'half', '', 'quarter', '', '8th', '', '16th', '', 'dot half', '', 'dot quarter', '', 'dot 8th', '',
         'dot 16th', '', 'half note triplet80', '', 'quarter note triplet', '', '8th note triplet', ''])
plt.title("Original vs Generated note length histogram confidence interval")
generated_nlh_matrix = get_note_length_histograms_multiple_files("All generated", all_generated_path, 140, 0)
generated_nlh_matrix = generated_nlh_matrix / (sum(sum(generated_nlh_matrix)))
original_nlh_matrix = get_note_length_histograms_multiple_files("All original", all_original_path, 226, 1)
# original_nlh_matrix = original_nlh_matrix / sum(sum(original_nlh_matrix))
# each column of pch_matrix is a note, starting from 0=C, 1=C# ...
print(plot_confidence_interval(1, original_nlh_matrix[:,0], mean_color='#228B22'))
print(plot_confidence_interval(2, generated_nlh_matrix[:,1]))
print(plot_confidence_interval(3, original_nlh_matrix[:,1], mean_color='#228B22'))
print(plot_confidence_interval(4, generated_nlh_matrix[:,1]))
print(plot_confidence_interval(5, original_nlh_matrix[:,2], mean_color='#228B22'))
print(plot_confidence_interval(6, generated_nlh_matrix[:,2]))
print(plot_confidence_interval(7, original_nlh_matrix[:,3], mean_color='#228B22'))
print(plot_confidence_interval(8, generated_nlh_matrix[:,3]))
print(plot_confidence_interval(9, original_nlh_matrix[:,4], mean_color='#228B22'))
print(plot_confidence_interval(10, generated_nlh_matrix[:,4]))
print(plot_confidence_interval(11, original_nlh_matrix[:,5], mean_color='#228B22'))
print(plot_confidence_interval(12, generated_nlh_matrix[:,5]))
print(plot_confidence_interval(13, original_nlh_matrix[:,6], mean_color='#228B22'))
print(plot_confidence_interval(14, generated_nlh_matrix[:,6]))
print(plot_confidence_interval(15, original_nlh_matrix[:,7], mean_color='#228B22'))
print(plot_confidence_interval(16, generated_nlh_matrix[:,7]))
print(plot_confidence_interval(17, original_nlh_matrix[:,8], mean_color='#228B22'))
print(plot_confidence_interval(18, generated_nlh_matrix[:,8]))
print(plot_confidence_interval(19, original_nlh_matrix[:,9], mean_color='#228B22'))
print(plot_confidence_interval(20, generated_nlh_matrix[:,9]))
print(plot_confidence_interval(21, original_nlh_matrix[:,10], mean_color='#228B22'))
print(plot_confidence_interval(22, generated_nlh_matrix[:,10]))
print(plot_confidence_interval(23, original_nlh_matrix[:,11], mean_color='#228B22'))
print(plot_confidence_interval(24, generated_nlh_matrix[:,11]))
plt.xticks(rotation=90)
plt.show()

# %% Mean and Confidence Interval 
pitch_labels = ['full', 'half', 'quarter', '8th', '16th', 'dot half', 'dot quarter', 'dot 8th',
         'dot 16th', 'half note triplet80', 'quarter note triplet', '8th note triplet']
all_generated_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/generated/_all/*.mid' ##
all_original_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/original/_all/*.mid' ##

generated_nlh_matrix = get_note_length_histograms_multiple_files("All generated", all_generated_path, 140, 0)
generated_nlh_matrix = generated_nlh_matrix / (sum(sum(generated_nlh_matrix)))
original_nlh_matrix = get_note_length_histograms_multiple_files("All original", all_original_path, 226, 1)

generated_mean = []
generated_CI = []
original_mean = []
original_CI = []

for column in zip(*generated_nlh_matrix):
    mean, conf_int = calculate_mean_and_confidence_interval(column)
    generated_mean.append(mean)
    generated_CI.append(conf_int)

for column in zip(*original_nlh_matrix):
    mean, conf_int = calculate_mean_and_confidence_interval(column)
    original_mean.append(mean)
    original_CI.append(conf_int)


plot_bar_chart_with_confidence_intervals(original_mean, generated_mean, original_CI, generated_CI, pitch_labels, "Note lengths")


#%%

# generated_battle_path = "C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/battle_output/*.mid"
# generated_mixed_path = "C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/mixed_output/*.mid"
# generated_ow_path = "C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/ow_output/*.mid"
# original_battle_path = "C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/EDIT_DISTANCE_COMPARISON/original_battle_edit_distance/*.mid"
# original_mixed_path = "C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/EDIT_DISTANCE_COMPARISON/original_mixed_edit_distance/*.mid"
# original_ow_path = "C:/Facultate/IDG_Master/_Sem2/pokemon_assets_generation/pokemon_tracks/EDIT_DISTANCE_COMPARISON/original_ow_edit_distance/*.mid"


# Paths for biomes
## INPUT FOLDERS
ed_cave_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/original/cave/*.mid' ## 
ed_city_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/original/city/*.mid' ## 
ed_forest_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/original/forest/*.mid' ##
ed_mountain_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/original/mountain/*.mid' ##
ed_route_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/original/route/*.mid' ##
ed_sea_ocean_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/original/sea_ocean/*.mid' ## 
ed_tower_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/original/tower/*.mid' ## 

## OUTPUT FOLDERS
ed_cave_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/generated/cave_output/*.mid' ##
ed_city_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/generated/city_output/*.mid' ##
ed_forest_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/generated/forest_output/*.mid' ##
ed_mountain_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/generated/mountain_output/*.mid' ##
ed_route_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/generated/route_output/*.mid' ##
ed_sea_ocean_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/generated/sea_ocean_output/*.mid' ##
ed_tower_output_folder_path = 'D:/IDG_Malta_Master_Thesis/pokemon_assets_generation/pokemon_tracks/biome_tracks/_edit_distance/generated/tower_output/*.mid' ##

####################### EDIT DISTANCE #######################
def compareTwoMidis(path1, path2):
    scoreGroundTruth = music21.converter.parse(path1)
    scoreOMR = music21.converter.parse(path2)
    omrGTP = omr.evaluators.OmrGroundTruthPair(omr=scoreOMR, ground=scoreGroundTruth)
    differences = omrGTP.getDifferences()
    return differences

def computeMinEditDistanceForTwoSets(path1, path2):
    minForEachList = {}
    for i,file in enumerate(glob.glob(path1)):
        print('\r', file, end='')
        scoreGroundTruth = music21.converter.parse(file)
        minDif = sys.maxsize
        songName = "0"
        for j,filee in enumerate(glob.glob(path2)):
            # print('\r', filee, end='')
            if(file == filee):
                continue
            scoreOMR = music21.converter.parse(filee)
            omrGTP = omr.evaluators.OmrGroundTruthPair(omr=filee, ground=file)
            differences = omrGTP.getDifferences()
            if(differences < minDif):
                minDif = differences
                songName = filee
        minForEachList[file] = (differences, songName)
    return minForEachList

# def computeMinEditDistanceForTwoSetsMethodTwo(path1, path2):
#     minForEachList = {}
#     for i,path1_file in enumerate(glob.glob(path1)):
#         minDif = sys.maxsize
#         songName = "0"
#         for j,path2_file in enumerate(glob.glob(path2)):
#             if(path1_file == path2_file):
#                 continue
#             omrGTP = omr.evaluators.OmrGroundTruthPair(omr=path2_file, ground=path1_file)
#             differences = omrGTP.getDifferences()
#             if(differences < minDif):
#                 minDif = differences
#                 songName = path2_file
#         minForEachList[path1_file] = (differences, songName)
#     return minForEachList

def computeAverageMinEditDistance(path1, path2, size=20):
    minForEachList = computeMinEditDistanceForTwoSets(path1, path2)
    min_edit_distance_values_list = np.zeros(size)
    sum = 0
    i = 0
    for value in minForEachList.values():
        # print(value[0])
        sum += value[0]
        min_edit_distance_values_list[i] = value[0]
        i += 1
    avgMinEditDist = sum/len(minForEachList)
    print(min_edit_distance_values_list)
    return (avgMinEditDist, min_edit_distance_values_list)

def computeStandardDeviation(list):
    standardDeviation = np.std(list)
    return standardDeviation

# #%%
# battleMinDistBetweenOriginal = computeMinEditDistanceForTwoSets(original_battle_path, original_battle_path)
# print("\n\n")
# print("ORIGINAL BATTLE MINIMUM EDIT DISTANCE")
# print(battleMinDistBetweenOriginal)
# print("\n\n")
# #%%
# owMinDistBetweenOriginal = computeMinEditDistanceForTwoSets(original_ow_path, original_ow_path)
# #%%
# print("ORIGINAL OVERWORLD MINIMUM EDIT DISTANCE")
# print(owMinDistBetweenOriginal)
# print("\n\n")
# #%%
# mixedMinDistBetweenOriginal = computeMinEditDistanceForTwoSets(original_mixed_path, original_mixed_path)
# #%%
# print("ORIGINAL MIXED MINIMUM EDIT DISTANCE")
# print(mixedMinDistBetweenOriginal)
# print("\n\n")

# #%%
# minEdDistBattle = computeMinEditDistanceForTwoSets(generated_battle_path, original_battle_path)
# #%%
# print("ORIGINAL BATTLE VS GENERATED BATTLE MINIMUM EDIT DISTANCE")
# print(minEdDistBattle)
# print("\n\n")
# #%%
# minEdDistOw = computeMinEditDistanceForTwoSets(generated_ow_path, original_ow_path)
# #%%
# print("ORIGINAL OVERWORLD VS GENERATED OW MINIMUM EDIT DISTANCE")
# print(minEdDistOw)
# print("\n\n")
# #%%
# minEdDistMixed = computeMinEditDistanceForTwoSets(generated_mixed_path, original_battle_path)
# #%%
# print("ORIGINAL BATTLE VS GENERATED MIXED MINIMUM EDIT DISTANCE")
# print(minEdDistMixed)
# print("\n\n")
# #%%
# minEdDistMixedTwo = computeMinEditDistanceForTwoSets(generated_mixed_path, original_ow_path)
# #%%
# print("ORIGINAL OW VS GENERATED MIXED MINIMUM EDIT DISTANCE")
# print(minEdDistMixedTwo)
# print("\n\n")

#%%
#testing
result = computeAverageMinEditDistance(ed_cave_output_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

########################################################################
#################### New, used for biome generation ####################
########################################################################

# %%
#################### Average minimum edit distance ####################
#################### This is in between generated tracks ####################
#################### CAVE vs REST ####################

print("\nCave vs Cave: \n")
result = computeAverageMinEditDistance(ed_cave_output_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs City: ")
result = computeAverageMinEditDistance(ed_cave_output_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs Forest: ")
result = computeAverageMinEditDistance(ed_cave_output_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs Mountain: ")
result = computeAverageMinEditDistance(ed_cave_output_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs Route: ")
result = computeAverageMinEditDistance(ed_cave_output_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs Sea: ")
result = computeAverageMinEditDistance(ed_cave_output_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs Tower: ")
result = computeAverageMinEditDistance(ed_cave_output_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### CITY vs REST ####################

print("\nCity vs City: ")
result = computeAverageMinEditDistance(ed_city_output_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Cave: ")
result = computeAverageMinEditDistance(ed_city_output_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Forest Average Minimum Edit Distance: ")
result = computeAverageMinEditDistance(ed_city_output_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Mountain: ")
result = computeAverageMinEditDistance(ed_city_output_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Route: ")
result = computeAverageMinEditDistance(ed_city_output_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Sea: ")
result = computeAverageMinEditDistance(ed_city_output_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Tower: ")
result = computeAverageMinEditDistance(ed_city_output_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### FOREST vs REST ####################

print("\nForest vs Forest: ")
result = computeAverageMinEditDistance(ed_forest_output_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs Cave: ")
result = computeAverageMinEditDistance(ed_forest_output_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs City: ")
result = computeAverageMinEditDistance(ed_forest_output_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs Mountain: ")
result = computeAverageMinEditDistance(ed_forest_output_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs Route: ")
result = computeAverageMinEditDistance(ed_forest_output_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs Sea: ")
result = computeAverageMinEditDistance(ed_forest_output_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs Tower: ")
result = computeAverageMinEditDistance(ed_forest_output_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### MOUNTAIN vs REST ####################

print("\nMountain vs Mountain: ")
result = computeAverageMinEditDistance(ed_mountain_output_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs Cave: ")
result = computeAverageMinEditDistance(ed_mountain_output_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs City: ")
result = computeAverageMinEditDistance(ed_mountain_output_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs Forest: ")
result = computeAverageMinEditDistance(ed_mountain_output_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs Route: ")
result = computeAverageMinEditDistance(ed_mountain_output_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs Sea: ")
result = computeAverageMinEditDistance(ed_mountain_output_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs Tower: ")
result = computeAverageMinEditDistance(ed_mountain_output_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### ROUTE vs REST ####################

print("\nRoute vs Route: ")
result = computeAverageMinEditDistance(ed_route_output_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs Cave: ")
result = computeAverageMinEditDistance(ed_route_output_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs City: ")
result = computeAverageMinEditDistance(ed_route_output_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs Forest: ")
result = computeAverageMinEditDistance(ed_route_output_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs Mountain: ")
result = computeAverageMinEditDistance(ed_route_output_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs Sea: ")
result = computeAverageMinEditDistance(ed_route_output_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs Tower: ")
result = computeAverageMinEditDistance(ed_route_output_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### SEA vs REST ####################

print("\nSea vs Sea: ")
result = computeAverageMinEditDistance(ed_sea_ocean_output_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs Cave: ")
result = computeAverageMinEditDistance(ed_sea_ocean_output_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs City: ")
result = computeAverageMinEditDistance(ed_sea_ocean_output_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs Forest: ")
result = computeAverageMinEditDistance(ed_sea_ocean_output_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs Mountain: ")
result = computeAverageMinEditDistance(ed_sea_ocean_output_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs Route: ")
result = computeAverageMinEditDistance(ed_sea_ocean_output_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs Tower: ")
result = computeAverageMinEditDistance(ed_sea_ocean_output_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### TOWER vs REST ####################

print("\nTower vs Tower: ")
result = computeAverageMinEditDistance(ed_tower_output_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)
print("\nTower vs Cave: ")
result = computeAverageMinEditDistance(ed_tower_output_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nTower vs City: ")
result = computeAverageMinEditDistance(ed_tower_output_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nTower vs Forest: ")
result = computeAverageMinEditDistance(ed_tower_output_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nTower vs Mountain: ")
result = computeAverageMinEditDistance(ed_tower_output_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nTower vs Route: ")
result = computeAverageMinEditDistance(ed_tower_output_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nTower vs Sea: ")
result = computeAverageMinEditDistance(ed_tower_output_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
# Plot Confidence Interval
plt.xticks([1,2,3,4,5,6,7], ["Cave", "City", "Forest", "Mountain", "Route", "Sea", "Tower"])
plt.title("Generated min edit distance confidence interval")
result = computeAverageMinEditDistance(ed_cave_output_folder_path, ed_cave_output_folder_path)
cave_list = result[1]
result = computeAverageMinEditDistance(ed_city_output_folder_path, ed_city_output_folder_path)
city_list = result[1]
result = computeAverageMinEditDistance(ed_forest_output_folder_path, ed_forest_output_folder_path)
forest_list = result[1]
result = computeAverageMinEditDistance(ed_mountain_output_folder_path, ed_mountain_output_folder_path)
mountain_list = result[1]
result = computeAverageMinEditDistance(ed_route_output_folder_path, ed_route_output_folder_path)
route_list = result[1]
result = computeAverageMinEditDistance(ed_sea_ocean_output_folder_path, ed_sea_ocean_output_folder_path)
sea_list = result[1]
result = computeAverageMinEditDistance(ed_tower_output_folder_path, ed_tower_output_folder_path)
tower_list = result[1]
plot_confidence_interval(1, cave_list)
plot_confidence_interval(2, city_list)
plot_confidence_interval(3, forest_list)
plot_confidence_interval(4, mountain_list)
plot_confidence_interval(5, route_list)
plot_confidence_interval(6, sea_list)
plot_confidence_interval(7, tower_list)
plt.show()
# %%

#################### This is between original and generated ####################
#################### CAVE VS REST
print("\nCave vs Cave: \n")
result = computeAverageMinEditDistance(ed_cave_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs City: ")
result = computeAverageMinEditDistance(ed_cave_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs Forest: ")
result = computeAverageMinEditDistance(ed_cave_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs Mountain: ")
result = computeAverageMinEditDistance(ed_cave_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs Route: ")
result = computeAverageMinEditDistance(ed_cave_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs Sea: ")
result = computeAverageMinEditDistance(ed_cave_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCave vs Tower: ")
result = computeAverageMinEditDistance(ed_cave_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### CITY vs REST ####################

print("\nCity vs City: ")
result = computeAverageMinEditDistance(ed_city_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Cave: ")
result = computeAverageMinEditDistance(ed_city_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Forest: ")
result = computeAverageMinEditDistance(ed_city_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Mountain: ")
result = computeAverageMinEditDistance(ed_city_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Route: ")
result = computeAverageMinEditDistance(ed_city_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Sea: ")
result = computeAverageMinEditDistance(ed_city_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nCity vs Tower: ")
result = computeAverageMinEditDistance(ed_city_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### FOREST vs REST ####################

print("\nForest vs Forest: ")
result = computeAverageMinEditDistance(ed_forest_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs Cave: ")
result = computeAverageMinEditDistance(ed_forest_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs City: ")
result = computeAverageMinEditDistance(ed_forest_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs Mountain: ")
result = computeAverageMinEditDistance(ed_forest_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs Route: ")
result = computeAverageMinEditDistance(ed_forest_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs Sea: ")
result = computeAverageMinEditDistance(ed_forest_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nForest vs Tower: ")
result = computeAverageMinEditDistance(ed_forest_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### MOUNTAIN vs REST ####################

print("\nMountain vs Mountain: ")
result = computeAverageMinEditDistance(ed_mountain_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs Cave: ")
result = computeAverageMinEditDistance(ed_mountain_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs City: ")
result = computeAverageMinEditDistance(ed_mountain_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs Forest: ")
result = computeAverageMinEditDistance(ed_mountain_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs Route: ")
result = computeAverageMinEditDistance(ed_mountain_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs Sea: ")
result = computeAverageMinEditDistance(ed_mountain_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nMountain vs Tower: ")
result = computeAverageMinEditDistance(ed_mountain_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### ROUTE vs REST ####################

print("\nRoute vs Route: ")
result = computeAverageMinEditDistance(ed_route_folder_path, ed_route_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs Cave: ")
result = computeAverageMinEditDistance(ed_route_folder_path, ed_cave_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs City: ")
result = computeAverageMinEditDistance(ed_route_folder_path, ed_city_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs Forest: ")
result = computeAverageMinEditDistance(ed_route_folder_path, ed_forest_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs Mountain: ")
result = computeAverageMinEditDistance(ed_route_folder_path, ed_mountain_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs Sea: ")
result = computeAverageMinEditDistance(ed_route_folder_path, ed_sea_ocean_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nRoute vs Tower: ")
result = computeAverageMinEditDistance(ed_route_folder_path, ed_tower_output_folder_path)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### SEA vs REST ####################

print("\nSea vs Sea: ")
result = computeAverageMinEditDistance(ed_sea_ocean_folder_path, ed_sea_ocean_output_folder_path, size=12)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs Cave: ")
result = computeAverageMinEditDistance(ed_sea_ocean_folder_path, ed_cave_output_folder_path, size=12)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs City: ")
result = computeAverageMinEditDistance(ed_sea_ocean_folder_path, ed_city_output_folder_path, size=12)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs Forest: ")
result = computeAverageMinEditDistance(ed_sea_ocean_folder_path, ed_forest_output_folder_path, size=12)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs Mountain: ")
result = computeAverageMinEditDistance(ed_sea_ocean_folder_path, ed_mountain_output_folder_path, size=12)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs Route: ")
result = computeAverageMinEditDistance(ed_sea_ocean_folder_path, ed_route_output_folder_path, size=12)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nSea vs Tower: ")
result = computeAverageMinEditDistance(ed_sea_ocean_folder_path, ed_tower_output_folder_path, size=12)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
#################### TOWER vs REST ####################

print("\nTower vs Tower: ")
result = computeAverageMinEditDistance(ed_tower_folder_path, ed_tower_output_folder_path, size=16)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)
print("\nTower vs Cave: ")
result = computeAverageMinEditDistance(ed_tower_folder_path, ed_cave_output_folder_path, size=16)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nTower vs City: ")
result = computeAverageMinEditDistance(ed_tower_folder_path, ed_city_output_folder_path, size=16)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nTower vs Forest: ")
result = computeAverageMinEditDistance(ed_tower_folder_path, ed_forest_output_folder_path, size=16)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nTower vs Mountain: ")
result = computeAverageMinEditDistance(ed_tower_folder_path, ed_mountain_output_folder_path, size=16)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nTower vs Route: ")
result = computeAverageMinEditDistance(ed_tower_folder_path, ed_route_output_folder_path, size=16)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

print("\nTower vs Sea: ")
result = computeAverageMinEditDistance(ed_tower_folder_path, ed_sea_ocean_output_folder_path, size=16)
value = result[0]
list = result[1]
min_edit_distance_standard_deviation = np.std(list)
confidence_interval = mean_confidence_interval(list)

print("Average minimum edit distance: ")
print(value)
print("Standard deviation: ")
print(min_edit_distance_standard_deviation)
print("Confidence interval: ")
print(confidence_interval)

# %%
# Plot Confidence Interval
plt.xticks([1,2,3,4,5,6,7], ["Cave", "City", "Forest", "Mountain", "Route", "Sea", "Tower"])
plt.title("Original vs Generated min edit distance confidence interval")
result = computeAverageMinEditDistance(ed_cave_folder_path, ed_cave_output_folder_path)
cave_list = result[1]
result = computeAverageMinEditDistance(ed_city_folder_path, ed_city_output_folder_path)
city_list = result[1]
result = computeAverageMinEditDistance(ed_forest_folder_path, ed_forest_output_folder_path)
forest_list = result[1]
result = computeAverageMinEditDistance(ed_mountain_folder_path, ed_mountain_output_folder_path)
mountain_list = result[1]
result = computeAverageMinEditDistance(ed_route_folder_path, ed_route_output_folder_path)
route_list = result[1]
result = computeAverageMinEditDistance(ed_sea_ocean_folder_path, ed_sea_ocean_output_folder_path, size=12)
sea_list = result[1]
result = computeAverageMinEditDistance(ed_tower_folder_path, ed_tower_output_folder_path, size=16)
tower_list = result[1]
plot_confidence_interval(1, cave_list)
plot_confidence_interval(2, city_list)
plot_confidence_interval(3, forest_list)
plot_confidence_interval(4, mountain_list)
plot_confidence_interval(5, route_list)
plot_confidence_interval(6, sea_list)
plot_confidence_interval(7, tower_list)
plt.show()
# %%
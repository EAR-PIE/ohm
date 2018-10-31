# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 20:46:29 2018

@author: samla
"""

from music21 import note,

#assign Note Object to variable
f = note.Note('F5')
#get Note Object name and octave
f.name; f.octave;

#returns a Pitch Object
f.pitch
#returns the frequency in Hz of the Pitch Object
f.pitch.frequency

#assign a Note Object with an Accidental to variable (sharp = '#' and flat = '-')
b_flat = note.Note('B-2')
#get type of Accidental with...
b_flat.pitch.accidental
#get value of Accidental with...
b_flat.pitch.accidental.alter

#show staff
f.show()
#play midi
f.show('midi')

#transpose notes using .transpose and keyworded strings ('M3' = 'major third', etc.)
d = b_flat.transpose('M3', inplace=False)

#assign Rest Object to variable (where type = 'whole', 'half', 'quarter', etc...)
r = note.Rest(type='whole')

#Duration Objects:
half_duration = duration.Duration('half')
dotted_quarter = duration.Duration(1.5)
half_duration.quarterLength
half_duration.type
half_duration.dots
dotted_quarter.dots = 2
dotted_quarter.quarterLength #== 1.75

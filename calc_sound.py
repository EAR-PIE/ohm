#assign A4 in Hz
f_of_0 = 440.0 #the frequency of one fixed note.
#to calculate C4 in Hz
n = 3 #number of half-steps away from fixed note --- if higher, than positive; if lower, than negative.
a = 2**(1/12) #the twelth root of 2.
#assign C4 in Hz value
f_of_n = f_of_0 * a**n #the frequency of the note n half-steps away from the one fixed note.


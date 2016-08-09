import random
import numpy as np
from random import randint
from itertools import groupby
from sets import Set


GESTURE_RECOGNITION_PROBABILITY = 0.2  # taken from classification accuracy
LEARNING_ITERATIONS = 500

iteration = 0

def myround(a, decimals=1):
     return np.around(a-10**(-(decimals+5)), decimals=decimals)

def decision(probability):
    return random.random() < probability

gestures = ["gesture_hug", "gesture_hit", "gesture_shit", "gesture_sit", "gesture_meat"]
reactions = ["hug_back", "evade_attack", "shit_back", "sit_back", "meat_back"]
color_feedback = ["green", "red"]

confidence_table=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]  #1st row is for gesture_hug, 1st column hug_back

history = []
a_list = []
for i in range(0, len(gestures)*len(gestures)):
    a_list.append(0.0)
history.append(a_list)

print history


#DEMONSTRATION ON DEMONSTRATOR SIDE

for i in range(1, LEARNING_ITERATIONS):

    iteration = iteration + 1

    next_data_instance = "" 			#perceived by learner
    next_data_instance_on_board = ""		#actual on demonstrator side
    if(decision(GESTURE_RECOGNITION_PROBABILITY)):
        gesture_temp = gestures[randint(0,len(gestures)-1)]
        next_data_instance_on_board = gesture_temp
        next_data_instance = gesture_temp
    else:
        next_data_instance_on_board = gestures[randint(0,len(gestures)-1)]
        next_data_instance = gestures[randint(0,len(gestures)-1)]


    print "demonstrator shows", next_data_instance, "the actual gesture is", next_data_instance_on_board


    #reaction
    reaction_on_board = (-1,-1,"",[])		#perceived by demonstrator
    reaction = (-1,-1,"",[])			#actual reaction
    for idx, a_gesture in enumerate(gestures):
        if(idx == gestures.index(next_data_instance)):
            if [len(list(group)) for key, group in groupby(confidence_table[idx])][0] == len(confidence_table[0]):
                print "all items are the same"
                index = randint(0,len(gestures)-1)
                rest = range(0, len(confidence_table[0]))
                rest = [x for x in rest if x != index]
                reaction_on_board = (idx,index,reactions[index],rest)
                reaction = (idx,index,reactions[index],rest)  #select with probability
            else:
                print "not all items are the same"
                index = confidence_table[idx].index(max(confidence_table[idx]))  #selecting max confidence
                maximum_confidence = confidence_table[idx][index]

                idxes = [i for i,x in enumerate(confidence_table[idx]) if x == confidence_table[idx][index]]  #those that repeat

                if len(idxes) == 1: #only 1 repeat
                    rest = range(0, len(confidence_table[0]))
                    rest = [x for x in rest if x!=index]
                    #reaction = (idx,index,reactions[index],rest) #select with probability
                    reaction_on_board = (idx,index,reactions[index],rest)
                    pos = {}
                    for idx2, reaction in enumerate(reactions):
                        if(idx2 == index):
                            pos[reaction] = int(GESTURE_RECOGNITION_PROBABILITY*1000)
                        else:
                            pos[reaction] = int((1000-GESTURE_RECOGNITION_PROBABILITY*1000)/(len(reactions)-1))
                    print pos
                    choice = random.choice([x for x in pos for y in range(pos[x])])
                    reaction = (idx,index,choice,rest) #select with probability

                else:  
                    #here we have a list of repeting max confidences and their positions in list
                    id_ = random.choice(idxes)
                    rest = range(0, len(confidence_table[0]))
                    rest = [x for x in rest if x != id_]
                    #reaction = (idx,id_,reactions[id_],rest) #select with probability
                    reaction_on_board = (idx,id_,reactions[id_],rest)
                    pos = {}
                    for idx2, reaction in enumerate(reactions):
                        if(idx2 in idxes):
                            pos[reaction] = int(GESTURE_RECOGNITION_PROBABILITY*1000)
                        else:
                            pos[reaction] = int((1000-GESTURE_RECOGNITION_PROBABILITY*1000)/(len(reactions)-1))
                    print pos
                    choice = random.choice([x for x in pos for y in range(pos[x])])
                    reaction = (idx,index,choice,rest) #select with probability

    print reaction, "while reaction on board", reaction_on_board


    feedback = ""
    feedback = color_feedback[0] if gestures.index(next_data_instance_on_board) == reactions.index(reaction[2]) else color_feedback[1]
    
    print "demonstrator feedbacks", feedback

    if(feedback == "green"):
        confidence_table[reaction[0]][reaction[1]] = confidence_table[reaction[0]][reaction[1]] + 1
        for item in reaction[3]:
            confidence_table[reaction[0]][item] = confidence_table[reaction[0]][item] - 1
    else:
        confidence_table[reaction[0]][reaction[1]] = confidence_table[reaction[0]][reaction[1]] - 1
        for item in reaction[3]:
            confidence_table[reaction[0]][item] = confidence_table[reaction[0]][item] + 1

    print "iteration", iteration
    history.append(np.array(confidence_table).flatten())
    
    print np.array(confidence_table)
    print np.asarray(history)
    #confidence_table3 = np.array(confidence_table) + iteration
    #confidence_table2 = np.array(confidence_table)
    #confidence_table2 = confidence_table2 / confidence_table2.astype(np.float).sum(axis=1)
    #print myround(confidence_table2, 5)
    #confidence_table3 = confidence_table3 / confidence_table3.astype(np.float).sum(axis=1)
    #print myround(confidence_table3, 5)
    #diagonal = np.array(confidence_table).diagonal()
    #n_correct = np.trace(np.array(confidence_table))
    #accuracy = n_correct*100.0/iteration
    #print "accuracy", accuracy
    #confidence_table = confidence_table.tolist()
    #if(sum(i > 90 for i in diagonal) == len(reactions)):
    #    print "iteration", iteration
    #    break

    np.savetxt('test1.out', np.asarray(history)[:,0:5], delimiter=',')
    np.savetxt('test2.out', np.asarray(history)[:,5:10], delimiter=',')
    np.savetxt('test3.out', np.asarray(history)[:,10:15], delimiter=',')
    np.savetxt('test4.out', np.asarray(history)[:,15:20], delimiter=',')
    np.savetxt('test5.out', np.asarray(history)[:,20:25], delimiter=',')

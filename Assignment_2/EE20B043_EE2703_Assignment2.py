"""
        			EE2703:Applied Programming Lab
      				    Assignment 2: Solution
Name  :Harisankar K J
RollNo:EE20B043
"""
from msilib.schema import Component
import sys
import cmath
import math
import numpy as np

PI = np.pi
INFINITY = 1e99
ZERO = 1e-99
CIRCUIT = ".circuit"  # defining for later use
END = ".end"


class componenets:  # defining a class components to store all the token values
    def __init__(self, line):

        token = line.split()
        if(len(token) == 4):
            self.name = token[0]
            self.node1 = token[1]
            self.node2 = token[2]
            self.value = float(token[3])
            self.type = token[0][0]

        elif(len(token) == 6) and token[3] == 'ac':
            self.name = token[0]
            self.node1 = token[1]
            self.node2 = token[2]
            self.value = float(token[4])/2
            self.type = token[0][0]
            self.phase = float(token[5])

        elif(len(token) == 5) and token[3] == 'dc':
            self.name = token[0]
            self.node1 = token[1]
            self.node2 = token[2]
            self.value = float(token[4])
            self.type = token[0][0]

        # including vcxs
        elif len(token) == 6 and (token[0][0] == "G" or token[0][0] == "E"):
            self.name = token[0]
            self.node1 = token[1]
            self.node2 = token[2]
            self.type = token[0][0]
            self.VC_node1 = token[3]
            self.VC_node2 = token[4]
            self.value = float(token[5])

        # including ccxs
        elif len(token) == 5 and (token[0][0] == "H" or token[0][0] == "F"):
            self.name = token[0]
            self.node1 = token[1]
            self.node2 = token[2]
            self.type = token[0][0]
            self.CC = token[3]
            self.value = float(token[4])


def x_matrix(list):  # defining the variables of x matrix since (Mx=b)

    x = []
    for element in list:

        if ('V_' + element.node1) not in x:  # appending values to the matrix without reccurance
            x.append('V_'+element.node1)

        if ('V_' + element.node2) not in x:
            x.append('V_'+element.node2)

        if element.type == 'V' or element.type == 'E' or element.type == 'H':

            x.append('I_'+element.name)

    return x


def M_b_matrix_DC(x_matrix, list):  # function to find the values of M and b matrix

    size = len(x_matrix)

    M = np.zeros((size, size), dtype=float)
    b = np.zeros((size), dtype=float)

    for elem in list:
        n1 = x_matrix.index("V_"+elem.node1)
        n2 = x_matrix.index("V_"+elem.node2)

        if elem.type == "R":
            M[n1, n1] += 1/elem.value
            M[n2, n2] += 1/elem.value
            M[n1, n2] -= 1/elem.value
            M[n2, n1] -= 1/elem.value

        if elem.type == "V":
            n_iv = x_matrix.index("I_"+elem.name)
            M[n1, n_iv] -= 1
            M[n2, n_iv] += 1
            M[n_iv, n1] += 1
            M[n_iv, n2] -= 1

            b[n_iv] += elem.value

        if elem.type == "I":
            b[n1] -= elem.value
            b[n2] += elem.value

        if elem.type == "L":
            M[n1, n1] += INFINITY/elem.value
            M[n2, n2] += INFINITY/elem.value
            M[n1, n2] -= INFINITY/elem.value
            M[n2, n1] -= INFINITY/elem.value

        if elem.type == "C":
            M[n1, n1] += ZERO*elem.value
            M[n2, n2] += ZERO*elem.value
            M[n1, n2] -= ZERO*elem.value
            M[n2, n1] -= ZERO*elem.value

        if elem.type == "G" or elem.type == "E":
            n_VC1 = x_matrix.index("V_"+elem.VC_node1)
            n_VC2 = x_matrix.index("V_"+elem.VC_node2)
            if elem.type == "G":
                M[n1, n_VC1] += elem.value
                M[n1, n_VC2] -= elem.value
                M[n2, n_VC1] -= elem.value
                M[n2, n_VC2] += elem.value
            else:
                n_current = x_matrix.index("I"+elem.name)
                M[n1, n_current] += 1
                M[n2, n_current] -= 1
                M[n_current, n1] += 1
                M[n_current, n2] -= 1
                M[n_current, n_VC1] -= elem.value
                M[n_current, n_VC2] += elem.value

        if elem.type == "F" or elem.type == "H":

            n_current = x_matrix.index("I_V"+elem.name)
            if elem.type == "F":
                M[n1, n_current] += elem.value
                M[n2, n_current] -= elem.value
            else:
                n_current2 = x_matrix.index("I_" + elem.name)
                M[n1, n_current2] += 1
                M[n2, n_current2] -= 1
                M[n_current2, n1] += 1
                M[n_current2, n2] -= 1
                M[n_current2, n_current] -= elem.value

    try:
        c = x_matrix.index("V_GND")
        M[c, c] += 1
        return M, b

    except Exception:
        print("Error No GND FOUND! Please change 0 to GND, or recheck file")
        exit()


def M_b_matrix_AC(x_matrix, list, freq):

    size = len(x_matrix)

    M = np.zeros((size, size), dtype=complex)
    b = np.zeros((size), dtype=complex)

    for elem in list:
        n1 = x_matrix.index("V_"+elem.node1)
        n2 = x_matrix.index("V_"+elem.node2)

        if elem.type == "R":
            M[n1, n1] += 1/elem.value
            M[n2, n2] += 1/elem.value
            M[n1, n2] -= 1/elem.value
            M[n2, n1] -= 1/elem.value

        if elem.type == "V":
            n_iv = x_matrix.index("I_"+elem.name)
            M[n1, n_iv] -= 1
            M[n2, n_iv] += 1
            M[n_iv, n1] += 1
            M[n_iv, n2] -= 1

            b[n_iv] += elem.value

        if elem.type == "I":
            b[n1] -= elem.value
            b[n2] += elem.value

        if elem.type == "L":
            M[n1, n1] += 1/(2*PI*freq*elem.value*(1j))
            M[n2, n2] += 1/(2*PI*freq*elem.value*(1j))
            M[n1, n2] -= 1/(2*PI*freq*elem.value*(1j))
            M[n2, n1] -= 1/(2*PI*freq*elem.value*(1j))

        if elem.type == "C":
            M[n1, n1] += (2*PI*freq*elem.value*(1j))
            M[n2, n2] += (2*PI*freq*elem.value*(1j))
            M[n1, n2] -= (2*PI*freq*elem.value*(1j))
            M[n2, n1] -= (2*PI*freq*elem.value*(1j))

        if elem.type == "G" or elem.type == "E":
            n_VC1 = x_matrix.index("V_"+elem.VC_node1)
            n_VC2 = x_matrix.index("V_"+elem.VC_node2)
            if elem.type == "G":
                M[n1, n_VC1] += elem.value
                M[n1, n_VC2] -= elem.value
                M[n2, n_VC1] -= elem.value
                M[n2, n_VC2] += elem.value
            else:
                n_current = x_matrix.index("I"+elem.name)
                M[n1, n_current] += 1
                M[n2, n_current] -= 1
                M[n_current, n1] += 1
                M[n_current, n2] -= 1
                M[n_current, n_VC1] -= elem.value
                M[n_current, n_VC2] += elem.value

        if elem.type == "F" or elem.type == "H":

            n_current = x.index("I_V"+elem.name)
            if elem.type == "F":
                M[n1, n_current] += elem.value
                M[n2, n_current] -= elem.value
            else:
                n_current2 = x.index("I_" + elem.name)
                M[n1, n_current2] += 1
                M[n2, n_current2] -= 1
                M[n_current2, n1] += 1
                M[n_current2, n2] -= 1
                M[n_current2, n_current] -= elem.value

    try:
        c = x_matrix.index("V_GND")
        M[c, c] += 1
        return M, b

    except Exception:
        print("Error No GND FOUND! Please change 0 to GND, or recheck file")
        exit()


try:
    len(sys.argv) == 2 and ".netlist" in sys.argv
    File = sys.argv[1]
except Exception:
    print("Invalid file")
    exit()


f = open(File, "r")
list = []
for line in f.readlines():
    a = line.split("#")[0].split("\n")[0]
    list.append(a)

# print(list)

DC = True
freq = 0

for line in list:

    tokens = line.split()
    if tokens[0] == '.ac':
        DC = False
        freq = float(str(tokens[2]))


try:
    circuit_start = list.index(CIRCUIT)
    circuit_end = list.index(END)
    circuit_start < circuit_end

    req_list = list[circuit_start+1:circuit_end]

    element_list = []

    for line in req_list:
        object = componenets(line)
        element_list.append(object)

        if object.type == "H" or object.type == "F":  # adding a dummy 0V component into the object list if CCXS
            for component in element_list:
                if component.name == object.CC:
                    location = element_list.index(component)
                    element_list.remove(component)

                    n1_new = component.node1
                    n_pseudo = "n" + object.name

                    component.node1 = n_pseudo
                    element_list.insert(location, component)
                    new_line = "V"+object.name+" " + n1_new+" "+n_pseudo+" 0"
                    element_list.append(componenets(new_line))

    x = x_matrix(element_list)

    if DC:  # solving and printing values for dc
        M, b = M_b_matrix_DC(x, element_list)

        sol = (np.linalg.solve(M, b))

        for i in range(len(x)):

            if(x[i][0] == "V"):
                print("Voltage at node " +
                      x[i][((x[i]).index("_") + 1):] + " : ", end=" ")
                print(round(float(sol[i]), 4), end=" V\n")

            elif(x[i][0] == "I"):
                print("Current through Voltage source " +
                      x[i][((x[i]).index("_") + 1):] + " : ", end=" ")
                print(round(float(sol[i]), 4), end=" A\n")

    else:  # solving and printing values for ac

        M, b = M_b_matrix_AC(x, element_list, freq)

        sol = (np.linalg.solve(M, b))
        for i in range(len(x)):

            if(x[i][0] == "V"):
                print("Voltage at node " +
                      x[i][((x[i]).index("_") + 1):] + " : ", end=" ")
                print(round((sol[i].real), 5) +
                      (round((sol[i].imag), 5))*(1j), end=" V\n")

            elif(x[i][0] == "I"):
                print("Current through Voltage source " +
                      x[i][((x[i]).index("_") + 1):] + " : ", end=" ")
                print(round(sol[i], 2), end=" A\n")


except Exception:
    print("Invalid definition")

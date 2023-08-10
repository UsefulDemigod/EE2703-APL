"""
        			EE2703:Applied Programming Lab
      				    Assignment 1: Solution
Name  :Harisankar K J
RollNo:EE20B043
"""
from sys import argv, exitck
# assigning constant variables for later comparison
CIRCUIT='.circuit'
END='.end'
# validating the number of arguments
if len(argv)!=2:
    print('\nUsage: %s <inputfile>' %argv[0])
    exit()
# validating the filename
try:
    # opening And Reading The File
    with open(argv[1]) as f:
        lines=f.readlines()
        start=-1
        end=-2
        # locating the start and end of the circuit by checking for ".cicuit" and ".end"
        for line in lines:
            if CIRCUIT==line[:len(CIRCUIT)]:
                start=lines.index(line)
            elif END==line[:len(END)]:
                end=lines.index(line)
                break
        # validating the contents in the netlist file (checking if .cicuit and .end placement)
        if start>=end or start<0 or end<0:
            print('Invalid Circuit Definition')
            exit(0)
        # transverseing through the netlist from the end to the start and printing the contents in reverse order
        while end-1>start:
            '''
            removing blank spaces
            using the '#' as a reference to remove the comments
            each element in the list is seperated by a space for ifentification
            '''
            line1=lines[end-1].split('#')[0].split()
            # reversing the contents in the received line
            line2=reversed(line1)
            # joining the contents of list uding space
            line3=' '.join(line2)
            # Printing The Final Line
            print(line3)
            end-=1
        # file closed
        f.close()
# error message for filename
except IOError:
    print('Invalid File')
    exit()
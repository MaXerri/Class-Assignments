Mario Xerri, max3 
___________________________________________________________________
GATE USAGE: 

Add32 Gate usage - There are 5 1 data bit gates used in the one bit adder, the 4 bit adder uses 4 of thse in addition to a gate to determine V.  This is 5x4 + 1 = 21.  There are 8 4 bit adders in the 32 bit adder and thus 168 gates in the 32 bit adder.  
= 168

LeftShift32: the 1,2,4,8, and 16 bit shifters fo not have gates used.  But the 32bitLeftShifter has 5 muxes that take in 32 bit signals. Each mux has 2 32 bit inputs and thus is 32*(2+1) = 96 gates and there are 5 muxes, so there are 96*5 = 480 gates 
= 480

Shift32: There are 2 muxes which select the flipped input and output or the nonflipped input and output to/from LeftShift32.  Thus there are 96 gates for each of these muxes.  So the total is 96*2 + 480 = 672
= 672

isAllZeros (I made this very simple component to check if all 32 bits in a number are 0): There is one NOR gate with 32 one bit inputs, so i think this counts as a single gate 
Total = 1

ALU32: there are 4 32 bit gates which is 128 gates,
       there are 5 1 bit gates which is 5 gates,
       there are 7 32 bit 2 input muxes which is 7 * 96 gates = 672 gates,
       there is 1 32 bit 4 input mux which is 32*(4+1) = 160 gates 
       there is 168 gates from add32
       there is 672 from shift32
       there is 1 from isAllZeros

TOTAL over entire ALU: 128 + 5 + 672 + 160 + 168 + 672  + 1 = 1806 gates
_______________________________________________________________________________________________
CRITICAL PATH: 

This likely occurs somewhere in the adder due to the culmination of all of the 1 bit adders.
For a one bit adder, the longest path from input to output is 3 as one of the paths from A to Cout is 3.  
For the four bit adder, there are 4* 3 units of delay plus the delay from the XOR gate determining V, thus 13 delay units
For the 32 bit adder, there are then 8*13=104 units of delay at max.

Max delay units from input to Add32 is 3 becasue there is a mux that decides if the input to B is inverted or not for subtraction and a one bit and gate which dictates the select bit for that mux.  
From Add32 output to C and V, the max delay units is for performing the EQ NEQ opertaions becasue the signal has to go thourgh the isAllZeros circuit which is 1 dela unit as it has one NOR gate and then goe through an XNOR gate to either perform NEQ or EQ which is another 1 unit of delay.  Then, it does through the mux that filters signal based on the 2nd LSB and then the mux that filters on the 2 MSBs. In total from the Add32 output to C that is an extra 6 units of delay.  

TOTAL Delay Perfomring EQ and NEQ = 3 (Pre Add32) + 104 (Add32) + 6(Post Add32) = 113 units of delay
